"""How few experiments would have reached the same conclusions? (AC8)

Candidate subsets are chosen by **design-efficiency criteria from AC3**, not at
random: a D-optimal exchange over the 33 available points for the model actually
being fitted. Random subsets would mostly answer "how bad is a bad design",
which nobody needs to know.

Three degradations are measured against the full design:

1. profile prediction error, refitting the surface on the subset;
2. equivalence-set agreement, since AC7's conclusions are the ones a formulator
   acts on;
3. directional agreement of the lever effects -- whether the subset still says
   the same thing about which lever does what.

Everything is deterministic: the exchange starts from a fixed seed (G10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import config
from pipeline.design.matrix import ModelSpec, build_model_matrix, term_names
from pipeline.profiles.fits import weibull


@dataclass(frozen=True)
class SubsetResult:
    """One candidate reduced design, evaluated against the full design."""

    size: int
    selected: tuple[tuple[int, str], ...]
    d_criterion: float
    estimable: bool
    profile_rmse_pct: float
    profile_rmse_worst_pct: float
    lever_direction_agrees: bool
    lever_max_abs_error: float
    equivalence_jaccard: float
    note: str = ""


@dataclass(frozen=True)
class StressTest:
    """The full AC8 result, plus the recommended point set."""

    results: tuple[SubsetResult, ...]
    full_profile_rmse_pct: float
    recommended_size: int
    recommended: SubsetResult | None
    breakdown_size: int
    rationale: str
    notes: tuple[str, ...] = field(default=())


def d_optimal_subset(
    matrix: np.ndarray, size: int, *, seed: int = config.PIPELINE_SEED
) -> list[int]:
    """Greedy Fedorov-style exchange maximising ``|X'X|`` for a subset of rows.

    Deterministic: the starting set is chosen by a seeded permutation and the
    exchange loop is exhaustive, so repeated runs return the same subset.
    """
    n, p = matrix.shape
    size = max(min(size, n), p)
    rng = np.random.default_rng(seed)
    current = list(rng.permutation(n)[:size])

    def logdet(rows: list[int]) -> float:
        sub = matrix[rows]
        sign, value = np.linalg.slogdet(sub.T @ sub)
        return value if sign > 0 else -np.inf

    best = logdet(current)
    improved = True
    while improved:
        improved = False
        outside = [i for i in range(n) if i not in current]
        for position in range(len(current)):
            for candidate in outside:
                trial = current.copy()
                trial[position] = candidate
                value = logdet(trial)
                if value > best + 1e-10:
                    current, best = trial, value
                    improved = True
                    break
            if improved:
                break
    return sorted(current)


def _fit_predict(
    matrix: np.ndarray, responses: dict[str, np.ndarray], rows: list[int]
) -> dict[str, np.ndarray] | None:
    sub = matrix[rows]
    if np.linalg.matrix_rank(sub) < sub.shape[1]:
        return None
    out: dict[str, np.ndarray] = {}
    for name, values in responses.items():
        beta, *_ = np.linalg.lstsq(sub, np.asarray(values)[rows], rcond=None)
        out[name] = matrix @ beta
    return out


def run_stress_test(
    composition: np.ndarray,
    process: np.ndarray,
    responses: dict[str, np.ndarray],
    spec: ModelSpec,
    kept_terms: list[str],
    *,
    time_grid: np.ndarray,
    observed_profiles: dict[tuple[int, str], np.ndarray],
    case_ids: np.ndarray,
    grades: np.ndarray,
    sizes: tuple[int, ...] | None = None,
    rmse_tolerance_pct: float = 2.0,
    min_residual_df: int = 5,
) -> StressTest:
    """Evaluate reduced designs and recommend a point set for a new API."""
    labels = term_names(spec)
    keep = [i for i, name in enumerate(labels) if name in set(kept_terms)]
    matrix = build_model_matrix(composition, process, spec)[:, keep]
    n, p = matrix.shape

    keys = [(int(case_ids[i]), str(grades[i])) for i in range(n)]

    def profile_error(pred: dict[str, np.ndarray]) -> tuple[float, float]:
        errors: list[float] = []
        for i, key in enumerate(keys):
            observed = observed_profiles.get(key)
            if observed is None:
                continue
            curve = weibull(
                time_grid,
                pred["weibull_f_inf"][i],
                10.0 ** pred["log10_td"][i],
                pred["weibull_beta"][i],
            )
            errors.append(float(np.sqrt(np.mean((curve - observed) ** 2))))
        if not errors:
            return float("nan"), float("nan")
        return float(np.sqrt(np.mean(np.square(errors)))), float(np.max(errors))

    full_pred = _fit_predict(matrix, responses, list(range(n)))
    if full_pred is None:
        return StressTest((), float("nan"), n, None, n, "full design is not estimable")
    full_rmse, _ = profile_error(full_pred)
    full_levers = _lever_directions(
        matrix, responses, list(range(n)), spec, keep, composition, process
    )

    candidate_sizes = sizes or tuple(range(p, n + 1, max((n - p) // 8, 1)))
    results: list[SubsetResult] = []

    for size in candidate_sizes:
        if size > n:
            continue
        rows = d_optimal_subset(matrix, size)
        pred = _fit_predict(matrix, responses, rows)
        if pred is None:
            results.append(
                SubsetResult(
                    size=size,
                    selected=tuple(keys[i] for i in rows),
                    d_criterion=float("nan"),
                    estimable=False,
                    profile_rmse_pct=float("nan"),
                    profile_rmse_worst_pct=float("nan"),
                    lever_direction_agrees=False,
                    lever_max_abs_error=float("nan"),
                    equivalence_jaccard=float("nan"),
                    note=f"{size} points cannot estimate the {p}-term model",
                )
            )
            continue

        rmse, worst = profile_error(pred)
        levers = _lever_directions(matrix, responses, rows, spec, keep, composition, process)
        agrees = bool(np.all(np.sign(levers) == np.sign(full_levers)))
        lever_err = float(np.max(np.abs(levers - full_levers)))

        sub = matrix[rows]
        sign, value = np.linalg.slogdet(sub.T @ sub)
        d_crit = float(np.exp(value / p) / size) if sign > 0 else float("nan")

        results.append(
            SubsetResult(
                size=size,
                selected=tuple(keys[i] for i in rows),
                d_criterion=d_crit,
                estimable=True,
                profile_rmse_pct=rmse,
                profile_rmse_worst_pct=worst,
                lever_direction_agrees=agrees,
                lever_max_abs_error=lever_err,
                equivalence_jaccard=_equivalence_agreement(
                    pred, full_pred, time_grid, keys
                ),
            )
        )

    # A design is only a candidate if it can estimate its own error. A saturated
    # subset (size == p) reproduces its training points exactly, has no residual
    # degrees of freedom, and makes leave-one-out cross-validation inestimable --
    # so it could never carry the CV error G6 requires beside a prediction. Low
    # apparent RMSE from such a design is an artefact of counting, not a result.
    min_size = p + min_residual_df
    usable = [
        r
        for r in results
        if r.estimable
        and r.size >= min_size
        and r.lever_direction_agrees
        and np.isfinite(r.profile_rmse_pct)
        and r.profile_rmse_pct <= full_rmse + rmse_tolerance_pct
    ]
    recommended = min(usable, key=lambda r: r.size) if usable else None
    failing = [r for r in results if r.estimable and not r.lever_direction_agrees]
    breakdown = max((r.size for r in failing), default=min_size)

    if recommended is not None:
        rationale = (
            f"{recommended.size} of {n} runs keeps profile prediction within "
            f"{rmse_tolerance_pct:.1f} percentage points of the full design "
            f"({recommended.profile_rmse_pct:.2f}% vs {full_rmse:.2f}%) and preserves "
            f"every directional conclusion about the levers, while retaining "
            f"{recommended.size - p} residual degrees of freedom so the design can "
            f"still estimate its own error. Designs below {min_size} runs are excluded "
            f"regardless of apparent accuracy: with {p} model terms they are saturated "
            "or nearly so, cannot support cross-validation, and therefore cannot carry "
            "the error estimate every prediction must be reported with."
        )
    else:
        rationale = (
            f"No reduced design of at least {min_size} runs met the tolerance while "
            "preserving directional conclusions. Run the full design."
        )

    return StressTest(
        results=tuple(results),
        full_profile_rmse_pct=full_rmse,
        recommended_size=recommended.size if recommended else n,
        recommended=recommended,
        breakdown_size=breakdown,
        rationale=rationale,
        notes=(
            "Subsets are D-optimal for the fitted model, selected by deterministic "
            "exchange (G10). Prediction error is measured against every measured "
            "profile, including those the subset did not see.",
            "A flat degradation curve means the fitted model has little lack of fit "
            "over this design: extra points confirm the surface rather than reshape "
            "it. Read the curve as a statement about THIS response and THIS model. A "
            "response with real curvature would degrade faster as points are removed, "
            "so the recommended size should be re-derived per response, not reused.",
        ),
    )


def _lever_directions(
    matrix: np.ndarray,
    responses: dict[str, np.ndarray],
    rows: list[int],
    spec: ModelSpec,
    keep: list[int],
    composition: np.ndarray,
    process: np.ndarray,
) -> np.ndarray:
    """Marginal effect of +10 pts HPMC on log10(Td), at each distinct grade."""
    sub = matrix[rows]
    beta, *_ = np.linalg.lstsq(sub, np.asarray(responses["log10_td"])[rows], rcond=None)

    centre = composition.mean(axis=0)
    step = np.array([0.0, 0.10, -0.10])
    effects: list[float] = []
    for v in np.unique(process):
        pair = np.vstack([centre, centre + step])
        design = build_model_matrix(pair, np.array([v, v]), spec)[:, keep]
        pred = design @ beta
        effects.append(float(pred[1] - pred[0]))
    return np.asarray(effects)


def _equivalence_agreement(
    pred: dict[str, np.ndarray],
    full: dict[str, np.ndarray],
    time_grid: np.ndarray,
    keys: list[tuple[int, str]],
) -> float:
    """Jaccard agreement of the f2>=50 neighbour sets under subset vs full fits."""
    from pipeline.equivalence.f2 import similarity_f2

    def curves(source: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [
            weibull(
                time_grid,
                source["weibull_f_inf"][i],
                10.0 ** source["log10_td"][i],
                source["weibull_beta"][i],
            )
            for i in range(len(keys))
        ]

    a_curves, b_curves = curves(pred), curves(full)
    scores: list[float] = []
    for i in range(len(keys)):
        set_a = {
            j
            for j in range(len(keys))
            if similarity_f2(time_grid, a_curves[i], a_curves[j]).similar
        }
        set_b = {
            j
            for j in range(len(keys))
            if similarity_f2(time_grid, b_curves[i], b_curves[j]).similar
        }
        union = set_a | set_b
        scores.append(len(set_a & set_b) / len(union) if union else 1.0)
    return float(np.mean(scores)) if scores else float("nan")
