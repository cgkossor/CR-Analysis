"""Leave-one-formulation-out cross-validation (AC6).

The unit left out is the **formulation**, meaning every replicate of a
(case, grade) design point leaves together. Splitting replicates across the
train/test boundary would leak near-identical profiles into training and inflate
apparent accuracy -- and since this number is what G6 attaches to every
prediction on the dashboard, a leaked CV figure is a guardrail violation wearing
the costume of compliance.

Error is reported in two spaces, because they answer different questions:

* **parameter space** -- how well the surface predicts Weibull parameters;
* **profile space** -- what that means in % released, plus f2 between the
  predicted and observed curves, which is the quantity a formulator actually
  reasons about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.design.matrix import ModelSpec, build_model_matrix, term_names
from pipeline.equivalence.f2 import similarity_f2
from pipeline.profiles.fits import weibull
from pipeline.profiles.grid import paired_finite


@dataclass(frozen=True)
class FoldResult:
    """Prediction for one held-out formulation."""

    case: int
    grade: str
    observed: dict[str, float]
    predicted: dict[str, float]
    profile_rmse_pct: float
    f2: float
    f2_valid: bool
    profile_note: str = ""
    n_points_compared: int = 0
    in_hull: bool = True


@dataclass(frozen=True)
class CrossValidation:
    """Leave-one-formulation-out results across the whole design."""

    responses: tuple[str, ...]
    folds: tuple[FoldResult, ...]
    rmse_by_response: dict[str, float]
    mae_by_response: dict[str, float]
    profile_rmse_pct: float
    profile_rmse_pct_worst: float
    median_f2: float
    n_folds: int
    notes: tuple[str, ...] = field(default=())

    def fold_for(self, case: int, grade: str) -> FoldResult | None:
        for fold in self.folds:
            if fold.case == case and fold.grade == grade:
                return fold
        return None


def leave_one_formulation_out(
    composition: np.ndarray,
    process: np.ndarray,
    responses: dict[str, np.ndarray],
    spec: ModelSpec,
    *,
    kept_terms: list[str] | None = None,
    time_grid: np.ndarray,
    observed_profiles: dict[tuple[int, str], np.ndarray],
    case_ids: np.ndarray,
    grades: np.ndarray,
) -> CrossValidation:
    """Refit the surface with each formulation held out, then predict it.

    ``responses`` maps a response name to its per-design-point values. Weibull
    reconstruction requires ``log10_td``, ``weibull_beta`` and ``weibull_f_inf``
    to be present; without them profile-space error is skipped rather than
    approximated.
    """
    comp = np.asarray(composition, dtype=float)
    proc = np.asarray(process, dtype=float)
    n = comp.shape[0]

    all_labels = term_names(spec)
    keep_idx = (
        list(range(len(all_labels)))
        if kept_terms is None
        else [i for i, name in enumerate(all_labels) if name in set(kept_terms)]
    )

    full_matrix = build_model_matrix(comp, proc, spec)[:, keep_idx]
    notes: list[str] = []

    can_rebuild = all(
        key in responses for key in ("log10_td", "weibull_beta", "weibull_f_inf")
    )
    if not can_rebuild:
        notes.append(
            "Profile-space error not computed: the Weibull parameter set "
            "(log10_td, weibull_beta, weibull_f_inf) is not fully present."
        )

    folds: list[FoldResult] = []
    errors: dict[str, list[float]] = {k: [] for k in responses}

    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        x_train = full_matrix[train]
        x_test = full_matrix[i : i + 1]

        if np.linalg.matrix_rank(x_train) < x_train.shape[1]:
            notes.append(
                f"Fold {i} skipped: removing design point (case {int(case_ids[i])}, "
                f"{grades[i]}) makes the model inestimable. The design has no slack "
                "for this term set."
            )
            continue

        observed: dict[str, float] = {}
        predicted: dict[str, float] = {}
        for name, values in responses.items():
            y = np.asarray(values, dtype=float)
            beta, *_ = np.linalg.lstsq(x_train, y[train], rcond=None)
            pred = float((x_test @ beta)[0])
            observed[name] = float(y[i])
            predicted[name] = pred
            if np.isfinite(pred) and np.isfinite(y[i]):
                errors[name].append(pred - y[i])

        rmse_pct = float("nan")
        f2_value = float("nan")
        f2_ok = False
        note = ""
        n_compared = 0
        key = (int(case_ids[i]), str(grades[i]))

        if can_rebuild and key in observed_profiles:
            td = 10.0 ** predicted["log10_td"]
            pred_curve = weibull(
                time_grid, predicted["weibull_f_inf"], td, predicted["weibull_beta"]
            )
            obs_curve = observed_profiles[key]
            if len(obs_curve) != len(time_grid):
                note = "observed profile length does not match the time grid"
            else:
                # Only where the formulation was actually measured. A profile
                # that stopped early, or whose sampling times did not reach a
                # grid point, has no value there -- and averaging over the gap
                # silently turns the whole fold into NaN, which is how this
                # number quietly became unreportable on ragged data.
                obs_pts, pred_pts = paired_finite(obs_curve, pred_curve)
                if obs_pts.size:
                    rmse_pct = float(np.sqrt(np.mean((pred_pts - obs_pts) ** 2)))
                    n_compared = int(obs_pts.size)
                    if n_compared < len(time_grid):
                        note = (
                            f"compared on {n_compared} of {len(time_grid)} grid points; "
                            "the rest lie outside this formulation's measured window"
                        )
                else:
                    note = (
                        "no grid point where both the observed and predicted profile "
                        "have a value; profile error is not computable for this fold"
                    )
                f2_result = similarity_f2(time_grid, obs_curve, pred_curve)
                f2_value = f2_result.value
                f2_ok = f2_result.valid
                if not f2_result.valid:
                    note = (note + "; " if note else "") + f2_result.note

        folds.append(
            FoldResult(
                case=int(case_ids[i]),
                grade=str(grades[i]),
                observed=observed,
                predicted=predicted,
                profile_rmse_pct=rmse_pct,
                f2=f2_value,
                f2_valid=f2_ok,
                profile_note=note,
                n_points_compared=n_compared,
            )
        )

    rmse_by = {
        name: float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
        for name, vals in errors.items()
    }
    mae_by = {
        name: float(np.mean(np.abs(vals))) if vals else float("nan")
        for name, vals in errors.items()
    }
    profile_errors = [f.profile_rmse_pct for f in folds if np.isfinite(f.profile_rmse_pct)]
    f2_values = [f.f2 for f in folds if f.f2_valid and np.isfinite(f.f2)]

    uncomputable = [f for f in folds if not np.isfinite(f.profile_rmse_pct)]
    if uncomputable:
        names = ", ".join(f"case {f.case}/{f.grade}" for f in uncomputable[:6])
        notes.append(
            f"{len(uncomputable)} of {len(folds)} fold(s) have no computable "
            f"profile-space error: {names}"
            + (" and others" if len(uncomputable) > 6 else "")
            + ". They are excluded from the summary RMSE, so that figure describes "
            f"only the {len(folds) - len(uncomputable)} fold(s) it could be computed on."
        )
    if folds and not profile_errors:
        notes.append(
            "Profile-space error could not be computed for ANY formulation. This "
            "usually means the profiles share no common measured timepoints, so "
            "predicted and observed curves never overlap. Check the timepoint section "
            "of the data-quality report; the summary error is reported as unavailable "
            "rather than as a number."
        )

    return CrossValidation(
        responses=tuple(responses),
        folds=tuple(folds),
        rmse_by_response=rmse_by,
        mae_by_response=mae_by,
        profile_rmse_pct=float(np.sqrt(np.mean(np.square(profile_errors))))
        if profile_errors
        else float("nan"),
        profile_rmse_pct_worst=float(np.max(profile_errors)) if profile_errors else float("nan"),
        median_f2=float(np.median(f2_values)) if f2_values else float("nan"),
        n_folds=len(folds),
        notes=tuple(notes),
    )
