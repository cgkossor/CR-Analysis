"""Design diagnostics, run before any response is modelled (AC3).

Everything here is a property of *where the experiments sit*, not of what they
measured. That separation is the point: it tells a formulator which regions of
the design space predictions can be trusted in, before any model has had a
chance to look good.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay

from pipeline.design.matrix import (
    ModelSpec,
    build_model_matrix,
    term_names,
)


@dataclass(frozen=True)
class AliasEntry:
    """One inestimable term and what it is aliased with."""

    term: str
    aliased_with: dict[str, float]
    exact: bool

    def describe(self) -> str:
        parts = " + ".join(
            f"{coef:+.4g}*{name}" for name, coef in self.aliased_with.items() if abs(coef) > 1e-8
        )
        kind = "EXACTLY aliased" if self.exact else "aliased"
        return f"{self.term} is {kind} with {parts or '(nothing estimable)'}"


@dataclass(frozen=True)
class DesignDiagnostics:
    """Full AC3 characterisation of one design under one model."""

    model_label: str
    n_points: int
    n_terms: int
    rank: int
    estimable: bool
    condition_number: float
    vif: dict[str, float]
    leverage: tuple[float, ...]
    max_leverage: float
    sum_leverage: float
    d_efficiency: float
    a_efficiency: float
    g_efficiency: float
    fds_spv: tuple[float, ...]
    fds_median_spv: float
    aliases: tuple[AliasEntry, ...] = field(default=())
    notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class LackOfFitEstimability:
    """Whether a lack-of-fit test can be run, and what its error term means."""

    estimable_at_replicate_level: bool
    pure_error_df_replicate: int
    estimable_at_design_point_level: bool
    pure_error_df_design_point: int
    n_distinct_points: int
    n_observations: int
    caveat: str

    def describe(self) -> str:
        if not self.estimable_at_replicate_level:
            return (
                "Lack of fit is NOT estimable: the design has no replicated points and no "
                "centre points, so there is no pure-error term. No F test is reported, "
                "because a lack-of-fit F without pure error is meaningless."
            )
        return (
            f"Lack of fit is estimable against replicate pure error "
            f"({self.pure_error_df_replicate} df). {self.caveat}"
        )


def _safe_inverse(xtx: np.ndarray) -> np.ndarray | None:
    try:
        return np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None


def variance_inflation(matrix: np.ndarray, names: list[str]) -> dict[str, float]:
    """VIF for each column, by regressing it on the remaining columns.

    Infinite VIF marks an exact linear dependency -- which is what a constant-sum
    composition produces, and the reason a naive three-component RSM cannot be
    fitted at all.
    """
    out: dict[str, float] = {}
    n_cols = matrix.shape[1]
    for j in range(n_cols):
        target = matrix[:, j]
        others = np.delete(matrix, j, axis=1)
        if others.shape[1] == 0 or float(np.ptp(target)) == 0.0:
            out[names[j]] = float("nan")
            continue
        coef, _, _, _ = np.linalg.lstsq(others, target, rcond=None)
        ss_res = float(np.sum((target - others @ coef) ** 2))
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        if ss_tot <= 0:
            out[names[j]] = float("nan")
        elif ss_res <= 1e-20 * max(ss_tot, 1.0):
            # Exact dependency: the column is perfectly reproduced by the others.
            out[names[j]] = float("inf")
        else:
            # VIF = 1/(1 - R^2) with R^2 = 1 - ss_res/ss_tot, i.e. ss_tot/ss_res.
            out[names[j]] = ss_tot / ss_res
    return out


def alias_structure(
    design_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    fitted_names: list[str],
    candidate_names: list[str],
    tol: float = 1e-8,
) -> tuple[AliasEntry, ...]:
    """Alias matrix ``A = (X1'X1)^-1 X1'X2`` for candidate terms not in the model.

    Each column of ``A`` says how a term absent from the fitted model would bias
    the coefficients that *are* fitted, if that term were real.
    """
    xtx = design_matrix.T @ design_matrix
    inv = _safe_inverse(xtx)
    if inv is None:
        inv = np.linalg.pinv(xtx)

    alias = inv @ design_matrix.T @ candidate_matrix
    entries: list[AliasEntry] = []
    for j, cname in enumerate(candidate_names):
        column = alias[:, j]
        loading = {fitted_names[i]: float(column[i]) for i in range(len(fitted_names))}
        reconstructed = design_matrix @ column
        residual = float(np.max(np.abs(reconstructed - candidate_matrix[:, j])))
        scale = max(float(np.max(np.abs(candidate_matrix[:, j]))), 1.0)
        entries.append(
            AliasEntry(term=cname, aliased_with=loading, exact=residual <= tol * scale)
        )
    return tuple(entries)


def _hull_sample(points_2d: np.ndarray, grid: int = 60) -> np.ndarray:
    """Deterministic grid of points inside the convex hull of ``points_2d``.

    The design region for prediction is the hull of the *tested* compositions,
    not the whole simplex: outside it, G3 forbids silent extrapolation, so
    prediction variance there is not a meaningful thing to average over.
    """
    tri = Delaunay(points_2d)
    lo = points_2d.min(axis=0)
    hi = points_2d.max(axis=0)
    axes = [np.linspace(lo[k], hi[k], grid) for k in range(points_2d.shape[1])]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, points_2d.shape[1])
    inside = tri.find_simplex(mesh) >= 0
    return mesh[inside]


def fraction_of_design_space(
    design_matrix: np.ndarray,
    composition: np.ndarray,
    process_levels: np.ndarray,
    spec: ModelSpec,
    grid: int = 40,
) -> np.ndarray:
    """Scaled prediction variance across the design region, sorted ascending.

    ``SPV = n * x0' (X'X)^-1 x0``. The FDS curve is this sorted, plotted against
    the fraction of the region at or below each value.
    """
    xtx = design_matrix.T @ design_matrix
    inv = _safe_inverse(xtx)
    if inv is None:
        return np.array([])

    total = float(np.sum(composition[0]))
    hull_pts = _hull_sample(composition[:, :2], grid=grid)
    if hull_pts.size == 0:
        return np.array([])
    third = total - hull_pts.sum(axis=1)
    keep = third >= -1e-9
    hull_pts = hull_pts[keep]
    third = np.clip(third[keep], 0.0, None)
    comp_grid = np.column_stack([hull_pts, third])

    lo, hi = float(np.min(process_levels)), float(np.max(process_levels))
    proc_grid = np.linspace(lo, hi, max(int(grid / 4), 3))

    n = design_matrix.shape[0]
    spv: list[float] = []
    for v in proc_grid:
        block = build_model_matrix(comp_grid, np.full(len(comp_grid), v), spec)
        d = np.einsum("ij,jk,ik->i", block, inv, block)
        spv.extend((n * d).tolist())
    return np.sort(np.asarray(spv, dtype=float))


def evaluate_design(
    composition: np.ndarray,
    process: np.ndarray,
    spec: ModelSpec,
    candidate_spec: ModelSpec | None = None,
) -> DesignDiagnostics:
    """Characterise a design under ``spec``, optionally aliasing against a richer model."""
    matrix = build_model_matrix(composition, process, spec)
    names = term_names(spec)
    n, p = matrix.shape

    xtx = matrix.T @ matrix
    rank = int(np.linalg.matrix_rank(xtx))
    estimable = rank == p

    svals = np.linalg.svd(matrix, compute_uv=False)
    cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else float("inf")

    notes: list[str] = []
    if not estimable:
        notes.append(
            f"Model is NOT estimable: {p} terms but rank {rank}. "
            f"{p - rank} exact linear dependenc(ies) among the columns."
        )
    if n < p:
        notes.append(
            f"Only {n} design points for {p} terms: the model is saturated or "
            "over-parameterised and cannot be fitted with any residual degrees of freedom."
        )

    inv = _safe_inverse(xtx)
    if inv is None:
        leverage = np.full(n, float("nan"))
        d_eff = a_eff = g_eff = float("nan")
        spv = np.array([])
    else:
        leverage = np.einsum("ij,jk,ik->i", matrix, inv, matrix)
        sign, logdet = np.linalg.slogdet(xtx)
        d_eff = float(100.0 * np.exp(logdet / p) / n) if sign > 0 else float("nan")
        trace_inv = float(np.trace(inv))
        a_eff = float(100.0 * p / (n * trace_inv)) if trace_inv > 0 else float("nan")
        spv = fraction_of_design_space(matrix, composition, process, spec)
        g_eff = float(100.0 * p / np.max(spv)) if spv.size else float("nan")

    aliases: tuple[AliasEntry, ...] = ()
    if candidate_spec is not None:
        cand = build_model_matrix(composition, process, candidate_spec)
        cand_names = term_names(candidate_spec)
        extra = [i for i, nm in enumerate(cand_names) if nm not in names]
        if extra:
            aliases = alias_structure(
                matrix, cand[:, extra], names, [cand_names[i] for i in extra]
            )

    return DesignDiagnostics(
        model_label=spec.label,
        n_points=n,
        n_terms=p,
        rank=rank,
        estimable=estimable,
        condition_number=cond,
        vif=variance_inflation(matrix, names),
        leverage=tuple(float(v) for v in leverage),
        max_leverage=float(np.nanmax(leverage)) if leverage.size else float("nan"),
        sum_leverage=float(np.nansum(leverage)) if leverage.size else float("nan"),
        d_efficiency=d_eff,
        a_efficiency=a_eff,
        g_efficiency=g_eff,
        fds_spv=tuple(float(v) for v in spv),
        fds_median_spv=float(np.median(spv)) if spv.size else float("nan"),
        aliases=aliases,
        notes=tuple(notes),
    )


def lack_of_fit_estimability(
    design_keys: list[tuple[object, ...]], n_model_terms: int
) -> LackOfFitEstimability:
    """Determine whether a lack-of-fit F test has a pure-error term to test against.

    ``design_keys`` is one key per *observation* identifying its design point.
    Repeated keys are replicates.

    The distinction this function exists to preserve: replicates drawn from a
    single compression batch estimate vessel-to-vessel and analytical variation
    only. They do not estimate batch-to-batch variation, so an F test built on
    them has a denominator that is too small and will over-declare lack of fit.
    The test is still reported -- with that caveat attached, never as a bare
    p-value.
    """
    n_obs = len(design_keys)
    distinct = len(set(design_keys))
    pure_df_rep = n_obs - distinct
    lof_df = distinct - n_model_terms

    caveat = (
        "The replicates are vessels from one compression batch, so this pure error "
        "captures tablet-to-tablet and analytical variation but NOT batch-to-batch "
        "variation. The denominator is therefore too small and the test is "
        "anti-conservative: it will over-declare lack of fit even for an adequate "
        "model. Read it as lack of fit relative to within-batch analytical error."
    )

    return LackOfFitEstimability(
        estimable_at_replicate_level=pure_df_rep > 0 and lof_df > 0,
        pure_error_df_replicate=pure_df_rep,
        estimable_at_design_point_level=False,
        pure_error_df_design_point=0,
        n_distinct_points=distinct,
        n_observations=n_obs,
        caveat=caveat,
    )
