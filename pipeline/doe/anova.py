"""Analysis of variance in the form a Minitab user expects.

Source / DF / Adj SS / Adj MS / F / P, plus the model summary line (S, R-sq,
R-sq(adj), R-sq(pred)).

Two things differ from a textbook factorial ANOVA and both matter:

**The model has no intercept.** On a mixture the Scheffe linear terms already
span the constant, so the sums of squares are taken about the mean explicitly
rather than assuming an intercept column carries it. Getting this wrong inflates
the model sum of squares by the grand total and makes every R-squared meaningless.

**Adjusted (Type III) sums of squares.** Each term is tested as though entered
last, so the answer does not depend on the order terms happen to appear in. With
a mixture design the columns are correlated by construction, so sequential sums
of squares would give a different answer per ordering -- which is not a property
anyone should have to reason about.

The fit is on design-point means. Vessel replicates are subsamples of one
compression batch, and treating them as independent runs would shrink every
standard error by roughly sqrt(r) and manufacture significance the design cannot
support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class AnovaRow:
    """One line of the ANOVA table."""

    source: str
    df: int
    adj_ss: float
    adj_ms: float
    f_value: float
    p_value: float
    is_group: bool = False

    @property
    def significant(self) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value < 0.05)


@dataclass(frozen=True)
class AnovaTable:
    """A complete ANOVA plus the model summary."""

    response: str
    rows: tuple[AnovaRow, ...]
    model_row: AnovaRow
    residual_df: int
    residual_ss: float
    residual_ms: float
    total_df: int
    total_ss: float

    s: float
    r_squared: float
    adj_r_squared: float
    pred_r_squared: float

    n_obs: int
    n_terms: int
    notes: tuple[str, ...] = field(default=())

    @property
    def significant_terms(self) -> tuple[AnovaRow, ...]:
        return tuple(r for r in self.rows if r.significant and not r.is_group)


def _sse(matrix: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of a reduced model, always able to fit the mean.

    A constant column is appended before fitting. This is not cosmetic. In a
    Scheffe model the linear blending terms carry the constant between them, so
    removing them all leaves a model that cannot represent even the grand mean --
    and the resulting "reduction in error" is measured against a model with no
    intercept at all, which produced an adjusted sum of squares larger than the
    total. Sums of squares here are taken about the mean, so every model they are
    compared against must at least contain it.

    When the constant is already spanned the extra column is redundant and
    ``lstsq`` handles the rank deficiency without changing the projection.
    """
    ones = np.ones((matrix.shape[0], 1))
    augmented = np.hstack([matrix, ones]) if matrix.shape[1] else ones
    beta, *_ = np.linalg.lstsq(augmented, y, rcond=None)
    return float(np.sum((y - augmented @ beta) ** 2))


def _group_of(term: str) -> str:
    """Collapse a model term into the source a formulator recognises.

    Individual Scheffe terms are the wrong unit for a summary table: nobody asks
    whether `lactose:v` matters, they ask whether the composition-by-grade
    interaction matters.
    """
    comp, _, proc = term.partition(":")
    order = "Interaction" if "*" in comp else "Linear"
    if comp == "intercept":
        order = "Constant"
    if not proc:
        return f"Composition ({order.lower()})"
    power = "quadratic" if proc == "v^2" else "linear"
    return f"Composition x grade ({order.lower()} x {power})"


def build(
    matrix: np.ndarray,
    response: np.ndarray,
    term_labels: list[str],
    *,
    response_name: str = "response",
    has_intercept: bool = False,
) -> AnovaTable:
    """Compute the ANOVA for a fitted model.

    ``has_intercept`` describes the parameterisation, not whether an intercept
    column is literally present: a Scheffe mixture model spans the constant
    through its linear terms even though no intercept column exists.
    """
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(response, dtype=float).reshape(-1)
    n, p = x.shape
    notes: list[str] = []

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    resid = y - fitted

    ss_error = float(resid @ resid)
    ss_total = float(np.sum((y - y.mean()) ** 2))
    ss_model = ss_total - ss_error

    # The model spans the constant either way -- through an explicit intercept
    # column, or through the Scheffe linear terms -- so it costs one degree of
    # freedom to the mean in both parameterisations.
    df_model = max(p - 1, 0)
    df_error = n - p
    df_total = n - 1

    if df_error <= 0:
        notes.append(
            f"Saturated model: {n} design points for {p} terms leaves no residual "
            "degrees of freedom, so no term can be tested."
        )
        ms_error = float("nan")
    else:
        ms_error = ss_error / df_error

    # --- Adjusted (Type III) sums of squares -------------------------------
    rows: list[AnovaRow] = []
    for j, name in enumerate(term_labels):
        reduced = np.delete(x, j, axis=1)
        adj_ss = _sse(reduced, y) - ss_error
        adj_ss = max(adj_ss, 0.0)
        if df_error > 0 and ms_error > 0:
            f_val = adj_ss / ms_error
            p_val = float(1.0 - stats.f.cdf(f_val, 1, df_error))
        else:
            f_val = float("nan")
            p_val = float("nan")
        rows.append(AnovaRow(name, 1, adj_ss, adj_ss, f_val, p_val))

    # --- Grouped sources ----------------------------------------------------
    groups: dict[str, list[int]] = {}
    for j, name in enumerate(term_labels):
        groups.setdefault(_group_of(name), []).append(j)

    group_rows: list[AnovaRow] = []
    for source, idx in groups.items():
        if len(idx) == len(term_labels):
            continue  # identical to the model row; nothing added
        if len(idx) >= p:
            continue
        reduced = np.delete(x, idx, axis=1)
        adj_ss = max(_sse(reduced, y) - ss_error, 0.0)
        df = len(idx)
        if df_error > 0 and ms_error > 0:
            f_val = (adj_ss / df) / ms_error
            p_val = float(1.0 - stats.f.cdf(f_val, df, df_error))
        else:
            f_val = float("nan")
            p_val = float("nan")
        group_rows.append(
            AnovaRow(source, df, adj_ss, adj_ss / df, f_val, p_val, is_group=True)
        )

    model_f = (
        (ss_model / df_model) / ms_error
        if df_model > 0 and df_error > 0 and ms_error > 0
        else float("nan")
    )
    model_p = (
        float(1.0 - stats.f.cdf(model_f, df_model, df_error))
        if np.isfinite(model_f)
        else float("nan")
    )
    model_row = AnovaRow(
        "Model", df_model, ss_model,
        ss_model / df_model if df_model else float("nan"),
        model_f, model_p, is_group=True,
    )

    # --- Summary line -------------------------------------------------------
    r2 = 1.0 - ss_error / ss_total if ss_total > 0 else float("nan")
    adj_r2 = (
        1.0 - (ss_error / df_error) / (ss_total / df_total)
        if df_error > 0 and df_total > 0 and ss_total > 0
        else float("nan")
    )

    # Predicted R-squared via the leave-one-out shortcut. Exact, and far cheaper
    # than refitting n times.
    try:
        xtx_inv = np.linalg.inv(x.T @ x)
        hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
        press_resid = resid / np.clip(1.0 - hat, 1e-12, None)
        press = float(press_resid @ press_resid)
        pred_r2 = 1.0 - press / ss_total if ss_total > 0 else float("nan")
    except np.linalg.LinAlgError:
        pred_r2 = float("nan")
        notes.append("Predicted R-squared unavailable: the model matrix is singular.")

    if n > 0 and p / n > 0.5:
        notes.append(
            f"{p} terms fitted to {n} design points. The model has more freedom than "
            "the design comfortably supports; R-sq(pred) is the number to judge it by, "
            "not R-sq."
        )
    if np.isfinite(adj_r2) and np.isfinite(pred_r2) and adj_r2 - pred_r2 > 0.2:
        notes.append(
            f"R-sq(adj) {adj_r2:.3f} exceeds R-sq(pred) {pred_r2:.3f} by more than 0.2: "
            "the model describes this data better than it predicts new data."
        )

    return AnovaTable(
        response=response_name,
        rows=tuple(group_rows + rows),
        model_row=model_row,
        residual_df=df_error,
        residual_ss=ss_error,
        residual_ms=ms_error,
        total_df=df_total,
        total_ss=ss_total,
        s=float(np.sqrt(ms_error)) if np.isfinite(ms_error) else float("nan"),
        r_squared=r2,
        adj_r_squared=adj_r2,
        pred_r_squared=pred_r2,
        n_obs=n,
        n_terms=p,
        notes=tuple(notes),
    )
