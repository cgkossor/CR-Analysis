"""Least-squares response surfaces with the diagnostics AC4 requires.

Fitted on **design-point means**, not on individual replicate profiles. Vessel
replicates from one compression batch are subsamples; treating them as
independent runs would shrink every standard error by roughly ``sqrt(r)`` and
manufacture significance the design cannot support.

Implemented directly on the normal equations rather than through a formula API,
because the mixture parameterisation has no intercept and several conventional
quantities (R^2, adequate precision, sequential sums of squares) need to be
computed with that in mind rather than inherited from an assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from pipeline import config


@dataclass(frozen=True)
class Coefficient:
    """One estimated coefficient with its inference."""

    name: str
    estimate: float
    std_error: float
    t_value: float
    p_value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class SurfaceFit:
    """A fitted response surface and everything needed to judge it."""

    response: str
    model_label: str
    n_obs: int
    n_terms: int
    df_residual: int
    coefficients: tuple[Coefficient, ...]

    r_squared: float
    adj_r_squared: float
    pred_r_squared: float
    press: float
    r2_gap_flagged: bool

    rmse: float
    adequate_precision: float
    adequate_precision_flagged: bool

    fitted: tuple[float, ...]
    residuals: tuple[float, ...]
    studentised: tuple[float, ...]
    leverage: tuple[float, ...]
    cooks_distance: tuple[float, ...]
    dffits: tuple[float, ...]

    estimable: bool
    notes: tuple[str, ...] = field(default=())

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        beta = np.array([c.estimate for c in self.coefficients])
        return np.asarray(matrix, dtype=float) @ beta


def fit_surface(
    matrix: np.ndarray,
    response: np.ndarray,
    term_labels: list[str],
    *,
    response_name: str = "response",
    model_label: str = "model",
    alpha: float = 0.05,
) -> SurfaceFit:
    """Ordinary least squares with AC4's diagnostic suite."""
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(response, dtype=float).reshape(-1)
    n, p = x.shape

    notes: list[str] = []
    rank = int(np.linalg.matrix_rank(x))
    if rank < p:
        notes.append(
            f"Model is not estimable: {p} terms, rank {rank}. Reduce the model before "
            "interpreting any coefficient."
        )
        return _inestimable(response_name, model_label, n, p, term_labels, notes)
    if n <= p:
        notes.append(
            f"Saturated or over-parameterised: {n} design points for {p} terms, leaving "
            f"{n - p} residual degrees of freedom. No error estimate is possible."
        )
        return _inestimable(response_name, model_label, n, p, term_labels, notes)

    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    fitted = x @ beta
    resid = y - fitted
    df_res = n - p

    sse = float(resid @ resid)
    mse = sse / df_res
    sst = float(np.sum((y - y.mean()) ** 2))

    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_res if sst > 0 else float("nan")

    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    # PRESS via the leave-one-out shortcut: exact, and far cheaper than refitting.
    denom = np.clip(1.0 - hat, 1e-12, None)
    press_resid = resid / denom
    press = float(press_resid @ press_resid)
    pred_r2 = 1.0 - press / sst if sst > 0 else float("nan")

    se = np.sqrt(np.diag(xtx_inv) * mse)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_res))
    coefficients = tuple(
        Coefficient(
            name=term_labels[j],
            estimate=float(beta[j]),
            std_error=float(se[j]),
            t_value=float(beta[j] / se[j]) if se[j] > 0 else float("nan"),
            p_value=(
                float(2.0 * (1.0 - stats.t.cdf(abs(beta[j] / se[j]), df_res)))
                if se[j] > 0
                else float("nan")
            ),
            ci_low=float(beta[j] - t_crit * se[j]),
            ci_high=float(beta[j] + t_crit * se[j]),
        )
        for j in range(p)
    )

    # Adequate precision: signal-to-noise over the design points. Below 4 the
    # surface cannot navigate the design space (AC4).
    signal = float(np.max(fitted) - np.min(fitted))
    noise = float(np.sqrt(p * mse / n))
    adeq = signal / noise if noise > 0 else float("inf")

    studentised = _externally_studentised(resid, hat, sse, n, p)
    cooks = (resid**2 / (p * mse)) * (hat / np.clip((1.0 - hat) ** 2, 1e-12, None))
    dffits = studentised * np.sqrt(hat / np.clip(1.0 - hat, 1e-12, None))

    gap_flag = bool(np.isfinite(adj_r2) and np.isfinite(pred_r2)
                    and (adj_r2 - pred_r2) > config.R2_GAP_FLAG)
    if gap_flag:
        notes.append(
            f"Adjusted R^2 ({adj_r2:.3f}) exceeds predicted R^2 ({pred_r2:.3f}) by more "
            f"than {config.R2_GAP_FLAG:.2f}. The model explains this data better than it "
            "predicts new data -- a sign of over-fitting, not of a good surface."
        )
    adeq_flag = bool(adeq < config.ADEQUATE_PRECISION_FLAG)
    if adeq_flag:
        notes.append(
            f"Adequate precision {adeq:.2f} is below {config.ADEQUATE_PRECISION_FLAG:g}: "
            "the modelled signal is not large enough against its own noise to navigate "
            "the design space. Do not use this surface for optimisation."
        )

    return SurfaceFit(
        response=response_name,
        model_label=model_label,
        n_obs=n,
        n_terms=p,
        df_residual=df_res,
        coefficients=coefficients,
        r_squared=r2,
        adj_r_squared=adj_r2,
        pred_r_squared=pred_r2,
        press=press,
        r2_gap_flagged=gap_flag,
        rmse=float(np.sqrt(mse)),
        adequate_precision=adeq,
        adequate_precision_flagged=adeq_flag,
        fitted=tuple(float(v) for v in fitted),
        residuals=tuple(float(v) for v in resid),
        studentised=tuple(float(v) for v in studentised),
        leverage=tuple(float(v) for v in hat),
        cooks_distance=tuple(float(v) for v in cooks),
        dffits=tuple(float(v) for v in dffits),
        estimable=True,
        notes=tuple(notes),
    )


def _externally_studentised(
    resid: np.ndarray, hat: np.ndarray, sse: float, n: int, p: int
) -> np.ndarray:
    """Residuals scaled by an error estimate that excludes the point itself."""
    df = n - p - 1
    if df <= 0:
        return np.full_like(resid, np.nan)
    one_minus_h = np.clip(1.0 - hat, 1e-12, None)
    s_sq_i = (sse - resid**2 / one_minus_h) / df
    s_sq_i = np.clip(s_sq_i, 1e-300, None)
    return resid / np.sqrt(s_sq_i * one_minus_h)


def _inestimable(
    response: str, label: str, n: int, p: int, terms: list[str], notes: list[str]
) -> SurfaceFit:
    nan_tuple = tuple(float("nan") for _ in range(n))
    return SurfaceFit(
        response=response,
        model_label=label,
        n_obs=n,
        n_terms=p,
        df_residual=n - p,
        coefficients=tuple(
            Coefficient(t, *(float("nan"),) * 6) for t in terms
        ),
        r_squared=float("nan"),
        adj_r_squared=float("nan"),
        pred_r_squared=float("nan"),
        press=float("nan"),
        r2_gap_flagged=False,
        rmse=float("nan"),
        adequate_precision=float("nan"),
        adequate_precision_flagged=False,
        fitted=nan_tuple,
        residuals=nan_tuple,
        studentised=nan_tuple,
        leverage=nan_tuple,
        cooks_distance=nan_tuple,
        dffits=nan_tuple,
        estimable=False,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class SequentialTerm:
    """One step of the sequential model sum of squares."""

    model: str
    n_terms: int
    added_terms: int
    sum_squares: float
    df: int
    f_value: float
    p_value: float
    recommended: bool


def sequential_sum_of_squares(
    matrices: dict[str, np.ndarray], response: np.ndarray, order: list[str]
) -> tuple[SequentialTerm, ...]:
    """Incremental F tests as terms are added, linear -> 2FI -> quadratic (AC4).

    Each step is tested against the residual of the *richest* model in ``order``,
    which is the conventional sequential-SS construction.
    """
    y = np.asarray(response, dtype=float).reshape(-1)
    n = len(y)

    richest = np.asarray(matrices[order[-1]], dtype=float)
    p_rich = richest.shape[1]
    if n <= p_rich:
        return ()
    beta_rich = np.linalg.lstsq(richest, y, rcond=None)[0]
    sse_rich = float(np.sum((y - richest @ beta_rich) ** 2))
    df_rich = n - p_rich
    mse_rich = sse_rich / df_rich

    steps: list[SequentialTerm] = []
    prev_sse = float(np.sum((y - y.mean()) ** 2))
    prev_p = 1

    for name in order:
        x = np.asarray(matrices[name], dtype=float)
        p = x.shape[1]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        sse = float(np.sum((y - x @ beta) ** 2))
        added = p - prev_p
        ss = prev_sse - sse
        if added <= 0 or mse_rich <= 0:
            f_val = float("nan")
            p_val = float("nan")
        else:
            f_val = (ss / added) / mse_rich
            p_val = float(1.0 - stats.f.cdf(f_val, added, df_rich)) if f_val > 0 else 1.0
        steps.append(
            SequentialTerm(
                model=name,
                n_terms=p,
                added_terms=added,
                sum_squares=ss,
                df=added,
                f_value=f_val,
                p_value=p_val,
                recommended=bool(np.isfinite(p_val) and p_val < 0.05),
            )
        )
        prev_sse, prev_p = sse, p

    return tuple(steps)


@dataclass(frozen=True)
class BoxCoxResult:
    """Box-Cox transformation analysis for one response."""

    lambda_hat: float
    ci_low: float
    ci_high: float
    recommendation: str
    log_recommended: bool
    none_recommended: bool
    applicable: bool
    note: str = ""


def box_cox_analysis(values: np.ndarray) -> BoxCoxResult:
    """Recommend a transformation for ``values`` (AC4).

    Release times and rate constants are commonly log-normal, and fitting them
    untransformed is a routine, avoidable error. The recommendation is read off
    the 95% confidence interval for lambda, not off the point estimate alone:
    lambda-hat is rarely exactly 0 or 1, and rounding it without the interval is
    how a transformation gets applied that the data do not actually support.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return BoxCoxResult(
            float("nan"), float("nan"), float("nan"), "insufficient data",
            False, False, applicable=False, note="fewer than 3 finite values",
        )
    if np.any(v <= 0):
        return BoxCoxResult(
            float("nan"), float("nan"), float("nan"),
            "not applicable to non-positive values",
            False, False, applicable=False,
            note="Box-Cox requires strictly positive responses; shift or model on a "
                 "scale that is positive by construction",
        )

    lam, (lo, hi) = stats.boxcox_normmax(v, method="mle"), (float("nan"), float("nan"))
    lam = float(lam)
    try:
        _, _, (lo, hi) = stats.boxcox(v, alpha=0.05)
        lo, hi = float(lo), float(hi)
    except (ValueError, TypeError):
        pass

    log_ok = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 0.0 <= hi)
    none_ok = bool(np.isfinite(lo) and np.isfinite(hi) and lo <= 1.0 <= hi)

    if log_ok and not none_ok:
        rec = "log transform (lambda = 0 lies inside the interval, lambda = 1 does not)"
    elif none_ok and not log_ok:
        rec = "no transform (lambda = 1 lies inside the interval)"
    elif log_ok and none_ok:
        rec = (
            "either; the interval spans both 0 and 1, so the data do not distinguish "
            "log from untransformed. Prefer the interpretable scale."
        )
    else:
        rec = f"power transform near lambda = {lam:.2f}"

    return BoxCoxResult(
        lambda_hat=lam,
        ci_low=lo,
        ci_high=hi,
        recommendation=rec,
        log_recommended=log_ok,
        none_recommended=none_ok,
        applicable=True,
    )
