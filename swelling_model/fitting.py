"""Fitting, uncertainty and extrapolation for the four MS5 models.

Conventions carried over from the MS5 batch fit (nondim_batch_fit.py):
  * fit the MEAN profile, as a fraction (0-1), unweighted least squares;
  * parameters are fitted in scaled units, D0 in 1e-6 mm²/s and γ in 1e-9 mm²/s²,
    with x_scale="jac" and tight tolerances. Unscaled fits stop early (the original
    script was 14 % off in D0 for this reason);
  * a log grid picks 6 starts, the best of the 6 local fits is kept;
  * bounds D0 in [1e-4, 1e4]e-6 and γ in [0, 1e3]e-9, so γ >= 0 (D never decreases).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, brentq, least_squares
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from .model import (
    MODEL_FLAGS,
    MODELS,
    Geometry,
    Swelling,
    alpha_static,
    doubling_time_h,
    fraction_released,
    time_to_fraction,
)

D_SCALE, G_SCALE = 1e-6, 1e-9
D_GRID = np.logspace(-2, 3, 26)
G_GRID = np.concatenate([[0.0], np.logspace(-3, 2, 21)])
LB = {1: [1e-4], 2: [1e-4, 0.0]}
UB = {1: [1e4], 2: [1e4, 1e3]}


@dataclass
class FitResult:
    model: str
    D0: float  # mm²/s
    gamma: float  # mm²/s²
    geometry: Geometry
    swelling: Swelling | None
    n: int
    sse: float
    D0_ci: tuple[float, float]  # Jacobian 95 % CI, mm²/s
    gamma_ci: tuple[float, float]
    corr_D0_gamma: float
    bound_hit: str
    cov_scaled: np.ndarray = field(repr=False)
    t_fit_max_h: float = float("nan")

    @property
    def n_params(self) -> int:
        return 2 if MODEL_FLAGS[self.model][1] else 1

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sse / self.n))

    @property
    def aicc(self) -> float:
        k = self.n_params + 1
        aic = self.n * np.log(self.sse / self.n) + 2 * k
        return float(aic + 2 * k * (k + 1) / (self.n - k - 1)) if self.n - k - 1 > 0 else np.nan

    def predict(self, t_h: np.ndarray) -> np.ndarray:
        return fraction_released(self.model, t_h, self.D0, self.gamma, self.geometry, self.swelling)

    def t_x(self, target: float) -> float:
        return time_to_fraction(self.model, target, self.D0, self.gamma, self.geometry,
                                self.swelling)

    def summary(self) -> dict[str, object]:
        static = not MODEL_FLAGS[self.model][0]
        return dict(
            model=self.model, D0_e6=self.D0 / D_SCALE, gamma_e9=self.gamma / G_SCALE,
            D0_ci_lo_e6=self.D0_ci[0] / D_SCALE, D0_ci_hi_e6=self.D0_ci[1] / D_SCALE,
            gamma_ci_lo_e9=self.gamma_ci[0] / G_SCALE, gamma_ci_hi_e9=self.gamma_ci[1] / G_SCALE,
            corr_D0_gamma=self.corr_D0_gamma, bound_hit=self.bound_hit, n=self.n,
            fit_to_h=self.t_fit_max_h, RMSE=self.rmse, AICc=self.aicc,
            t_star_h=doubling_time_h(self.D0, self.gamma),
            alpha=alpha_static(self.D0, self.gamma, self.geometry.R0) if static else np.nan,
            beta=self.geometry.beta, t50_h=self.t_x(0.5), t80_h=self.t_x(0.8),
            swelling=self.swelling.source if self.swelling else "",
        )


def _residual_fn(model: str, t_h: np.ndarray, y: np.ndarray, geometry: Geometry,
                 swelling: Swelling | None):
    def f(p: np.ndarray) -> np.ndarray:
        g = p[1] * G_SCALE if len(p) > 1 else 0.0
        return fraction_released(model, t_h, p[0] * D_SCALE, g, geometry, swelling) - y
    return f


def _local_fits(f, npar: int, starts: list[np.ndarray]) -> OptimizeResult | None:
    best = None
    lb, ub = np.array(LB[npar]), np.array(UB[npar])
    for p0 in starts:
        p0 = np.clip(np.asarray(p0, float), lb + 1e-9, ub - 1e-9)
        r = least_squares(f, p0, bounds=(lb, ub), x_scale="jac", ftol=1e-12, xtol=1e-12,
                          gtol=1e-12, max_nfev=5000)
        if best is None or r.cost < best.cost:
            best = r
    return best


def fit_model(
    model: str,
    t_h: np.ndarray,
    frac: np.ndarray,
    geometry: Geometry,
    swelling: Swelling | None = None,
    n_starts: int = 6,
) -> FitResult:
    """Fit one model to a mean profile (fraction released 0-1 vs time in hours)."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    t_h, y = np.asarray(t_h, float), np.asarray(frac, float)
    ok = np.isfinite(t_h) & np.isfinite(y)
    t_h, y = t_h[ok], y[ok]
    npar = 2 if MODEL_FLAGS[model][1] else 1
    f = _residual_fn(model, t_h, y, geometry, swelling)
    grid = [np.array([d, g]) for d in D_GRID for g in G_GRID] if npar == 2 else \
        [np.array([d]) for d in D_GRID]
    grid.sort(key=lambda p: float(np.sum(f(p) ** 2)))
    res = _local_fits(f, npar, grid[:n_starts])
    assert res is not None
    p, J = res.x, res.jac
    dof = max(len(y) - npar, 1)
    cov = np.linalg.pinv(J.T @ J) * (2 * res.cost / dof)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    q = t_dist.ppf(0.975, dof)
    lo, hi = np.maximum(p - q * se, 0.0), p + q * se
    corr = float(cov[0, 1] / (se[0] * se[1])) if npar == 2 and se.all() else np.nan
    hits = [nm for nm, v, lb_, ub_ in zip(("D0", "gamma"), p, LB[npar], UB[npar], strict=False)
            if abs(v - lb_) <= 1e-6 * max(1.0, abs(lb_)) or abs(v - ub_) <= 1e-6 * abs(ub_)]
    return FitResult(
        model=model, D0=p[0] * D_SCALE, gamma=(p[1] if npar == 2 else 0.0) * G_SCALE,
        geometry=geometry, swelling=swelling, n=len(y), sse=float(2 * res.cost),
        D0_ci=(lo[0] * D_SCALE, hi[0] * D_SCALE),
        gamma_ci=(lo[1] * G_SCALE, hi[1] * G_SCALE) if npar == 2 else (0.0, 0.0),
        corr_D0_gamma=corr, bound_hit=";".join(hits), cov_scaled=cov,
        t_fit_max_h=float(t_h.max()),
    )


def fit_all(
    t_h: np.ndarray, frac: np.ndarray, geometry: Geometry, swelling: Swelling | None,
    models: tuple[str, ...] = MODELS,
) -> dict[str, FitResult]:
    """Fit every requested model; D1/D2 are skipped when no swelling is given."""
    return {m: fit_model(m, t_h, frac, geometry, swelling) for m in models
            if swelling is not None or not MODEL_FLAGS[m][0]}


def profile_ci(res: FitResult, t_h: np.ndarray, frac: np.ndarray, which: str = "gamma",
               level: float = 0.95) -> tuple[float, float]:
    """Profile-likelihood CI (F-test threshold) for 'D0' or 'gamma' of a 2-parameter fit.

    Prefer this over the Jacobian CI: D0 and γ trade off along a curved valley, so the
    Jacobian interval is often too narrow on one side. A lower bound of 0 means the
    parameter is not bounded from below by the data.
    """
    if res.n_params != 2:
        raise ValueError("profile_ci needs a 2-parameter model (D1 or S1)")
    t_h, y = np.asarray(t_h, float), np.asarray(frac, float)
    f = _residual_fn(res.model, t_h, y, res.geometry, res.swelling)
    i = 0 if which == "D0" else 1
    o = 1 - i
    p_hat = np.array([res.D0 / D_SCALE, res.gamma / G_SCALE])
    crit = res.sse * (1 + f_dist.ppf(level, 1, len(y) - 2) / (len(y) - 2))

    def sse_at(v: float) -> float:
        def g(q: np.ndarray) -> np.ndarray:
            p = np.empty(2)
            p[i], p[o] = v, q[0]
            return f(p)
        best = None
        for q0 in (p_hat[o], p_hat[o] * 0.3 + 1e-3, p_hat[o] * 3 + 1e-3):
            r = least_squares(g, [np.clip(q0, LB[2][o] + 1e-9, UB[2][o] - 1e-9)],
                              bounds=([LB[2][o]], [UB[2][o]]), x_scale="jac", ftol=1e-12,
                              xtol=1e-12)
            if best is None or r.cost < best.cost:
                best = r
        assert best is not None
        return 2 * best.cost

    def h(v: float) -> float:
        return sse_at(v) - crit

    v0 = p_hat[i]
    lo = 0.0 if h(0.0) <= 0 else brentq(h, 0.0, v0, xtol=1e-6 * max(v0, 1e-6))
    hi_v = v0 * 1.5 + 1e-3
    while h(hi_v) <= 0 and hi_v < 1e4:
        hi_v *= 2
    hi = brentq(h, v0, hi_v, xtol=1e-6 * max(v0, 1e-6))
    scale = D_SCALE if i == 0 else G_SCALE
    return lo * scale, hi * scale


# ---------------------------------------------------------------------------------
# Similarity / error metrics
# ---------------------------------------------------------------------------------

def f2(ref_pct: np.ndarray, test_pct: np.ndarray) -> float:
    """Plain similarity factor f2 (no 85 % truncation), as used in MS5."""
    d = np.asarray(ref_pct, float) - np.asarray(test_pct, float)
    return float(50 * np.log10(100 / np.sqrt(1 + np.mean(d**2))))


def score(res: FitResult, t_h: np.ndarray, frac: np.ndarray) -> dict[str, float]:
    t_h, y = np.asarray(t_h, float), np.asarray(frac, float)
    yhat = res.predict(t_h)
    r = y - yhat
    tss = float(np.sum((y - y.mean()) ** 2))
    return dict(n=len(y), RMSE=float(np.sqrt(np.mean(r**2))),
                R2_raw=1 - float(np.sum(r**2)) / tss if tss > 0 else np.nan,
                max_abs_err_pp=float(100 * np.max(np.abs(r))), f2=f2(100 * y, 100 * yhat))


# ---------------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------------

@dataclass
class Extrapolation:
    fit: FitResult
    t_pred_h: np.ndarray
    pred: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    t80_band_h: tuple[float, float]
    method: str
    test: dict[str, float] | None = None

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(dict(model=self.fit.model, t_h=self.t_pred_h, pred_pct=100 * self.pred,
                                 lo_pct=100 * self.lo, hi_pct=100 * self.hi, band=self.method))


def _band_from_params(res: FitResult, params: np.ndarray, t_pred_h: np.ndarray,
                      level: float) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    curves, t80s = [], []
    for p in params:
        D0, g = p[0] * D_SCALE, (p[1] if len(p) > 1 else 0.0) * G_SCALE
        curves.append(fraction_released(res.model, t_pred_h, D0, g, res.geometry, res.swelling))
        t80s.append(time_to_fraction(res.model, 0.8, D0, g, res.geometry, res.swelling,
                                     t_max_h=max(100.0, 5 * float(t_pred_h.max()))))
    c = np.array(curves)
    a = (1 - level) / 2 * 100
    t80 = np.array(t80s)
    t80 = t80[np.isfinite(t80)]
    band = (float(np.percentile(t80, a)), float(np.percentile(t80, 100 - a))) if len(t80) else \
        (np.nan, np.nan)
    return np.percentile(c, a, axis=0), np.percentile(c, 100 - a, axis=0), band


def extrapolate(
    model: str,
    t_h: np.ndarray,
    frac: np.ndarray,
    geometry: Geometry,
    swelling: Swelling | None,
    cutoff_h: float,
    t_pred_h: np.ndarray,
    per_tablet: pd.DataFrame | None = None,
    n_boot: int = 300,
    level: float = 0.95,
    seed: int = 42,
) -> Extrapolation:
    """Fit on t <= cutoff_h, predict the whole profile on t_pred_h, with a band.

    Band: tablet bootstrap when ``per_tablet`` (columns tablet, time_h, frac) is given,
    which is the honest choice; otherwise parameters are drawn from the Jacobian
    covariance (clipped at the bounds), which understates the D0-γ valley. Any points
    after the cutoff in (t_h, frac) are scored as a held-out test set.
    """
    t_h, y = np.asarray(t_h, float), np.asarray(frac, float)
    tr = t_h <= cutoff_h
    res = fit_model(model, t_h[tr], y[tr], geometry, swelling)
    t_pred_h = np.asarray(t_pred_h, float)
    rng = np.random.default_rng(seed)
    p_hat = np.array([res.D0 / D_SCALE, res.gamma / G_SCALE])[: res.n_params]
    if per_tablet is not None:
        pt = per_tablet[per_tablet["time_h"] <= cutoff_h]
        piv = pt.pivot_table(index="tablet", columns="time_h", values="frac")
        tabs = piv.index.to_numpy()
        f_cols = piv.columns.to_numpy(float)
        params = []
        for _ in range(n_boot):
            yb = piv.loc[rng.choice(tabs, len(tabs), replace=True)].mean(axis=0).to_numpy()
            m = np.isfinite(yb)
            f = _residual_fn(model, f_cols[m], yb[m], geometry, swelling)
            r = _local_fits(f, res.n_params, [p_hat])
            if r is not None:
                params.append(r.x)
        method = f"tablet bootstrap (n={len(params)}, {len(tabs)} tablets)"
    else:
        draws = rng.multivariate_normal(p_hat, res.cov_scaled, size=n_boot)
        params = list(np.clip(draws, LB[res.n_params], UB[res.n_params]))
        method = "Jacobian covariance draws (approximate)"
    lo, hi, t80_band = _band_from_params(res, np.array(params), t_pred_h, level)
    test = score(res, t_h[~tr], y[~tr]) if (~tr).sum() >= 2 else None
    return Extrapolation(res, t_pred_h, res.predict(t_pred_h), lo, hi, t80_band, method, test)


def train_window_scan(
    model: str, t_h: np.ndarray, frac: np.ndarray, geometry: Geometry,
    swelling: Swelling | None, cutoffs_h: list[float],
) -> pd.DataFrame:
    """Refit with progressively shorter training windows and score each on the rest.

    Use this on a formulation that HAS the full profile to learn how short a window
    your system tolerates before predicting the formulations that do not.
    """
    t_h, y = np.asarray(t_h, float), np.asarray(frac, float)
    rows = []
    for c in cutoffs_h:
        tr = t_h <= c
        if tr.sum() < 4 or (~tr).sum() < 2:
            continue
        r = fit_model(model, t_h[tr], y[tr], geometry, swelling)
        s = score(r, t_h[~tr], y[~tr])
        rows.append(dict(model=model, cutoff_h=c, released_at_cutoff_pct=100 * y[tr][-1],
                         n_train=int(tr.sum()), D0_e6=r.D0 / D_SCALE, gamma_e9=r.gamma / G_SCALE,
                         bound_hit=r.bound_hit, t80_h=r.t_x(0.8),
                         **{f"test_{k}": v for k, v in s.items()}))
    return pd.DataFrame(rows)
