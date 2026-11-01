"""Early-delay variants of the MS5 models: a plain lag and a first-order release step.

In MS5, D1 with one extra early-delay parameter fitted every formulation better and predicted
the 30 h APAP profile better (test RMSE 1.3-1.4 pp vs 2.3 pp). Two forms of the delay are
offered here, each with three parameters (D0, γ and the delay):

``lag``   M(t) = M_base(t - t_lag), zero before t_lag. A sharp delay: nothing happens, then
          the base model starts.
``k``     drug becomes mobile by a first-order step, σ(t) = exp(-k t) (σ = fraction not yet
          mobile, uniform in the tablet), and then diffuses as in the base model:

              M(t) = 1 - e^{-kt} - ∫0^t k e^{-ks} G(I_r(t) - I_r(s), I_z(t) - I_z(s)) ds,
              G = S_r S_z = 1 - series_release.

          k -> ∞ recovers the base model. The equivalent mean delay is τ = 1/k.

"Base" is any of D1, D2, S1, S2. The k-form was written as a dissolution step, but in MS5
k came out at ≈ 3 h⁻¹ for both APAP and IBU (solubilities 40-400× apart), so read k as an
early delay of unknown origin (wetting/hydration, drug dissolution or an experimental dead
time) until composition says otherwise. ``delay_probe`` is the test for that; see
EARLY_DELAY.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, brentq, least_squares
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist

from .fitting import D_SCALE, G_SCALE, fit_model
from .model import (
    _LAM2,
    _MU2,
    MODEL_FLAGS,
    Geometry,
    Swelling,
    fraction_released,
    integral_dynamic,
    integral_static,
)

Resid = Callable[[np.ndarray], np.ndarray]

DELAY_KINDS = ("lag", "k")
#: Scaled bounds: D0 [1e-6 mm²/s], γ [1e-9 mm²/s²], k [1/h] or t_lag [h].
_LB = {"lag": [1e-4, 0.0, 0.0], "k": [1e-4, 0.0, 1e-3]}
_UB = {"lag": [1e4, 1e3, 3.0], "k": [1e4, 1e3, 1e4]}
_X_GRID = {"lag": np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4]), "k": np.logspace(-1, 3, 9)}
_X_NAME = {"lag": "t_lag_h", "k": "k_h"}


def _integrals(model: str, t_s: np.ndarray, D0: float, gamma: float, geometry: Geometry,
               swelling: Swelling | None) -> tuple[np.ndarray, np.ndarray]:
    dynamic, time_dep = MODEL_FLAGS[model]
    g = gamma if time_dep else 0.0
    if not dynamic:
        return (integral_static(t_s, D0, g, geometry.R0),
                integral_static(t_s, D0, g, geometry.Z0))
    if swelling is None:
        raise ValueError(f"{model} needs swelling boundaries")
    sw = swelling
    return (integral_dynamic(t_s, D0, g, sw.R0, sw.Rf, sw.kr_h / 3600.0),
            integral_dynamic(t_s, D0, g, sw.Z0, sw.Zf, sw.kz_h / 3600.0))


def _G(dIr: np.ndarray, dIz: np.ndarray) -> np.ndarray:
    sr = np.sum(4.0 / _LAM2 * np.exp(-_LAM2 * dIr[:, None]), axis=1)
    sz = np.sum(2.0 / _MU2 * np.exp(-_MU2 * dIz[:, None]), axis=1)
    return sr * sz


def fraction_released_lag(model: str, t_h: np.ndarray, D0: float, gamma: float, t_lag_h: float,
                          geometry: Geometry, swelling: Swelling | None = None) -> np.ndarray:
    """Base model shifted by ``t_lag_h`` (0 before the lag). D0 mm²/s, γ mm²/s²."""
    t = np.asarray(t_h, float) - t_lag_h
    out = np.zeros_like(t)
    m = t > 0
    if m.any():
        out[m] = fraction_released(model, t[m], D0, gamma, geometry, swelling)
    return out


def fraction_released_k(model: str, t_h: np.ndarray, D0: float, gamma: float, k_h: float,
                        geometry: Geometry, swelling: Swelling | None = None,
                        n_nodes: int = 300) -> np.ndarray:
    """Base model with a first-order release step k [1/h] (Duhamel convolution).

    The integral is taken as a Stieltjes sum over dV = d(1 - e^{-ks}), exact for any k when
    G is locally constant. The nodes are the union of a grid clustered at s = t (where G
    changes fastest) and a grid uniform in the mobile fraction 1 - e^{-ks} (which resolves
    large k). Agrees with an independent modal solution to < 0.002 pp at the MS5 data times
    for k from 0.3 to 1e4 1/h.
    """
    t_h = np.atleast_1d(np.asarray(t_h, float))
    k = k_h / 3600.0
    out = np.zeros_like(t_h)
    u = np.linspace(0.0, 1.0, n_nodes)
    for i, th in enumerate(t_h):
        t = th * 3600.0
        if t <= 0:
            continue
        vmax = -np.expm1(-k * t)
        s_v = -np.log1p(-u[:-1] * vmax) / k
        s = np.unique(np.concatenate([t * (1 - u**2), s_v, [0.0, t]]))
        Ir, Iz = _integrals(model, np.concatenate([s, [t]]), D0, gamma, geometry, swelling)
        G = _G(Ir[-1] - Ir[:-1], Iz[-1] - Iz[:-1])
        dV = -np.diff(np.expm1(-k * s))           # mass made mobile in each interval
        out[i] = vmax - float(np.sum(0.5 * (G[1:] + G[:-1]) * dV))
    return out


def predict_delay(model: str, kind: str, t_h: np.ndarray, D0: float, gamma: float, x: float,
                  geometry: Geometry, swelling: Swelling | None = None) -> np.ndarray:
    if kind == "lag":
        return fraction_released_lag(model, t_h, D0, gamma, x, geometry, swelling)
    if kind == "k":
        return fraction_released_k(model, t_h, D0, gamma, x, geometry, swelling)
    raise ValueError(f"kind must be one of {DELAY_KINDS}")


@dataclass
class DelayFit:
    model: str
    kind: str
    p: np.ndarray  # scaled: D0_e6, gamma_e9, x (k_h or t_lag_h)
    geometry: Geometry
    swelling: Swelling | None
    n: int
    sse: float
    corr: np.ndarray = field(repr=False)
    jac_ci: np.ndarray = field(repr=False)  # (3, 2) scaled
    prof_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    t_fit_max_h: float = float("nan")

    @property
    def label(self) -> str:
        return f"{self.model}+{self.kind}"

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sse / self.n))

    @property
    def aicc(self) -> float:
        k = 4
        aic = self.n * np.log(self.sse / self.n) + 2 * k
        return float(aic + 2 * k * (k + 1) / (self.n - k - 1)) if self.n - k - 1 > 0 else np.nan

    @property
    def delay_min(self) -> float:
        """Delay in minutes: t_lag, or the mean delay τ = 1/k."""
        return 60.0 * (self.p[2] if self.kind == "lag" else 1.0 / self.p[2])

    def predict(self, t_h: np.ndarray) -> np.ndarray:
        return predict_delay(self.model, self.kind, t_h, self.p[0] * D_SCALE,
                             self.p[1] * G_SCALE, self.p[2], self.geometry, self.swelling)

    def bounded(self, name: str) -> bool | float:
        """True if the profile CI of ``name`` stays inside the parameter bounds."""
        if name not in self.prof_ci:
            return np.nan
        j = ["D0", "gamma", "x"].index(name)
        lo, hi = self.prof_ci[name]
        lb, ub = _LB[self.kind][j], _UB[self.kind][j]
        return bool(lo > lb * (1 + 1e-9) + 1e-12 and hi < ub * (1 - 1e-9))

    def summary(self) -> dict[str, object]:
        d: dict[str, object] = dict(
            model=self.label, D0_e6=self.p[0], gamma_e9=self.p[1],
            **{_X_NAME[self.kind]: self.p[2]},
            delay_min=self.delay_min, n=self.n, fit_to_h=self.t_fit_max_h, RMSE=self.rmse,
            AICc=self.aicc, corr_D0_gamma=self.corr[0, 1], corr_D0_delay=self.corr[0, 2],
            corr_gamma_delay=self.corr[1, 2],
            swelling=self.swelling.source if self.swelling else "")
        for nm, (lo, hi) in self.prof_ci.items():
            key = {"D0": "D0", "gamma": "gamma", "x": "delay"}[nm]
            d[f"{key}_prof_lo"], d[f"{key}_prof_hi"] = lo, hi
            d[f"{key}_bounded"] = self.bounded(nm)
        return d


def _resid(model: str, kind: str, t_h: np.ndarray, y: np.ndarray, geometry: Geometry,
           swelling: Swelling | None) -> Resid:
    def f(p: np.ndarray) -> np.ndarray:
        return predict_delay(model, kind, t_h, p[0] * D_SCALE, p[1] * G_SCALE, p[2], geometry,
                             swelling) - y
    return f


def _lsq(f: Resid, p0: np.ndarray, lb: Sequence[float] | np.ndarray,
         ub: Sequence[float] | np.ndarray) -> OptimizeResult:
    p0 = np.clip(np.asarray(p0, float), np.asarray(lb) + 1e-9, np.asarray(ub) - 1e-9)
    return least_squares(f, p0, bounds=(lb, ub), x_scale="jac", ftol=1e-12, xtol=1e-12,
                         gtol=1e-12, max_nfev=3000)


def _profile(f: Resid, p_hat: np.ndarray, sse: float, lb: list[float], ub: list[float], j: int,
             n: int,
             level: float) -> tuple[float, float]:
    crit = sse * (1 + f_dist.ppf(level, 1, n - 3) / (n - 3))
    o = [i for i in range(3) if i != j]

    def h(v: float) -> float:
        def g(q: np.ndarray) -> np.ndarray:
            p = p_hat.copy()
            p[j], p[o] = v, q
            return f(p)
        best = min((_lsq(g, p_hat[o] * sc + 1e-3 * (sc != 1), np.array(lb)[o], np.array(ub)[o])
                    for sc in (1.0, 0.3, 3.0)), key=lambda r: r.cost)
        return 2 * best.cost - crit

    v0, L, U = p_hat[j], lb[j], ub[j]
    tol = 1e-6 * max(abs(v0), 1e-6)
    lo = L if h(L) <= 0 else brentq(h, L, v0, xtol=tol)
    hv = min(v0 * 1.5 + 1e-3, U)
    while h(hv) <= 0 and hv < U:
        hv = min(hv * 2, U)
    hi = U if h(hv) <= 0 else brentq(h, v0, hv, xtol=tol)
    return float(lo), float(hi)


def fit_delay(model: str, kind: str, t_h: np.ndarray, frac: np.ndarray, geometry: Geometry,
              swelling: Swelling | None = None, profile: tuple[str, ...] = ("D0", "x"),
              level: float = 0.95) -> DelayFit:
    """Fit ``model`` + delay (``kind`` 'lag' or 'k') to a mean profile (fraction 0-1, hours).

    Starts from the base-model fit. ``profile`` lists the parameters ('D0', 'gamma', 'x')
    that get a profile-likelihood CI; each costs about as much as the fit itself.
    """
    if kind not in DELAY_KINDS:
        raise ValueError(f"kind must be one of {DELAY_KINDS}")
    if not MODEL_FLAGS[model][1]:
        raise ValueError("use a model with D = D0 + γt (D1 or S1) as the base")
    t_h, y = np.asarray(t_h, float), np.asarray(frac, float)
    ok = np.isfinite(t_h) & np.isfinite(y)
    t_h, y = t_h[ok], y[ok]
    base = fit_model(model, t_h, y, geometry, swelling)
    b = np.array([base.D0 / D_SCALE, base.gamma / G_SCALE])
    f = _resid(model, kind, t_h, y, geometry, swelling)
    lb, ub = _LB[kind], _UB[kind]
    starts = [np.array([d, g, x]) for d in b[0] * np.array([0.3, 1, 3])
              for g in np.maximum(b[1], 1e-3) * np.array([0.5, 1, 2]) for x in _X_GRID[kind]]
    starts.sort(key=lambda p: float(np.sum(f(np.clip(p, lb, ub)) ** 2)))
    res = min((_lsq(f, p0, lb, ub) for p0 in starts[:6]), key=lambda r: r.cost)
    p, n = res.x, len(y)
    dof = max(n - 3, 1)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * (2 * res.cost / dof)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = cov / np.outer(se, se) if se.all() else np.full((3, 3), np.nan)
    q = t_dist.ppf(0.975, dof)
    out = DelayFit(model, kind, p, geometry, swelling, n, float(2 * res.cost), corr,
                   np.c_[p - q * se, p + q * se], t_fit_max_h=float(t_h.max()))
    for nm in profile:
        j = ["D0", "gamma", "x"].index(nm)
        out.prof_ci[nm] = _profile(f, p, out.sse, lb, ub, j, n, level)
    return out


# ---------------------------------------------------------------------------------
# Composition probe: does the delay track the matrix, the API, or nothing?
# ---------------------------------------------------------------------------------

def delay_probe(fits: pd.DataFrame, delay_col: str = "delay_min") -> dict[str, pd.DataFrame]:
    """Regress the delay (and D0, γ) of the delay fits on composition.

    ``fits`` needs one row per formulation and delay model with columns
    model ('D1+k', 'D1+lag', ...), api, grade, lactose_wt_pct, hpmc_wt_pct, and
    delay_min, D0_e6, gamma_e9 (the CLI's fits.csv with --delay gives all of them).

    Returns
      ``spearman``: within each model x API x grade, ρ of each parameter vs lactose wt %;
      ``ols``: log10(parameter) ~ lactose wt % + C(api) + C(grade), per model, with
               coefficients, SE and p (statsmodels). HPMC is left out because, with API
               load in the model via C(api) only, lactose and HPMC are near-collinear
               (mixture constraint): read the lactose coefficient as "lactose replacing
               HPMC at fixed API".
      ``api_effect``: per model and grade, the delay range across APIs and a Kruskal-Wallis p.
    Pre-registered reading (EARLY_DELAY.md): delay falls with lactose and not with API ->
    wetting/hydration; delay differs by API, not lactose -> drug dissolution; neither ->
    experimental (dead time / t = 0).
    """
    import statsmodels.formula.api as smf
    from scipy.stats import kruskal, spearmanr

    need = {"model", "api", "grade", "lactose_wt_pct", delay_col, "D0_e6", "gamma_e9"}
    miss = need - set(fits.columns)
    if miss:
        raise KeyError(f"delay_probe needs columns {sorted(miss)}")
    f = fits[fits.model.str.contains(r"\+")].copy()
    params = [delay_col, "D0_e6", "gamma_e9"]
    sp, ols, api = [], [], []
    for (m, a, g), d in f.groupby(["model", "api", "grade"]):
        for c in params:
            ok = d[c] > 0
            rho, p = (spearmanr(d.lactose_wt_pct[ok], d[c][ok]) if ok.sum() >= 3 and
                      d.lactose_wt_pct[ok].nunique() >= 3 and d[c][ok].nunique() >= 2
                      else (np.nan, np.nan))
            sp.append(dict(model=m, api=a, grade=g, param=c, n=int(ok.sum()),
                           rho_vs_lactose=float(rho), p=float(p),
                           median=float(d[c][ok].median()) if ok.any() else np.nan))
    for m, d in f.groupby("model"):
        for c in params:
            dd = d[d[c] > 0].assign(logy=lambda x, c=c: np.log10(x[c]))
            terms = ["lactose_wt_pct"] + [f"C({v})" for v in ("api", "grade")
                                          if dd[v].nunique() > 1]
            if len(dd) < len(terms) + 3:
                continue
            r = smf.ols("logy ~ " + " + ".join(terms), data=dd).fit()
            for term in r.params.index:
                ols.append(dict(model=m, param=c, term=term, coef=r.params[term],
                                se=r.bse[term], p=r.pvalues[term], n=int(r.nobs),
                                r2=r.rsquared))
        for g, d2 in d.groupby("grade"):
            groups = [x[delay_col].to_numpy() for _, x in d2.groupby("api") if len(x) >= 2]
            p = kruskal(*groups).pvalue if len(groups) >= 2 else np.nan
            api.append(dict(model=m, grade=g, n_api=len(groups), kruskal_p=float(p),
                            **{f"median_{a}": float(x[delay_col].median())
                               for a, x in d2.groupby("api")}))
    return dict(spearman=pd.DataFrame(sp), ols=pd.DataFrame(ols), api_effect=pd.DataFrame(api))
