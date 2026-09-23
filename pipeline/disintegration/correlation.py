"""How closely disintegration tracks each dissolution response.

Everything is computed on design-point means. Replicates are not independent
formulations, and correlating them would inflate n about threefold.

* **Pearson and Spearman** for ln DT against every dissolution metric, with
  95% intervals from resampling formulations. Pearson is taken on the log scale
  for time-like metrics, where the relation is multiplicative. Spearman is
  scale-free and guards against one extreme formulation carrying the result.
* **Deming regression** of ln DT on ln Td. Both sides are means of noisy
  replicates, and ordinary least squares, which treats x as exact, would bias
  the slope toward zero. The error-variance ratio is estimated from the
  replicates themselves.
* **Partial correlation** of ln DT with ln Td after removing grade and
  composition. It asks whether dissolution tells you anything about
  disintegration *beyond what the recipe already does*.

Censored DT points are excluded here (they are lower bounds, not values) and
counted, so the reader knows the correlation describes the uncensored part of
the design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.disintegration import settings


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    log: bool
    expected_sign: int
    """+1 slower dissolution -> longer DT, -1 the reverse, 0 no prior."""


#: The dissolution metrics DT is compared with. The keys are columns of the matched frame.
METRICS: tuple[Metric, ...] = (
    Metric("td_h", "Weibull Td", True, 1),
    Metric("mdt_h", "MDT", True, 1),
    Metric("t50", "t50", True, 1),
    Metric("t80", "t80", True, 1),
    Metric("pct_1h", "% at 1 h", False, -1),
    Metric("pct_2h", "% at 2 h", False, -1),
    Metric("pct_4h", "% at 4 h", False, -1),
    Metric("pct_8h", "% at 8 h", False, -1),
    Metric("pct_12h", "% at 12 h", False, -1),
    Metric("pct_24h", "% at 24 h", False, -1),
    Metric("weibull_beta", "Weibull β", False, 0),
    Metric("peppas_n", "Peppas n", False, 0),
)


@dataclass(frozen=True)
class Correlation:
    metric: Metric
    n: int
    pearson: float
    pearson_ci: tuple[float, float]
    spearman: float
    spearman_ci: tuple[float, float]
    spearman_p: float

    @property
    def sign_agrees(self) -> bool:
        return bool(np.sign(self.pearson) == np.sign(self.spearman))

    @property
    def sign_as_expected(self) -> bool | None:
        if self.metric.expected_sign == 0 or not np.isfinite(self.spearman):
            return None
        return bool(np.sign(self.spearman) == self.metric.expected_sign)


@dataclass(frozen=True)
class Deming:
    label: str
    n: int
    slope: float
    intercept: float
    slope_ci: tuple[float, float]
    error_ratio: float
    """Replicate error variance of y over that of x."""
    ols_slope: float


@dataclass(frozen=True)
class Partial:
    n: int
    r: float
    p: float
    controls: tuple[str, ...]


@dataclass(frozen=True)
class WithinGrade:
    grade: str
    n: int
    pearson: float
    spearman: float


@dataclass(frozen=True)
class CorrelationResult:
    table: tuple[Correlation, ...]
    within_grade: tuple[WithinGrade, ...]
    """ln DT against ln Td inside each grade. A pooled r below these means the grades
    sit on offset lines, not that the relation is weak."""
    deming: tuple[Deming, ...]
    """Pooled first, then one per grade."""
    partial: Partial
    n_matched: int
    n_censored_excluded: int

    def by_key(self, key: str) -> Correlation | None:
        for c in self.table:
            if c.metric.key == key:
                return c
        return None

    @property
    def ranked(self) -> list[Correlation]:
        """Strongest first, by |Spearman|."""
        return sorted(
            (c for c in self.table if np.isfinite(c.spearman)),
            key=lambda c: -abs(c.spearman),
        )


def _pair(frame: pd.DataFrame, metric: Metric) -> tuple[np.ndarray, np.ndarray]:
    x = frame[metric.key].to_numpy(dtype=float)
    y = frame["ln_dt"].to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if metric.log:
        ok &= x > 0
        x = np.where(ok, np.log(np.where(ok, x, 1.0)), np.nan)
    return x[ok], y[ok]


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(x, y)[0, 1]), float(stats.spearmanr(x, y).statistic)


def _rowwise_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r of each row of ``a`` with the same row of ``b``; NaN when degenerate."""
    da = a - a.mean(axis=1, keepdims=True)
    db = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((da**2).sum(axis=1) * (db**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (da * db).sum(axis=1) / denom, np.nan)


def _bootstrap(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int
) -> tuple[tuple[float, float], tuple[float, float]]:
    n = x.size
    if n < 4:
        nan = (float("nan"), float("nan"))
        return nan, nan
    idx = rng.integers(0, n, size=(n_boot, n))
    xb, yb = x[idx], y[idx]
    pear = _rowwise_corr(xb, yb)
    spear = _rowwise_corr(stats.rankdata(xb, axis=1), stats.rankdata(yb, axis=1))

    def ci(v: np.ndarray) -> tuple[float, float]:
        v = v[np.isfinite(v)]
        if v.size < 10:
            return float("nan"), float("nan")
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    return ci(pear), ci(spear)


def deming(x: np.ndarray, y: np.ndarray, ratio: float) -> tuple[float, float]:
    """(slope, intercept) of Deming regression; ``ratio`` = var(err y) / var(err x)."""
    mx, my = float(np.mean(x)), float(np.mean(y))
    sxx = float(np.mean((x - mx) ** 2))
    syy = float(np.mean((y - my) ** 2))
    sxy = float(np.mean((x - mx) * (y - my)))
    if sxy == 0 or not np.isfinite(ratio) or ratio <= 0:
        return float("nan"), float("nan")
    d = syy - ratio * sxx
    slope = (d + np.sqrt(d * d + 4 * ratio * sxy * sxy)) / (2 * sxy)
    return float(slope), float(my - slope * mx)


def _deming_fit(
    label: str, frame: pd.DataFrame, rng: np.random.Generator, n_boot: int
) -> Deming | None:
    ok = frame[np.isfinite(frame["ln_td"]) & np.isfinite(frame["ln_dt"])]
    if len(ok) < 4:
        return None
    x = ok["ln_td"].to_numpy(dtype=float)
    y = ok["ln_dt"].to_numpy(dtype=float)
    var_y = float(np.nanmean(ok["ln_dt_se2"].to_numpy(dtype=float)))
    var_x = float(np.nanmean(ok["ln_td_se2"].to_numpy(dtype=float)))
    ratio = var_y / var_x if var_x > 0 else float("nan")
    slope, intercept = deming(x, y, ratio)
    boots = np.full(n_boot, np.nan)
    n = x.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = deming(x[idx], y[idx], ratio)[0]
    boots = boots[np.isfinite(boots)]
    ci = (
        (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
        if boots.size >= 10 else (float("nan"), float("nan"))
    )
    ols = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 0 else float("nan")
    return Deming(label, n, slope, intercept, ci, ratio, ols)


def _partial(frame: pd.DataFrame) -> Partial:
    ok = frame[np.isfinite(frame["ln_td"]) & np.isfinite(frame["ln_dt"])]
    grades = sorted(ok["grade"].unique(), key=lambda g: float(
        ok.loc[ok["grade"] == g, "viscosity_cp"].iloc[0]))
    cols = [np.ones(len(ok))]
    cols += [(ok["grade"] == g).to_numpy(dtype=float) for g in grades[1:]]
    cols += [ok["api_wt"].to_numpy(dtype=float), ok["hpmc_wt"].to_numpy(dtype=float)]
    z = np.column_stack(cols)
    k = z.shape[1]
    n = len(ok)
    if n - k - 1 < 2:
        return Partial(n, float("nan"), float("nan"), ())

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(z, v, rcond=None)
        return np.asarray(v - z @ beta)

    rx = resid(ok["ln_td"].to_numpy(dtype=float))
    ry = resid(ok["ln_dt"].to_numpy(dtype=float))
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return Partial(n, float("nan"), float("nan"), ())
    r = float(np.corrcoef(rx, ry)[0, 1])
    dof = n - k - 1
    t = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    p = float(2 * stats.t.sf(abs(t), dof))
    return Partial(n, r, p, ("grade", "API wt%", "HPMC wt%"))


def analyse(matched: pd.DataFrame, grade_order: list[str]) -> CorrelationResult:
    usable = matched[~matched["dt_censored"]]
    rng = np.random.default_rng(settings.SEED)

    table: list[Correlation] = []
    for metric in METRICS:
        if metric.key not in usable.columns:
            continue
        x, y = _pair(usable, metric)
        pear, spear = _safe_corr(x, y)
        p_s = float(stats.spearmanr(x, y).pvalue) if np.isfinite(spear) else float("nan")
        p_ci, s_ci = _bootstrap(x, y, rng, settings.BOOTSTRAP_N)
        table.append(Correlation(metric, int(x.size), pear, p_ci, spear, s_ci, p_s))

    within: list[WithinGrade] = []
    td = next(m for m in METRICS if m.key == "td_h")
    for g in grade_order:
        x, y = _pair(usable[usable["grade"] == g], td)
        pear, spear = _safe_corr(x, y)
        if np.isfinite(pear):
            within.append(WithinGrade(g, int(x.size), pear, spear))

    fits: list[Deming] = []
    pooled = _deming_fit("all grades", usable, rng, settings.BOOTSTRAP_N)
    if pooled is not None:
        fits.append(pooled)
    for g in grade_order:
        fit = _deming_fit(g, usable[usable["grade"] == g], rng, settings.BOOTSTRAP_N)
        if fit is not None:
            fits.append(fit)

    return CorrelationResult(
        table=tuple(table),
        within_grade=tuple(within),
        deming=tuple(fits),
        partial=_partial(usable),
        n_matched=len(matched),
        n_censored_excluded=int(matched["dt_censored"].sum()),
    )
