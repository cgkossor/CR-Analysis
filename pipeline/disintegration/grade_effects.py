"""What the HPMC grade does to disintegration at identical composition.

The design runs every composition (case) at every grade, so grade can be
compared with composition held fixed. Four views of that comparison:

* **ANCOVA** of ln DT on ln Td by grade. Do the grades share one DT-vs-dissolution
  line? The test is for a common slope first, then for a common offset. If they
  do not share a line, a single pooled correlation hides structure, even when
  it is high.
* **Randomised-block model** ``ln DT ~ grade + case``. Case is the block, so
  each grade effect is estimated *at matched composition*. Tukey-adjusted
  pairwise ratios follow. The same model on ln Td gives the dissolution
  counterpart, and comparing the two shows whether grade separates
  disintegration more or less than it separates release.
* **Erosion lag** R = DT / Td: how long the matrix survives relative to the
  time scale of its own release. It is traced against HPMC level by grade and
  against the Peppas exponent (release mechanism).
* **Model diagnostics** for the ANCOVA: normality, constant variance,
  influence, and collinearity of ln Td with the recipe.

Censored DT points are excluded throughout and counted. Td is used rather than
t80 as the dissolution covariate because it is defined for every formulation,
whereas t80 is censored for most high-viscosity ones. A covariate that goes
missing on exactly one grade would bias the grade comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence


@dataclass(frozen=True)
class GradeLine:
    grade: str
    n: int
    offset: float
    """ln DT at the pooled mean ln Td."""
    offset_ci: tuple[float, float]
    slope: float
    slope_ci: tuple[float, float]


@dataclass(frozen=True)
class Ancova:
    n: int
    centre_ln_td: float
    lines: tuple[GradeLine, ...]
    p_common_slope: float
    """F-test: separate slopes vs one shared slope."""
    p_common_offset: float
    """F-test (given a shared slope): grade offsets vs none."""
    shared_slope: float
    shared_slope_ci: tuple[float, float]
    pooled_slope: float
    """ln DT on ln Td ignoring grade. Differs from the shared slope when grades sit on
    offset lines."""
    selected: str
    r2: float
    residual_df: int
    residuals: np.ndarray
    fitted: np.ndarray
    cooks: np.ndarray
    leverage: np.ndarray
    labels: tuple[str, ...]
    shapiro_p: float
    breusch_pagan_p: float
    vif_ln_td: float


@dataclass(frozen=True)
class Contrast:
    a: str
    b: str
    ratio: float
    """exp(effect[a] - effect[b]) at matched composition."""
    ci: tuple[float, float]
    p_adj: float


@dataclass(frozen=True)
class BlockModel:
    response: str
    n: int
    n_blocks: int
    residual_df: int
    effects: dict[str, float]
    """ln-scale grade effect relative to the lowest-viscosity grade."""
    contrasts: tuple[Contrast, ...]
    p_grade: float

    @property
    def span(self) -> float:
        v = list(self.effects.values())
        return float(max(v) - min(v)) if v else float("nan")


@dataclass(frozen=True)
class LagTrend:
    grade: str
    n: int
    slope_per_10wt: float
    ci: tuple[float, float]
    median_ratio: float


@dataclass(frozen=True)
class GradeEffects:
    ancova: Ancova | None
    block_dt: BlockModel | None
    block_td: BlockModel | None
    divergence: pd.DataFrame
    """Per complete case: SD across grades of ln DT and of ln Td, and their ratio."""
    divergence_index: float
    """Median over complete cases of SD(ln DT) / SD(ln Td). Below 1: grade separates DT less."""
    compression: float
    """Grade-effect span for DT over that for Td, from the block models."""
    lag: tuple[LagTrend, ...]
    lag_vs_peppas_rho: float
    lag_vs_peppas_p: float
    n_complete_blocks: int
    n_excluded_censored: int


def _grades(frame: pd.DataFrame, order: list[str]) -> list[str]:
    present = set(frame["grade"])
    return [g for g in order if g in present]


def _ci(res: sm.regression.linear_model.RegressionResultsWrapper, i: int) -> tuple[float, float]:
    lo, hi = res.conf_int()[i]
    return float(lo), float(hi)


def ancova(frame: pd.DataFrame, order: list[str]) -> Ancova | None:
    grades = _grades(frame, order)
    if len(frame) < 2 * len(grades) + 2 or len(grades) < 2:
        return None
    y = frame["ln_dt"].to_numpy(dtype=float)
    centre = float(frame["ln_td"].mean())
    x = frame["ln_td"].to_numpy(dtype=float) - centre
    g = np.column_stack([(frame["grade"] == k).to_numpy(dtype=float) for k in grades])

    cell = sm.OLS(y, np.column_stack([g, g * x[:, None]])).fit()
    parallel = sm.OLS(y, np.column_stack([g, x])).fit()
    single = sm.OLS(y, sm.add_constant(x)).fit()

    def f_test(small: object, big: object) -> float:
        s, b = small, big
        num = (s.ssr - b.ssr) / max(1.0, s.df_resid - b.df_resid)  # type: ignore[attr-defined]
        den = b.ssr / b.df_resid  # type: ignore[attr-defined]
        if den <= 0:
            return float("nan")
        return float(stats.f.sf(num / den, s.df_resid - b.df_resid, b.df_resid))  # type: ignore[attr-defined]

    p_slope = f_test(parallel, cell)
    p_offset = f_test(single, parallel)
    k = len(grades)
    lines = tuple(
        GradeLine(
            grade=grades[i],
            n=int(g[:, i].sum()),
            offset=float(cell.params[i]),
            offset_ci=_ci(cell, i),
            slope=float(cell.params[k + i]),
            slope_ci=_ci(cell, k + i),
        )
        for i in range(k)
    )
    selected, model = (
        ("separate slopes", cell) if p_slope < 0.05 else ("common slope", parallel)
    )
    influence = OLSInfluence(model)
    resid = np.asarray(model.resid, dtype=float)
    # Breusch-Pagan needs an explicit constant; same column space as ``exog``.
    bp_exog = sm.add_constant(np.column_stack([x, g[:, 1:]]))
    try:
        bp = float(het_breuschpagan(resid, bp_exog)[1])
    except (ValueError, np.linalg.LinAlgError):
        bp = float("nan")
    shapiro = float(stats.shapiro(resid).pvalue) if resid.size >= 3 else float("nan")

    # Collinearity of the dissolution covariate with the recipe: VIF of ln Td
    # regressed on grade and composition.
    recipe = np.column_stack(
        [g, frame["api_wt"].to_numpy(dtype=float), frame["hpmc_wt"].to_numpy(dtype=float)]
    )
    aux = sm.OLS(x, recipe).fit()
    ss_tot = float(np.sum((x - x.mean()) ** 2))
    r2_aux = 1.0 - float(aux.ssr) / ss_tot if ss_tot > 0 else float("nan")
    vif = 1.0 / (1.0 - r2_aux) if np.isfinite(r2_aux) and r2_aux < 1 else float("inf")

    ss_y = float(np.sum((y - y.mean()) ** 2))
    return Ancova(
        n=len(y),
        centre_ln_td=centre,
        lines=lines,
        p_common_slope=p_slope,
        p_common_offset=p_offset,
        shared_slope=float(parallel.params[k]),
        shared_slope_ci=_ci(parallel, k),
        pooled_slope=float(single.params[1]),
        selected=selected,
        r2=1.0 - float(model.ssr) / ss_y if ss_y > 0 else float("nan"),
        residual_df=int(model.df_resid),
        residuals=resid,
        fitted=np.asarray(model.fittedvalues, dtype=float),
        cooks=np.asarray(influence.cooks_distance[0], dtype=float),
        leverage=np.asarray(influence.hat_matrix_diag, dtype=float),
        labels=tuple(f"case {int(c)}/{gr}" for c, gr in zip(frame["case"], frame["grade"],
                                                            strict=True)),
        shapiro_p=shapiro,
        breusch_pagan_p=bp,
        vif_ln_td=float(vif),
    )


def block_model(frame: pd.DataFrame, response: str, order: list[str]) -> BlockModel | None:
    grades = _grades(frame, order)
    cases = sorted(frame["case"].unique())
    if len(grades) < 2 or len(cases) < 2:
        return None
    y = frame[response].to_numpy(dtype=float)
    cols = [np.ones(len(frame))]
    cols += [(frame["grade"] == gr).to_numpy(dtype=float) for gr in grades[1:]]
    cols += [(frame["case"] == c).to_numpy(dtype=float) for c in cases[1:]]
    x = np.column_stack(cols)
    if np.linalg.matrix_rank(x) < x.shape[1] or len(y) <= x.shape[1]:
        return None
    res = sm.OLS(y, x).fit()
    cov = np.asarray(res.cov_params())
    k = len(grades)
    # Grade effects relative to the reference (first) grade: coefficients 1..k-1.
    effects = {grades[0]: 0.0} | {grades[i]: float(res.params[i]) for i in range(1, k)}

    def vec(grade: str) -> np.ndarray:
        v = np.zeros(x.shape[1])
        i = grades.index(grade)
        if i > 0:
            v[i] = 1.0
        return v

    q_crit = float(stats.studentized_range.ppf(0.95, k, res.df_resid)) / np.sqrt(2)
    contrasts: list[Contrast] = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = grades[j], grades[i]
            lvec = vec(a) - vec(b)
            diff = float(lvec @ res.params)
            se = float(np.sqrt(lvec @ cov @ lvec))
            p_adj = (
                float(stats.studentized_range.sf(abs(diff) / se * np.sqrt(2), k, res.df_resid))
                if se > 0 else float("nan")
            )
            contrasts.append(
                Contrast(a, b, float(np.exp(diff)),
                         (float(np.exp(diff - q_crit * se)), float(np.exp(diff + q_crit * se))),
                         p_adj)
            )

    # Overall grade F-test: full model vs blocks only.
    reduced = sm.OLS(y, np.column_stack([cols[0], *cols[k:]])).fit()
    num = (reduced.ssr - res.ssr) / (k - 1)
    den = res.ssr / res.df_resid
    p_grade = float(stats.f.sf(num / den, k - 1, res.df_resid)) if den > 0 else float("nan")

    return BlockModel(
        response=response,
        n=len(y),
        n_blocks=len(cases),
        residual_df=int(res.df_resid),
        effects=effects,
        contrasts=tuple(contrasts),
        p_grade=p_grade,
    )


def _lag(frame: pd.DataFrame, order: list[str]) -> tuple[LagTrend, ...]:
    out: list[LagTrend] = []
    for gr in _grades(frame, order):
        sub = frame[frame["grade"] == gr]
        ln_r = (sub["ln_dt"] - sub["ln_td"]).to_numpy(dtype=float)
        h = sub["hpmc_wt"].to_numpy(dtype=float) / 10.0
        if len(sub) < 3 or np.ptp(h) == 0:
            continue
        res = sm.OLS(ln_r, sm.add_constant(h)).fit()
        out.append(
            LagTrend(gr, len(sub), float(res.params[1]), _ci(res, 1),
                     float(np.exp(np.median(ln_r))))
        )
    return tuple(out)


def analyse(matched: pd.DataFrame, order: list[str]) -> GradeEffects:
    usable = matched[~matched["dt_censored"] & np.isfinite(matched["ln_td"])].copy()
    grades = _grades(usable, order)

    # Complete blocks: every grade present and uncensored for that case.
    complete = [
        c for c, g in usable.groupby("case") if set(g["grade"]) >= set(grades)
    ]
    rows: list[dict[str, float | int]] = []
    for c in complete:
        g = usable[usable["case"] == c]
        sd_dt = float(np.std(g["ln_dt"], ddof=1))
        sd_td = float(np.std(g["ln_td"], ddof=1))
        rows.append({"case": int(c), "sd_ln_dt": sd_dt, "sd_ln_td": sd_td,
                     "ratio": sd_dt / sd_td if sd_td > 0 else float("nan")})
    divergence = pd.DataFrame(rows, columns=["case", "sd_ln_dt", "sd_ln_td", "ratio"])
    index = float(divergence["ratio"].median()) if not divergence.empty else float("nan")

    block_dt = block_model(usable, "ln_dt", order)
    block_td = block_model(usable, "ln_td", order)
    compression = (
        block_dt.span / block_td.span
        if block_dt is not None and block_td is not None and block_td.span > 0
        else float("nan")
    )

    ln_r = (usable["ln_dt"] - usable["ln_td"]).to_numpy(dtype=float)
    n_peppas = usable["peppas_n"].to_numpy(dtype=float) if "peppas_n" in usable else None
    if n_peppas is not None and np.isfinite(n_peppas).sum() >= 4:
        ok = np.isfinite(n_peppas)
        sp = stats.spearmanr(n_peppas[ok], ln_r[ok])
        rho, p = float(sp.statistic), float(sp.pvalue)
    else:
        rho, p = float("nan"), float("nan")

    return GradeEffects(
        ancova=ancova(usable, order),
        block_dt=block_dt,
        block_td=block_td,
        divergence=divergence,
        divergence_index=index,
        compression=compression,
        lag=_lag(usable, order),
        lag_vs_peppas_rho=rho,
        lag_vs_peppas_p=p,
        n_complete_blocks=len(complete),
        n_excluded_censored=int(matched["dt_censored"].sum()),
    )


@dataclass(frozen=True)
class Recovered:
    name: str
    truth: float
    estimate: float
    ci: tuple[float, float]

    @property
    def covered(self) -> bool:
        return self.ci[0] <= self.truth <= self.ci[1]


def recover(matched: pd.DataFrame, truth: dict[str, float], order: list[str]) -> tuple[
    Recovered, ...
]:
    """Refit the generator's own structural model and compare with its truth.

    Only meaningful on a synthetic workbook whose notes carry the parameters.
    """
    needed = {"hpmc_centre", "hpmc_scale", "api_centre", "api_scale", "kappa"}
    if not needed <= truth.keys():
        return ()
    usable = matched[~matched["dt_censored"] & np.isfinite(matched["ln_td"])]
    grades = [g for g in _grades(usable, order) if f"gamma[{g}]" in truth]
    if not grades:
        return ()
    usable = usable[usable["grade"].isin(grades)]
    y = (usable["ln_dt"] - usable["ln_td"]).to_numpy(dtype=float)
    z_h = (usable["hpmc_wt"].to_numpy(dtype=float) - truth["hpmc_centre"]) / truth["hpmc_scale"]
    z_a = (usable["api_wt"].to_numpy(dtype=float) - truth["api_centre"]) / truth["api_scale"]
    g = np.column_stack([(usable["grade"] == k).to_numpy(dtype=float) for k in grades])
    res = sm.OLS(y, np.column_stack([g, g * z_h[:, None], -z_a])).fit()
    k = len(grades)
    out = [
        Recovered(f"gamma[{gr}]", truth[f"gamma[{gr}]"], float(res.params[i]), _ci(res, i))
        for i, gr in enumerate(grades)
    ]
    out += [
        Recovered(f"delta[{gr}]", truth[f"delta[{gr}]"], float(res.params[k + i]),
                  _ci(res, k + i))
        for i, gr in enumerate(grades) if f"delta[{gr}]" in truth
    ]
    out.append(Recovered("kappa", truth["kappa"], float(res.params[2 * k]), _ci(res, 2 * k)))
    return tuple(out)
