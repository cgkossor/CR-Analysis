"""Parameter trends across formulations: D0 by API, γ by composition (MW_eq).

MS5 convention: fitted parameters are plotted against the weighted-average HPMC
molecular weight of the whole tablet,

    MW_eq = Σ_i w_i · MW_i      (w_i = mass fraction of HPMC grade i in the TABLET)

Drug, lactose and other non-polymer excipients count as MW 0, so a binary
drug/HPMC tablet gives MW_eq = (1 - x_drug) · MW_HPMC, the form used in the paper.

The reading that came out of MS5 (Fig. 8):
  * D0 is set by the API. It did not trend with MW_eq (S1 Spearman ρ = -0.19 APAP,
    0.20-0.38 IBU depending on whether the one bound-hit D0 is kept) but differed
    about 29x between APAP and IBU (S1 medians 5.2 vs 0.18 x 1e-6 mm²/s).
    Summarise D0 per API.
  * γ is set by the formulation. It fell with MW_eq for APAP (ρ = -0.90 S1, -0.73 D1):
    a stronger, more entangled gel slows the growth of D. For IBU, K100LV separated
    from K15M/K100M as a threshold (erosion) rather than a continuous trend.
  * γ_D1/γ_S1 rose with MW_eq (ρ = 0.72 APAP, 0.80 IBU): the more the gel swells,
    the more geometry S1 has to absorb into γ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

#: Nominal HPMC molecular weights [kDa] used in MS5. K4M is a literature value that
#: was not used in MS5; confirm it (and any other grade) against your supplier data.
HPMC_MW_KDA: dict[str, float] = {"K100LV": 29.0, "K4M": 86.0, "K15M": 134.0, "K100M": 267.0}

#: Okabe-Ito grade colours shared with the MS1/MS5 figures.
GRADE_COLORS = {"K100LV": "#0072B2", "K4M": "#009E73", "K15M": "#E69F00", "K100M": "#CC79A7"}

#: Bound at which D0 counts as unidentifiable (D0 -> 0, γ-driven release).
_D0_FLOOR_E6 = 1e-3


def mw_eq(hpmc_wt_pct: dict[str, float], mw_kda: dict[str, float] | None = None) -> float:
    """Weighted-average HPMC MW [kDa] of a tablet from {grade: wt % of tablet}.

    >>> round(mw_eq({"K15M": 60}), 1)                  # 40 % drug / 60 % K15M
    80.4
    >>> round(mw_eq({"K100LV": 15, "K100M": 15}), 1)   # 30 % HPMC blend + drug/filler
    44.4
    """
    mw = mw_kda or HPMC_MW_KDA
    missing = [g for g in hpmc_wt_pct if g not in mw]
    if missing:
        raise KeyError(f"no MW for grade(s) {missing}; pass mw_kda={{grade: kDa}}")
    return float(sum(w / 100.0 * mw[g] for g, w in hpmc_wt_pct.items()))


def composition_from_columns(row: pd.Series, grade: str) -> dict[str, float] | None:
    """Read HPMC composition from one formulation row of the input CSV.

    Accepted, in order of precedence:
      wt_pct_<GRADE> columns (blends, e.g. wt_pct_K15M, wt_pct_K100LV),
      hpmc_wt_pct (single grade, taken from ``grade``),
      drug_load_pct (binary drug/HPMC tablet: HPMC = 100 - load).
    """
    blend = {c[len("wt_pct_"):]: float(row[c]) for c in row.index
             if c.startswith("wt_pct_") and pd.notna(row[c]) and float(row[c]) > 0}
    if blend:
        return blend
    if "hpmc_wt_pct" in row and pd.notna(row["hpmc_wt_pct"]):
        return {grade: float(row["hpmc_wt_pct"])}
    if "drug_load_pct" in row and pd.notna(row["drug_load_pct"]):
        return {grade: 100.0 - float(row["drug_load_pct"])}
    return None


def parameter_trends(fits: pd.DataFrame, models: tuple[str, ...] = ("S1", "D1")) -> pd.DataFrame:
    """Spearman ρ and log-log slope of D0, γ and γ_D1/γ_S1 against MW_eq, per API.

    ``fits`` needs columns api, model, mw_eq, D0_e6, gamma_e9 (the CLI's fits.csv).
    D0 values at the lower bound are excluded from the D0 trend (not identifiable)
    but counted in ``n_D0_at_bound``. A D0 row also reports the per-API median, which
    is the number to compare between APIs.
    """
    rows = []
    f = fits.dropna(subset=["mw_eq"])
    for (api, model), d in f[f.model.isin(models)].groupby(["api", "model"]):
        for col, name in (("D0_e6", "D0"), ("gamma_e9", "gamma")):
            ok = d[col] > (_D0_FLOOR_E6 if name == "D0" else 0)
            rows.append(_trend(api, model, name, d.mw_eq[ok], d[col][ok]) |
                        dict(n_D0_at_bound=int((~ok).sum()) if name == "D0" else 0))
    piv = f[f.model.isin(["D1", "S1"])].pivot_table(index=["api", "formulation", "mw_eq"],
                                                    columns="model", values="gamma_e9")
    if {"D1", "S1"} <= set(piv.columns):
        r = (piv["D1"] / piv["S1"]).replace([np.inf, -np.inf], np.nan).dropna().reset_index()
        for api, d in r.groupby("api"):
            rows.append(_trend(api, "D1/S1", "gamma_ratio", d.mw_eq, d[0]) | dict(n_D0_at_bound=0))
    return pd.DataFrame(rows)


def _trend(api: str, model: str, param: str, x: pd.Series, y: pd.Series) -> dict[str, object]:
    n = len(y)
    out: dict[str, object] = dict(api=api, model=model, param=param, n=n,
                                  median=float(np.median(y)) if n else np.nan,
                                  min=float(np.min(y)) if n else np.nan,
                                  max=float(np.max(y)) if n else np.nan,
                                  spearman_rho=np.nan, spearman_p=np.nan, loglog_slope=np.nan,
                                  n_mw_levels=int(x.nunique()))
    if n >= 3 and x.nunique() >= 3 and y.nunique() >= 2:
        rho, p = spearmanr(x, y)
        out.update(spearman_rho=float(rho), spearman_p=float(p),
                   loglog_slope=float(linregress(np.log10(x), np.log10(y)).slope))
    return out


def plot_trends(fits: pd.DataFrame, path: str, models: tuple[str, ...] = ("S1", "D1")) -> None:
    """D0 and γ against MW_eq as in MS5 Fig. 8: colour = grade, marker = API.

    Error bars are the profile CIs when fits.csv carries them (CLI output); a lower
    bound of 0 is drawn as a downward arrow instead of a clipped bar.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = fits.dropna(subset=["mw_eq"])
    apis = sorted(f.api.unique())
    markers = dict(zip(apis, "osD^v<>", strict=False))
    fig, axs = plt.subplots(2, len(models), figsize=(4.2 * len(models), 6.4), dpi=200,
                            squeeze=False)
    for j, m in enumerate(models):
        for i, (col, ci, lab) in enumerate(
                (("D0_e6", "D0_prof", "D$_0$ (10$^{-6}$ mm$^2$ s$^{-1}$)"),
                 ("gamma_e9", "gamma_prof", "γ (10$^{-9}$ mm$^2$ s$^{-2}$)"))):
            ax = axs[i, j]
            for k, api in enumerate(apis):
                for grade, d in f[(f.model == m) & (f.api == api) & (f[col] > 0)].groupby("grade"):
                    c = GRADE_COLORS.get(str(grade), "k")
                    kw = dict(color=c, mfc=c if k == 0 else "none", ms=5, ls="none",
                              marker=markers[api], label=f"{api} {grade}")
                    if f"{ci}_lo" in d:
                        lo, hi = d[f"{ci}_lo"], d[f"{ci}_hi"]
                        ax.errorbar(d.mw_eq, d[col], yerr=[(d[col] - lo).where(lo > 0, 0),
                                                           hi - d[col]],
                                    capsize=2, elinewidth=0.8, **kw)
                        for x, y in zip(d.mw_eq[lo <= 0], d[col][lo <= 0], strict=True):
                            ax.annotate("", xy=(x, y / 3), xytext=(x, y),
                                        arrowprops=dict(arrowstyle="->", color=c, lw=0.8))
                    else:
                        ax.plot(d.mw_eq, d[col], **kw)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("MW$_{eq}$ (kDa)")
            ax.set_ylabel(lab)
            ax.set_title(m, fontsize=9)
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
