"""Command line: fit the four models to every formulation in a tidy CSV.

    python -m swelling_model --input release.csv [--cutoff-h 8] [--predict-to-h 24]
    python -m swelling_model --demo            # synthetic data, checks the install

Input CSV, one row per tablet and time point (extra columns are ignored):

    formulation  grade  tablet  time_h  released_pct  diameter_mm  height_mm

``time_min`` may replace ``time_h``. diameter/height are the initial (dry) release
tablet dimensions, full height; they are averaged per formulation. Optional
``--swelling-data`` CSV (grade, time_h, diameter_mm, height_mm) replaces the
preloaded swelling defaults for the grades it contains.

Optional columns for the parameter trends (D0 by API, γ vs MW_eq):
    api                         API name (defaults to "API")
    wt_pct_<GRADE>              wt % of each HPMC grade in the tablet (blends), or
    hpmc_wt_pct                 wt % of the single grade in ``grade``, or
    drug_load_pct               binary drug/HPMC tablet (HPMC = 100 - load)

Early-delay probe (EARLY_DELAY.md): ``--delay k,lag`` also fits the base model
(``--delay-base``, default D1, S1 when a grade has no swelling) with a first-order release
step (+k) and with a plain lag (+lag), on the same training window. With columns
    lactose_wt_pct (and api, grade)    plus optionally api_wt_pct, hpmc_wt_pct
the delay, D0 and γ of these fits are regressed on composition (delay_probe_*.csv).

Outputs (default outputs/swelling_model/, gitignored): fits.csv, predictions.csv,
test_scores.csv when a cutoff leaves held-out points, one PNG per formulation, and
trends.csv / trends.png when MW_eq is available for >= 3 formulations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .delay import delay_probe, fit_delay
from .fitting import D_SCALE, G_SCALE, extrapolate, fit_model, profile_ci
from .model import MODEL_FLAGS, MODELS, Geometry, fraction_released
from .swelling import fit_swelling, swelling_for_tablet
from .trends import composition_from_columns, mw_eq, parameter_trends, plot_trends

COLORS = {"D1": "magenta", "D2": "purple", "S1": "cyan", "S2": "blue"}
DELAY_COLORS = {"k": "tab:blue", "lag": "tab:green"}
COMP_COLS = ("api_wt_pct", "hpmc_wt_pct", "lactose_wt_pct", "drug_load_pct")


def load_release(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "time_h" not in df and "time_min" in df:
        df["time_h"] = df["time_min"] / 60.0
    need = {"formulation", "grade", "tablet", "time_h", "released_pct", "diameter_mm", "height_mm"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"input is missing columns: {sorted(missing)}")
    df["frac"] = df["released_pct"] / 100.0
    return df


def demo_data(seed: int = 1) -> pd.DataFrame:
    """Synthetic 24 h profile from a known S1/D1-like formulation (no real data)."""
    rng = np.random.default_rng(seed)
    geom = Geometry.from_tablet(12.7, 2.8)
    sw = swelling_for_tablet("K15M", geom.R0, geom.Z0)
    t = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24], float)
    true = fraction_released("D1", t, 3.0e-6, 2.5e-9, geom, sw)
    rows = [dict(formulation="DEMO", grade="K15M", tablet=k, time_h=ti,
                 released_pct=100 * max(0.0, yi + rng.normal(0, 0.015)) if ti > 0 else 0.0,
                 diameter_mm=12.7, height_mm=2.8)
            for k in range(6) for ti, yi in zip(t, true, strict=True)]
    df = pd.DataFrame(rows)
    df["frac"] = df["released_pct"] / 100
    return df


def score_arrays(t: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    r = y - yhat
    return dict(n=len(y), RMSE=float(np.sqrt(np.mean(r**2))),
                max_abs_err_pp=float(100 * np.max(np.abs(r))),
                f2=float(50 * np.log10(100 / np.sqrt(1 + np.mean((100 * r) ** 2)))))


def _style(ax) -> None:
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)


def run(df: pd.DataFrame, out: Path, cutoff_h: float | None, predict_to_h: float | None,
        rate_scaling: str, swelling_df: pd.DataFrame | None, models: tuple[str, ...],
        n_boot: int, delay_kinds: tuple[str, ...] = (), delay_base: str = "D1") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    own_sw = {}
    if swelling_df is not None:
        for g, d in swelling_df.groupby("grade"):
            own_sw[g] = fit_swelling(d, str(g))
    fit_rows, pred_rows, test_rows = [], [], []
    for form, d in df.groupby("formulation"):
        grade = str(d["grade"].iloc[0])
        api = str(d["api"].iloc[0]) if "api" in d else "API"
        comp = composition_from_columns(d.iloc[0], grade)
        mweq = mw_eq(comp) if comp else np.nan
        geom = Geometry.from_tablet(d["diameter_mm"].mean(), d["height_mm"].mean())
        try:
            sw = swelling_for_tablet(grade, geom.R0, geom.Z0, rate_scaling=rate_scaling,
                                     defaults=own_sw.get(grade))
        except KeyError as e:
            print(f"[{form}] {e} -> fitting S1/S2 only")
            sw = None
        prof = d.groupby("time_h")["frac"].agg(["mean", "std", "count"]).reset_index()
        t, y = prof["time_h"].to_numpy(), prof["mean"].to_numpy()
        tr = t <= (cutoff_h if cutoff_h is not None else np.inf)
        t_end = predict_to_h or float(t.max())
        tg = np.linspace(0, t_end, 300)
        fig, ax = plt.subplots(figsize=(5.2, 4), dpi=200)
        ax.errorbar(t[tr], 100 * y[tr], yerr=100 * prof["std"][tr], fmt="^", color="k", mfc="k",
                    ecolor="grey", capsize=2, ms=4, lw=0.8, label="Data (fit)", zorder=10)
        if (~tr).any():
            ax.errorbar(t[~tr], 100 * y[~tr], yerr=100 * prof["std"][~tr], fmt="^", color="k",
                        mfc="w", ecolor="grey", capsize=2, ms=4, lw=0.8, label="Data (held out)",
                        zorder=10)
            ax.axvline(cutoff_h, color="gray", ls=":", lw=0.8)
        for m in models:
            if MODEL_FLAGS[m][0] and sw is None:
                continue
            per_tab = d[["tablet", "time_h", "frac"]]
            if cutoff_h is not None or predict_to_h is not None:
                ex = extrapolate(m, t, y, geom, sw, cutoff_h if cutoff_h is not None else np.inf,
                                 tg, per_tablet=per_tab, n_boot=n_boot)
                res = ex.fit
                pred_rows.append(ex.frame().assign(formulation=form))
                if ex.test:
                    test_rows.append(dict(formulation=form, model=m, cutoff_h=cutoff_h, **ex.test))
                if res.n_params == 2:
                    ax.fill_between(tg, 100 * ex.lo, 100 * ex.hi, color=COLORS[m], alpha=0.15, lw=0)
                row = res.summary() | dict(t80_band_lo_h=ex.t80_band_h[0],
                                           t80_band_hi_h=ex.t80_band_h[1], band=ex.method)
            else:
                res = fit_model(m, t, y, geom, sw)
                row = res.summary()
            if res.n_params == 2:
                tf, yf = t[tr], y[tr]
                for w in ("D0", "gamma"):
                    lo, hi = profile_ci(res, tf, yf, w)
                    s = D_SCALE if w == "D0" else G_SCALE
                    row[f"{w}_prof_lo"], row[f"{w}_prof_hi"] = lo / s, hi / s
            fit_rows.append(dict(formulation=form, api=api, grade=grade, mw_eq=mweq, R0_mm=geom.R0,
                                 Z0_mm=geom.Z0) | row)
            yy = 100 * res.predict(tg)
            cut = tg <= (cutoff_h if cutoff_h is not None else np.inf)
            ax.plot(tg[cut], yy[cut], color=COLORS[m], lw=1.4, label=m)
            ax.plot(tg[~cut], yy[~cut], color=COLORS[m], lw=1.4, ls="--")
        base = delay_base if (sw is not None or not MODEL_FLAGS[delay_base][0]) else "S1"
        comp_extra = {c: float(d[c].iloc[0]) for c in COMP_COLS if c in d}
        for kind in delay_kinds:
            dfit = fit_delay(base, kind, t[tr], y[tr], geom, sw)
            row = dict(formulation=form, api=api, grade=grade, mw_eq=mweq, R0_mm=geom.R0,
                       Z0_mm=geom.Z0) | comp_extra | dfit.summary()
            fit_rows.append(row)
            if (~tr).sum() >= 2:
                yh = dfit.predict(t[~tr])
                test_rows.append(dict(formulation=form, model=dfit.label, cutoff_h=cutoff_h,
                                      **score_arrays(t[~tr], y[~tr], yh)))
            yy = 100 * dfit.predict(tg)
            cut = tg <= (cutoff_h if cutoff_h is not None else np.inf)
            ax.plot(tg[cut], yy[cut], color=DELAY_COLORS[kind], lw=1.2, label=dfit.label)
            ax.plot(tg[~cut], yy[~cut], color=DELAY_COLORS[kind], lw=1.2, ls="--")
        ax.set_xlabel("Time (hr)")
        ax.set_ylabel("Drug Released (%)")
        ax.set_title(f"{form} ({grade})", fontsize=8)
        ax.set_ylim(0, 110)
        _style(ax)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(form))
        fig.savefig(out / f"{safe}.png")
        plt.close(fig)
        print(f"[{form}] done")
    fits = pd.DataFrame(fit_rows)
    fits.to_csv(out / "fits.csv", index=False)
    dl = fits[fits.model.str.contains(r"\+")] if delay_kinds else fits.iloc[:0]
    if len(dl) and "lactose_wt_pct" in dl and dl.formulation.nunique() >= 6:
        for name, tab in delay_probe(dl).items():
            tab.to_csv(out / f"delay_probe_{name}.csv", index=False)
        print("delay probe written (delay_probe_spearman/ols/api_effect.csv)")
    if fits.mw_eq.notna().sum() >= 3 * fits.model.nunique():
        tr = parameter_trends(fits)
        tr.to_csv(out / "trends.csv", index=False)
        plot_trends(fits, str(out / "trends.png"))
        print(tr[["api", "model", "param", "n", "median", "spearman_rho", "loglog_slope"]]
              .round(3).to_string(index=False))
    if delay_kinds:
        cols = ["formulation", "model", "RMSE", "AICc", "delay_min", "D0_e6", "D0_bounded"]
        print(fits[fits.model.str.contains(r"\+")][[c for c in cols if c in fits]]
              .round(3).to_string(index=False))
    if pred_rows:
        pd.concat(pred_rows).to_csv(out / "predictions.csv", index=False)
    if test_rows:
        pd.DataFrame(test_rows).to_csv(out / "test_scores.csv", index=False)
        print(pd.DataFrame(test_rows)[["formulation", "model", "RMSE", "f2"]].round(3)
              .to_string(index=False))
    print(f"outputs in {out}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m swelling_model", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path)
    ap.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--out", type=Path, default=Path("outputs/swelling_model"))
    ap.add_argument("--cutoff-h", type=float, help="fit only t <= cutoff; later points = test")
    ap.add_argument("--predict-to-h", type=float, help="extend the predicted curves to this time")
    ap.add_argument("--rate-scaling", default="none", choices=["none", "axial_L2"])
    ap.add_argument("--swelling-data", type=Path, help="own swelling CSV (overrides defaults)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--delay", default="", help="early-delay fits to add: k, lag or k,lag")
    ap.add_argument("--delay-base", default="D1", choices=["D1", "S1"])
    a = ap.parse_args()
    if not a.demo and a.input is None:
        ap.error("give --input or --demo")
    df = demo_data() if a.demo else load_release(a.input)
    sw = pd.read_csv(a.swelling_data) if a.swelling_data else None
    cutoff = a.cutoff_h if not a.demo or a.cutoff_h else 8.0
    run(df, a.out, cutoff, a.predict_to_h or (24.0 if a.demo else None), a.rate_scaling, sw,
        tuple(m.strip() for m in a.models.split(",")), a.n_boot,
        tuple(k.strip() for k in a.delay.split(",") if k.strip()), a.delay_base)


if __name__ == "__main__":
    main()
