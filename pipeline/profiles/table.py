"""Run metric extraction across a whole database and assemble tidy tables.

Two tables come out, and the distinction between them is load-bearing:

``replicate_table``
    One row per (id, replicate) -- 99 rows for the placeholder. Weibull is fitted
    here, per replicate profile, because averaging curves before fitting a
    non-linear model distorts the parameters.

``design_point_table``
    One row per (case, grade) -- 33 rows. Replicate parameters are averaged onto
    the design point, and the replicate spread is retained *separately* as
    analytical repeatability.

Response surfaces are fitted on the second. Vessel replicates drawn from one
compression batch are subsamples, not independent experimental units: fitting on
all 99 would inflate the effective sample size and shrink every standard error
by roughly sqrt(3), manufacturing significance that the design does not support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.io.load import Database
from pipeline.profiles.metrics import ProfileMetrics, extract_metrics


def _row_from_metrics(metrics: ProfileMetrics) -> dict[str, object]:
    row: dict[str, object] = {}

    for name, value in metrics.release_times.items():
        row[name] = value.numeric_or_nan
        row[f"{name}_censored"] = value.censored
        row[f"{name}_bound"] = value.bound if value.bound is not None else float("nan")

    row.update(metrics.pct_at)
    for name, ok in metrics.pct_at_valid.items():
        row[f"{name}_valid"] = ok

    row["mdt_h"] = metrics.mdt_h
    row["mdt_truncated"] = metrics.mdt_truncated
    row["early_slope"] = metrics.early_slope
    row["late_slope"] = metrics.late_slope
    row["slope_ratio"] = metrics.slope_ratio
    row["slope_ratio_valid"] = metrics.slope_ratio_valid
    row["peak_pct"] = metrics.peak_pct
    row["n_points"] = metrics.n_points

    weibull = metrics.fits["weibull"]
    row["weibull_valid"] = weibull.valid
    row["weibull_f_inf"] = weibull.params.get("f_inf", float("nan"))
    row["weibull_td"] = weibull.params.get("td", float("nan"))
    row["weibull_beta"] = weibull.params.get("beta", float("nan"))
    row["weibull_r2"] = weibull.r_squared
    row["weibull_rmse"] = weibull.rmse
    row["weibull_asymptote_identified"] = weibull.asymptote_identified
    row["log10_td"] = (
        float(np.log10(weibull.params["td"]))
        if weibull.valid and weibull.params.get("td", 0.0) > 0
        else float("nan")
    )

    peppas = metrics.fits["peppas"]
    row["peppas_valid"] = peppas.valid
    row["peppas_k"] = peppas.params.get("k", float("nan"))
    row["peppas_n"] = peppas.params.get("n", float("nan"))
    row["peppas_r2"] = peppas.r_squared
    row["peppas_n_points"] = peppas.n_points

    for name in ("higuchi", "first_order"):
        fit = metrics.fits[name]
        row[f"{name}_valid"] = fit.valid
        row[f"{name}_r2"] = fit.r_squared
        row[f"{name}_rmse"] = fit.rmse
        for param, estimate in fit.params.items():
            row[f"{name}_{param}"] = estimate

    return row


def build_replicate_table(db: Database) -> pd.DataFrame:
    """Metrics for every replicate profile (one row per id x replicate)."""
    rows: list[dict[str, object]] = []
    key_cols = ["id", "api", "case", "grade", "replicate"]

    for keys, group in db.profiles.groupby(key_cols, sort=True):
        ordered = group.sort_values("time_h")
        metrics = extract_metrics(
            ordered["time_h"].to_numpy(dtype=float),
            ordered["pct_released"].to_numpy(dtype=float),
        )
        row: dict[str, object] = dict(zip(key_cols, keys, strict=True))
        row["tablet_mass_mg"] = float(ordered["tablet_mass_mg"].iloc[0])
        row["dose_mg"] = float(ordered["dose_mg"].iloc[0])
        row["api_wt"] = float(ordered["api_wt"].iloc[0])
        row["hpmc_wt"] = float(ordered["hpmc_wt"].iloc[0])
        row["lactose_wt"] = float(ordered["lactose_wt"].iloc[0])
        row["run_date"] = ordered["run_date"].iloc[0]
        row.update(_row_from_metrics(metrics))
        rows.append(row)

    table = pd.DataFrame(rows)
    table["viscosity_cp"] = table["grade"].map(db.grade_viscosity_cp).astype(float)
    table["log10_visc"] = np.log10(table["viscosity_cp"])
    return table


#: Parameters carried from the replicate level up to the design point.
AGGREGATED = ("log10_td", "weibull_beta", "weibull_f_inf", "mdt_h", "t50", "peppas_n")


def build_design_point_table(replicates: pd.DataFrame) -> pd.DataFrame:
    """Collapse replicates onto design points, keeping the replicate spread separately.

    The mean is what the surface is fitted on. The SD travels alongside it as
    *analytical repeatability* and must never be substituted for the
    cross-validated prediction error that G6 attaches to predictions -- it is a
    much smaller number measuring a different thing.
    """
    keys = ["api", "case", "grade"]
    agg: dict[str, tuple[str, str]] = {}
    for name in AGGREGATED:
        agg[f"{name}_mean"] = (name, "mean")
        agg[f"{name}_sd"] = (name, "std")
        agg[f"{name}_n"] = (name, "count")

    table = replicates.groupby(keys, sort=True).agg(**agg).reset_index()

    context = (
        replicates.groupby(keys, sort=True)
        .agg(
            api_wt=("api_wt", "first"),
            hpmc_wt=("hpmc_wt", "first"),
            lactose_wt=("lactose_wt", "first"),
            viscosity_cp=("viscosity_cp", "first"),
            log10_visc=("log10_visc", "first"),
            tablet_mass_mg=("tablet_mass_mg", "mean"),
            run_date=("run_date", "first"),
            n_replicates=("replicate", "nunique"),
            n_censored_t80=("t80_censored", "sum"),
            peak_pct=("peak_pct", "mean"),
            n_asymptote_identified=("weibull_asymptote_identified", "sum"),
        )
        .reset_index()
    )

    merged = table.merge(context, on=keys, how="left")
    merged["censoring_status"] = np.select(
        [
            merged["n_censored_t80"] == 0,
            merged["n_censored_t80"] == merged["n_replicates"],
        ],
        ["none", "full"],
        default="partial",
    )
    return merged
