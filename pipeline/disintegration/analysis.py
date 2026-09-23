"""Run the disintegration section end to end on top of a dissolution analysis.

One result object, so the report, figures, diagnostics and audit all read the
same numbers. The join to dissolution is on (case, grade), the key the DoE
responses already use. Every row that fails to join is counted, never dropped
silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.analysis import Analysis
from pipeline.disintegration import correlation, grade_effects, predict, replicates, settings
from pipeline.disintegration.load import DisintegrationData
from pipeline.doe.analysis import DoeAnalysis, ResponseAnalysis
from pipeline.doe.analysis import run as run_doe
from pipeline.doe.responses import ResponseData, ResponseSpec

#: Dissolution columns carried into the matched frame (replicate means per point).
_DISSOLUTION_COLUMNS = (
    "t50", "t80", "pct_1h", "pct_2h", "pct_4h", "pct_8h", "pct_12h", "pct_24h", "mdt_h",
    "weibull_beta", "peppas_n",
)

DT_RESPONSE = ResponseSpec(
    "dt_h", "Disintegration time", "h", "dt_h", censorable=True,
    note="Geometric mean of the replicates. Formulations still intact at the test "
         "end are censored and excluded from this model; they are the slowest-eroding "
         "ones, so the fit describes the faster part of the design.",
)
LOG_DT_RESPONSE = ResponseSpec(
    "log10_dt", "log10 disintegration time", "log10 h", "log10_dt", censorable=True,
    note="The same response on the log scale, where replicate scatter is roughly "
         "constant. Used for model comparison.",
)


@dataclass(frozen=True)
class Matching:
    n_dt_points: int
    n_dissolution_points: int
    n_matched: int
    dt_without_dissolution: tuple[str, ...]
    dissolution_without_dt: tuple[str, ...]
    composition_mismatch: tuple[str, ...]
    composition_sum_off: tuple[str, ...]
    grades_without_viscosity: tuple[str, ...]


@dataclass(frozen=True)
class Plausibility:
    n_compared_t50: int
    below_t50: tuple[str, ...]
    n_compared_t80: int
    below_t80: tuple[str, ...]


@dataclass(frozen=True)
class DisintegrationAnalysis:
    data: DisintegrationData
    matched: pd.DataFrame
    grade_order: tuple[str, ...]
    matching: Matching
    precision: replicates.Precision
    correlation: correlation.CorrelationResult
    grades: grade_effects.GradeEffects
    doe: DoeAnalysis
    loo: tuple[predict.LooModel, ...]
    recovery: tuple[grade_effects.Recovered, ...]
    plausibility: Plausibility
    is_synthetic: bool

    @property
    def dt_fit(self) -> ResponseAnalysis | None:
        return self.doe.by_key("dt_h")

    @property
    def log_dt_fit(self) -> ResponseAnalysis | None:
        return self.doe.by_key("log10_dt")

    @property
    def censored_points(self) -> pd.DataFrame:
        return self.matched[self.matched["dt_censored"]]


def _label(case: object, grade: object) -> str:
    return f"case {int(case)}/{grade}"  # type: ignore[call-overload]


def _dissolution_frame(a: Analysis) -> pd.DataFrame:
    dp = a.design_points
    keep = [
        "case", "grade", "api", "api_wt", "hpmc_wt", "lactose_wt", "viscosity_cp", "v_coded",
        "log10_td_mean", "log10_td_sd", "log10_td_n",
    ]
    base = dp[[c for c in keep if c in dp.columns]].copy()
    cols = [c for c in _DISSOLUTION_COLUMNS if c in a.replicates.columns]
    means = a.replicates.groupby(["case", "grade"], sort=True)[cols].mean().reset_index()
    return base.merge(means, on=["case", "grade"], how="left")


def build_matched(a: Analysis, points: pd.DataFrame) -> tuple[pd.DataFrame, Matching]:
    diss = _dissolution_frame(a)
    both = diss.merge(points, on=["case", "grade"], how="outer", indicator=True,
                      suffixes=("", "_dtsheet"))
    only_dt = both[both["_merge"] == "right_only"]
    only_diss = both[both["_merge"] == "left_only"]
    matched = both[both["_merge"] == "both"].drop(columns="_merge").copy()

    tol = settings.COMP_SUM_TOL
    mismatch: list[str] = []
    sum_off: list[str] = []
    for r in matched.itertuples():
        pairs = [(r.api_wt, r.api_wt_dt), (r.hpmc_wt, r.hpmc_wt_dt),
                 (r.lactose_wt, r.lactose_wt_dt)]
        if any(np.isfinite(d) and abs(float(s) - float(d)) > tol for s, d in pairs):
            mismatch.append(_label(r.case, r.grade))
        dt_comp = [d for _, d in pairs]
        if all(np.isfinite(dt_comp)) and abs(sum(dt_comp) - 100.0) > tol:
            sum_off.append(_label(r.case, r.grade))

    matched["dt_h"] = matched["gmean_h"]
    matched["ln_dt"] = matched["ln_mean"]
    matched["log10_dt"] = matched["ln_mean"] / math.log(10.0)
    matched["ln_dt_se2"] = matched["ln_sd"] ** 2 / matched["n"]
    matched["dt_censored"] = matched["censored"].astype(bool)
    ln10 = math.log(10.0)
    matched["ln_td"] = matched["log10_td_mean"] * ln10
    matched["td_h"] = np.exp(matched["ln_td"])
    matched["ln_td_se2"] = (matched["log10_td_sd"] * ln10) ** 2 / matched["log10_td_n"]
    matched["erosion_lag"] = matched["dt_h"] / matched["td_h"]
    matched = matched.sort_values(["case", "viscosity_cp"], kind="mergesort").reset_index(
        drop=True
    )

    no_visc = sorted(
        {str(g) for g, v in zip(matched["grade"], matched["viscosity_cp"], strict=True)
         if not np.isfinite(v)}
    )
    matching = Matching(
        n_dt_points=len(points),
        n_dissolution_points=len(diss),
        n_matched=len(matched),
        dt_without_dissolution=tuple(_label(c, g) for c, g in zip(
            only_dt["case"], only_dt["grade"], strict=True)),
        dissolution_without_dt=tuple(_label(c, g) for c, g in zip(
            only_diss["case"], only_diss["grade"], strict=True)),
        composition_mismatch=tuple(mismatch),
        composition_sum_off=tuple(sum_off),
        grades_without_viscosity=tuple(no_visc),
    )
    return matched, matching


def _response_data(
    spec: ResponseSpec, dp: pd.DataFrame, matched: pd.DataFrame
) -> ResponseData:
    joined = dp[["case", "grade"]].merge(
        matched[["case", "grade", spec.source_column, "dt_censored"]],
        on=["case", "grade"], how="left",
    )
    values = joined[spec.source_column].to_numpy(dtype=float)
    censored = joined["dt_censored"].fillna(False).to_numpy(dtype=bool)
    available = np.isfinite(values) & ~censored
    labels = [_label(c, g) for c, g in zip(joined["case"], joined["grade"], strict=True)]
    missing = tuple(labels[i] for i in range(len(values)) if not available[i])
    return ResponseData(
        spec=spec,
        values=np.where(available, values, np.nan),
        available=available,
        n_total=len(values),
        n_missing=len(missing),
        missing_points=missing,
    )


def _plausibility(matched: pd.DataFrame) -> Plausibility:
    ok = matched[~matched["dt_censored"]]
    out: dict[str, tuple[int, tuple[str, ...]]] = {}
    for key in ("t50", "t80"):
        if key not in ok.columns:
            out[key] = (0, ())
            continue
        sub = ok[np.isfinite(ok[key])]
        below = sub[sub["dt_h"] < sub[key]]
        out[key] = (len(sub), tuple(_label(c, g) for c, g in zip(
            below["case"], below["grade"], strict=True)))
    return Plausibility(out["t50"][0], out["t50"][1], out["t80"][0], out["t80"][1])


def run_disintegration(a: Analysis, data: DisintegrationData) -> DisintegrationAnalysis:
    dp = a.design_points
    order = [
        str(g) for g in dp.drop_duplicates("grade").sort_values("viscosity_cp")["grade"]
    ]
    precision = replicates.analyse(data.long, order)
    matched, matching = build_matched(a, precision.points)

    n_levels = int(dp["v_coded"].nunique())
    power = min(max(n_levels - 1, 0), 2)
    doe = run_doe(
        dp, a.replicates, power,
        responses=[_response_data(DT_RESPONSE, dp, matched),
                   _response_data(LOG_DT_RESPONSE, dp, matched)],
    )
    log_fit = doe.by_key("log10_dt")
    return DisintegrationAnalysis(
        data=data,
        matched=matched,
        grade_order=tuple(order),
        matching=matching,
        precision=precision,
        correlation=correlation.analyse(matched, order),
        grades=grade_effects.analyse(matched, order),
        doe=doe,
        loo=predict.compare(matched, log_fit, order),
        recovery=grade_effects.recover(matched, data.truth, order),
        plausibility=_plausibility(matched),
        is_synthetic=bool(data.is_synthetic or a.quality.is_synthetic),
    )
