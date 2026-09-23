"""Replicate precision: per-formulation summaries, outliers, and where the noise lives.

Disintegration of a swollen gel is a coarse end point, so the first question is
whether replicates agree well enough for a formulation mean to mean anything.
Three quantities answer it:

* the per-formulation CV, and Dixon's Q for a single discordant tablet. Q is the
  test that is defined for three or four replicates, where Grubbs is not. A
  flagged replicate is reported, never removed;
* the intraclass correlation ICC(1): the share of total variance that is
  *between* formulations. Near 1 means formulations differ far more than
  tablets within one do, which is what makes every later model possible;
* a Brown-Forsythe test of whether scatter differs by grade. A heavier gel
  erodes more erratically, and a model that assumes equal variance would
  over-trust it.

All on the log scale. Times are positive and right-skewed, and a ratio of two
formulations is the natural comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.disintegration import settings

#: Two-sided Dixon Q critical values at alpha = 0.05 (Rorabacher, 1991).
DIXON_Q95: dict[int, float] = {
    3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625, 7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466,
}


@dataclass(frozen=True)
class DixonFlag:
    case: int
    grade: str
    replicate: int
    dt_h: float
    q: float
    q_critical: float


@dataclass(frozen=True)
class GradePrecision:
    grade: str
    n_formulations: int
    pooled_cv: float
    median_cv: float


@dataclass(frozen=True)
class Precision:
    points: pd.DataFrame
    """One row per (case, grade): n, n_censored, mean/sd/cv, geometric mean, CI."""
    flags: tuple[DixonFlag, ...]
    icc: float
    by_grade: tuple[GradePrecision, ...]
    brown_forsythe_p: float
    brown_forsythe_stat: float


def dixon_q(values: np.ndarray) -> tuple[float, int]:
    """(Q, index of the suspect value). Q is gap / range for the more extreme end."""
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return float("nan"), -1
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    spread = s[-1] - s[0]
    if spread <= 0:
        return 0.0, int(order[0])
    q_low = (s[1] - s[0]) / spread
    q_high = (s[-1] - s[-2]) / spread
    if q_high >= q_low:
        return float(q_high), int(order[-1])
    return float(q_low), int(order[0])


def _summarise(g: pd.DataFrame) -> dict[str, float | int | bool]:
    ok = g.loc[~g["censored"], "dt_h"].to_numpy(dtype=float)
    every = g["dt_h"].to_numpy(dtype=float)
    n = len(every)
    logs = np.log(every)
    mean = float(np.mean(every))
    sd = float(np.std(every, ddof=1)) if n > 1 else float("nan")
    half = (
        float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
    )
    return {
        "n": n,
        "n_censored": int(g["censored"].sum()),
        "censored": bool(g["censored"].any()),
        "mean_h": mean,
        "sd_h": sd,
        "cv": sd / mean if mean > 0 else float("nan"),
        "ci_lo_h": mean - half,
        "ci_hi_h": mean + half,
        "gmean_h": float(np.exp(np.mean(logs))),
        "ln_mean": float(np.mean(logs)),
        "ln_sd": float(np.std(logs, ddof=1)) if n > 1 else float("nan"),
        "min_h": float(np.min(every)),
        "max_h": float(np.max(every)),
        "n_uncensored": int(ok.size),
    }


def point_table(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (case, grade), g in long.groupby(["case", "grade"], sort=True):
        first = g.iloc[0]
        rows.append(
            {
                "case": int(case),
                "grade": str(grade),
                "id": str(first["id"]),
                "api": str(first["api"]),
                "api_wt_dt": float(first["api_wt"]),
                "hpmc_wt_dt": float(first["hpmc_wt"]),
                "lactose_wt_dt": float(first["lactose_wt"]),
                **_summarise(g),
            }
        )
    return pd.DataFrame(rows)


def _icc1(long: pd.DataFrame) -> float:
    """One-way random-effects ICC(1) on ln DT, unequal group sizes allowed."""
    usable = long.loc[~long["censored"]].copy()
    usable["ln"] = np.log(usable["dt_h"].to_numpy(dtype=float))
    groups = [g["ln"].to_numpy() for _, g in usable.groupby(["case", "grade"]) if len(g) > 1]
    k = len(groups)
    if k < 2:
        return float("nan")
    sizes = np.array([len(g) for g in groups], dtype=float)
    n_total = float(sizes.sum())
    grand = float(np.concatenate(groups).mean())
    ssb = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    ssw = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    msb = ssb / (k - 1)
    msw = ssw / (n_total - k)
    n0 = (n_total - float((sizes**2).sum()) / n_total) / (k - 1)
    denom = msb + (n0 - 1) * msw
    return float((msb - msw) / denom) if denom > 0 else float("nan")


def analyse(long: pd.DataFrame, grade_order: list[str]) -> Precision:
    points = point_table(long)

    flags: list[DixonFlag] = []
    for (case, grade), g in long.groupby(["case", "grade"], sort=True):
        ok = g.loc[~g["censored"]]
        n = len(ok)
        crit = DIXON_Q95.get(n)
        if crit is None:
            continue
        q, idx = dixon_q(np.log(ok["dt_h"].to_numpy(dtype=float)))
        if np.isfinite(q) and q > crit:
            hit = ok.iloc[idx]
            flags.append(
                DixonFlag(int(case), str(grade), int(hit["replicate"]), float(hit["dt_h"]),
                          q, crit)
            )

    # Within-formulation deviations on the log scale, grouped by grade.
    usable = long.loc[~long["censored"]].copy()
    usable["ln"] = np.log(usable["dt_h"].to_numpy(dtype=float))
    usable["dev"] = usable["ln"] - usable.groupby(["case", "grade"])["ln"].transform("mean")
    by_grade: list[GradePrecision] = []
    samples: list[np.ndarray] = []
    for grade in grade_order:
        sub = usable[usable["grade"] == grade]
        if sub.empty:
            continue
        per = points[(points["grade"] == grade) & (~points["censored"])]
        dof = int(sum(len(g) - 1 for _, g in sub.groupby("case")))
        pooled_var = float((sub["dev"] ** 2).sum() / dof) if dof > 0 else float("nan")
        by_grade.append(
            GradePrecision(
                grade=grade,
                n_formulations=int(per.shape[0]),
                pooled_cv=float(np.sqrt(np.expm1(pooled_var))) if np.isfinite(pooled_var)
                else float("nan"),
                median_cv=float(per["cv"].median()) if not per.empty else float("nan"),
            )
        )
        samples.append(sub["dev"].to_numpy(dtype=float))

    if len(samples) >= 2 and all(s.size >= 2 for s in samples):
        bf = stats.levene(*samples, center="median")
        bf_stat, bf_p = float(bf.statistic), float(bf.pvalue)
    else:
        bf_stat, bf_p = float("nan"), float("nan")

    return Precision(
        points=points,
        flags=tuple(flags),
        icc=_icc1(long),
        by_grade=tuple(by_grade),
        brown_forsythe_p=bf_p,
        brown_forsythe_stat=bf_stat,
    )


def cv_offenders(points: pd.DataFrame) -> pd.DataFrame:
    return points[points["cv"] > settings.DT_CV_WARN]
