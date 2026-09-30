"""Target release profile -> the formulations that reach it.

A formulator starts from a profile, usually one worked back from a PK target:
"about 20% at 1 h, half by 6 h, nearly all by 20 h". Asking which drug load,
polymer content and grade give that profile is the inverse of the forward
prediction, and the answer is a region of composition space, not a single
point.

Each target row is a time, a % released and a +/- band. A candidate is scored
by its worst-case deviation in units of the band, so 1.0 means it touches the
edge of a band somewhere. Two readings are kept apart:

* ``feasible``: the prediction itself sits inside every band.
* ``plausible``: each deviation is first forgiven by the cross-validated error
  (G6). The model cannot distinguish a miss smaller than its own error from a
  hit, so these are candidates worth making, not ones the model endorses.

This is the reference the dashboard's ``model.js`` is tested against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pipeline.analysis import Analysis


@dataclass(frozen=True)
class TargetPoint:
    """One row of the target table."""

    time_h: float
    pct: float
    band: float


@dataclass(frozen=True)
class TargetScore:
    """How a predicted profile sits against a target table."""

    score: float
    plausible_score: float
    feasible: bool
    plausible: bool
    f2: float | None


def default_band(time_h: float) -> float:
    """Tighter early, where burst release is the safety question."""
    return 5.0 if time_h <= 2.0 else 10.0


def f2(reference: np.ndarray, test: np.ndarray) -> float | None:
    """FDA similarity factor over the target points (informational only)."""
    reference = np.asarray(reference, dtype=float)
    test = np.asarray(test, dtype=float)
    if reference.size == 0:
        return None
    mean_sq = float(np.mean((reference - test) ** 2))
    return float(50.0 * np.log10(100.0 / np.sqrt(1.0 + mean_sq)))


def score_against_target(
    predicted: np.ndarray, targets: list[TargetPoint], cv_rmse_pct: float | None
) -> TargetScore | None:
    """Worst-case band-normalised deviation, raw and CV-forgiven."""
    predicted = np.asarray(predicted, dtype=float)
    cv = float(cv_rmse_pct) if cv_rmse_pct and cv_rmse_pct > 0 else 0.0
    worst = 0.0
    worst_plausible = 0.0
    for value, target in zip(predicted, targets, strict=True):
        band = target.band if target.band > 0 else 1e-9
        dev = abs(float(value) - target.pct)
        if not np.isfinite(dev):
            return None
        worst = max(worst, dev / band)
        worst_plausible = max(worst_plausible, max(dev - cv, 0.0) / band)
    return TargetScore(
        score=worst,
        plausible_score=worst_plausible,
        feasible=worst <= 1.0 + 1e-12,
        plausible=worst_plausible <= 1.0 + 1e-12,
        f2=f2(np.array([t.pct for t in targets]), predicted),
    )


def evaluate(
    analysis: Analysis,
    api_wt: float,
    hpmc_wt: float,
    lactose_wt: float,
    log10_visc: float,
    targets: list[TargetPoint],
) -> TargetScore | None:
    """Predict one formulation at the target times and score it."""
    from pipeline.analysis import predict_profile

    times = np.array([t.time_h for t in targets], dtype=float)
    pred = predict_profile(analysis, api_wt, hpmc_wt, lactose_wt, log10_visc, times)
    return score_against_target(
        np.asarray(pred["profile"]), targets, analysis.cross_validation.profile_rmse_pct
    )
