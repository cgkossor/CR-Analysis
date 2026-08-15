"""Release-metric extraction for a single profile (AC2).

Every metric carries a validity flag. Two rules run throughout:

* **Unequal spacing is respected.** The sampling schedule is front-loaded, so any
  quantity involving an integral or a slope uses the actual time intervals.
  Treating the grid as uniform would silently weight the first two hours far
  above the remaining twenty-two.
* **Censoring is propagated, never coerced** (G5). A metric that cannot be
  computed because the profile stopped short says so; it does not become NaN and
  slip out of a downstream mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import config
from pipeline.profiles.censoring import CensoredValue, time_to_percent
from pipeline.profiles.fits import FitResult, fit_all

#: Levels for the t_x family (AC2).
RELEASE_LEVELS: tuple[float, ...] = (10.0, 25.0, 50.0, 80.0)

#: Timepoints (h) at which % released is reported (AC2).
SAMPLE_TIMES_H: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 12.0, 24.0)

#: Windows for the early and late phase slopes (AC2).
EARLY_WINDOW_H: tuple[float, float] = (0.0, 2.0)
LATE_WINDOW_H: tuple[float, float] = (8.0, 24.0)


@dataclass(frozen=True)
class ProfileMetrics:
    """All AC2 metrics for one replicate profile."""

    release_times: dict[str, CensoredValue] = field(default_factory=dict)
    pct_at: dict[str, float] = field(default_factory=dict)
    pct_at_valid: dict[str, bool] = field(default_factory=dict)

    mdt_h: float = float("nan")
    mdt_truncated: bool = True
    mdt_note: str = ""

    early_slope: float = float("nan")
    early_slope_n: int = 0
    late_slope: float = float("nan")
    late_slope_n: int = 0
    slope_ratio: float = float("nan")
    slope_ratio_valid: bool = False

    fits: dict[str, FitResult] = field(default_factory=dict)

    peak_pct: float = float("nan")
    n_points: int = 0
    censored_at_80: bool = False


def _prepare(time_h: np.ndarray, pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_h, dtype=float)
    y = np.asarray(pct, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t, kind="mergesort")
    return t[order], y[order]


def percent_at(time_h: np.ndarray, pct: np.ndarray, target_h: float) -> tuple[float, bool]:
    """% released at ``target_h`` by linear interpolation.

    Returns ``(value, valid)``. Extrapolation beyond the observed window is
    never performed -- a request outside the range comes back invalid (G3).
    """
    t, y = _prepare(time_h, pct)
    if len(t) == 0:
        return float("nan"), False
    if target_h < t[0] or target_h > t[-1]:
        return float("nan"), False
    return float(np.interp(target_h, t, y)), True


def mean_dissolution_time(
    time_h: np.ndarray, pct: np.ndarray
) -> tuple[float, bool, str]:
    """Model-independent mean dissolution time over the observed window.

    ``MDT = sum(t_mid * dM) / sum(dM)`` using the actual (unequal) intervals.

    Returns ``(mdt, truncated, note)``. ``truncated`` is True whenever the
    profile has not plateaued by the last observation, in which case the value
    is a lower bound on the true MDT and must be labelled as such -- exactly the
    censored formulations whose MDT matters most.
    """
    t, y = _prepare(time_h, pct)
    if len(t) < 2:
        return float("nan"), True, "fewer than 2 points"

    dm = np.diff(y)
    t_mid = (t[1:] + t[:-1]) / 2.0
    total = float(np.sum(dm))
    if total <= 0:
        return float("nan"), True, "no net release observed"

    mdt = float(np.sum(t_mid * dm) / total)

    # The observed-window MDT equals the true MDT once release is essentially
    # complete. That happens either because the curve has flattened, or because
    # it has simply run out of drug to release -- a profile that reaches ~100%
    # is finished even if its final segment is still steep.
    peak = float(np.max(y))
    tail = y[-1] - y[max(len(y) - 3, 0)]
    plateaued = bool(tail < 0.02 * max(peak, 1.0))
    complete = bool(y[-1] >= config.COMPLETE_RELEASE_PCT)
    if plateaued or complete:
        return mdt, False, ""
    return (
        mdt,
        True,
        "profile still rising at the last observation; MDT is a lower bound over the "
        "observed window, not the true mean dissolution time",
    )


def _window_slope(
    time_h: np.ndarray, pct: np.ndarray, window: tuple[float, float]
) -> tuple[float, int]:
    """Least-squares slope of % released against time inside ``window`` (%/h)."""
    t, y = _prepare(time_h, pct)
    mask = (t >= window[0]) & (t <= window[1])
    tw, yw = t[mask], y[mask]
    if len(tw) < 2 or float(np.ptp(tw)) == 0.0:
        return float("nan"), len(tw)
    slope = float(np.polyfit(tw, yw, 1)[0])
    return slope, len(tw)


def extract_metrics(time_h: np.ndarray, pct: np.ndarray) -> ProfileMetrics:
    """Compute every AC2 metric for one profile."""
    t, y = _prepare(time_h, pct)
    if len(t) == 0:
        return ProfileMetrics()

    release_times = {
        f"t{int(level)}": time_to_percent(t, y, level) for level in RELEASE_LEVELS
    }

    pct_at: dict[str, float] = {}
    pct_valid: dict[str, bool] = {}
    for target in SAMPLE_TIMES_H:
        value, ok = percent_at(t, y, target)
        pct_at[f"pct_{int(target)}h"] = value
        pct_valid[f"pct_{int(target)}h"] = ok

    mdt, truncated, note = mean_dissolution_time(t, y)
    early, n_early = _window_slope(t, y, EARLY_WINDOW_H)
    late, n_late = _window_slope(t, y, LATE_WINDOW_H)

    ratio = float("nan")
    ratio_ok = False
    if np.isfinite(early) and np.isfinite(late) and abs(late) > 1e-9:
        ratio = early / late
        ratio_ok = True

    return ProfileMetrics(
        release_times=release_times,
        pct_at=pct_at,
        pct_at_valid=pct_valid,
        mdt_h=mdt,
        mdt_truncated=truncated,
        mdt_note=note,
        early_slope=early,
        early_slope_n=n_early,
        late_slope=late,
        late_slope_n=n_late,
        slope_ratio=ratio,
        slope_ratio_valid=ratio_ok,
        fits=fit_all(t, y),
        peak_pct=float(np.max(y)),
        n_points=len(t),
        censored_at_80=release_times["t80"].censored,
    )
