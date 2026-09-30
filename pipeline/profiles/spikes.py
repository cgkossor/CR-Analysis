"""Remove momentary spikes from measured profiles.

An in-situ probe occasionally reads a bubble on the window, or picks up
electrical interference, as a single reading far off the curve that the next
reading ignores. Left in, such a point moves t50, bends a Weibull fit, and
draws a needle on every plot.

The test is deliberately conservative. Within each replicate, a reading is
compared with the median of the readings around it. A monotone release curve
equals its own rolling median, so a fast rise does not register. A reading is
removed only when all of the following hold:

* it is further from that median than both an absolute floor
  (:data:`config.SPIKE_MIN_PCT`, % released) and a multiple of the replicate's
  own robust noise level (:data:`config.SPIKE_NOISE_K` x MAD);
* it belongs to a run of at most :data:`config.SPIKE_MAX_RUN` consecutive
  flagged readings, because a longer excursion is behaviour, not a glitch;
* the run sits above both neighbouring readings, or below both, by more than
  that same threshold, so it jumps away and comes back;
* it is neither the first nor the last reading of the replicate.

A replicate with fewer than :data:`config.SPIKE_MIN_READINGS` readings is left
alone: with few, widely spaced pulls a high point cannot be told apart from a
real step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import config

#: Columns of the removal log returned by :func:`remove_spikes`.
LOG_COLUMNS = (
    "id", "case", "grade", "replicate", "time_h", "pct_released", "local_median",
    "deviation", "threshold",
)


def spike_mask(values: np.ndarray) -> np.ndarray:
    """True for each reading judged a momentary spike. ``values`` in time order."""
    return _detect(values)[0]


def _detect(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(flags, rolling median, threshold) for one replicate in time order."""
    y = np.asarray(values, dtype=float)
    n = y.size
    flags = np.zeros(n, dtype=bool)
    if n < config.SPIKE_MIN_READINGS:
        return flags, np.full(n, np.nan), float("nan")

    median = (
        pd.Series(y)
        .rolling(config.SPIKE_WINDOW, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    resid = y - median
    finite = np.isfinite(resid)
    if not finite.any():
        return flags, median, float("nan")
    noise = 1.4826 * float(np.median(np.abs(resid[finite])))
    threshold = max(config.SPIKE_MIN_PCT, config.SPIKE_NOISE_K * noise)
    candidate = finite & (np.abs(resid) > threshold)
    candidate[0] = candidate[-1] = False

    i = 0
    while i < n:
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j < n and candidate[j]:
            j += 1
        if j - i <= config.SPIKE_MAX_RUN and _is_excursion(y, i, j, threshold):
            flags[i:j] = True
        i = j
    return flags, median, threshold


def _is_excursion(y: np.ndarray, i: int, j: int, threshold: float) -> bool:
    """Readings ``i:j`` jump away from BOTH neighbouring readings and come back.

    A point on a monotone rise always lies between its neighbours, so this is
    what keeps a steep early release from ever counting as a spike.
    """
    before, after = y[i - 1], y[j]
    run = y[i:j]
    up = bool(np.all(run > max(before, after) + threshold))
    down = bool(np.all(run < min(before, after) - threshold))
    return up or down


def remove_spikes(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop spike readings from a long profile frame.

    Returns the cleaned frame and a log with one row per removed reading (see
    :data:`LOG_COLUMNS`): its value, the local median it was judged against,
    how far off it was and the threshold it crossed. ``long`` must be sorted by
    id, replicate and time, as the loader leaves it.
    """
    empty = pd.DataFrame(columns=list(LOG_COLUMNS))
    if not config.SPIKE_FILTER or long.empty:
        return long, empty
    values = long["pct_released"].to_numpy(dtype=float)
    drop = np.zeros(len(long), dtype=bool)
    medians = np.full(len(long), np.nan)
    thresholds = np.full(len(long), np.nan)
    for idx in long.groupby(["id", "replicate"], sort=False).indices.values():
        flags, median, threshold = _detect(values[idx])
        drop[idx] = flags
        medians[idx] = median
        thresholds[idx] = threshold
    log = long.loc[drop, [c for c in LOG_COLUMNS[:6] if c in long.columns]].copy()
    log["local_median"] = medians[drop]
    log["deviation"] = values[drop] - medians[drop]
    log["threshold"] = thresholds[drop]
    return long.loc[~drop].reset_index(drop=True), log.reset_index(drop=True)
