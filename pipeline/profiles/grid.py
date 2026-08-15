"""Reconcile timepoints that are nominally the same but not exactly equal.

Real studies do not sample at exact times. A nominal 60-minute pull happens at
59.5 or 61 minutes depending on the operator and the day, so every formulation
ends up with its own time vector even though the *design* has one schedule.

Comparing profiles by exact timestamp then fails silently: aligning them on the
union of all observed times leaves a gap wherever a profile has no sample at
another profile's exact time, and any statistic computed across the whole vector
comes back NaN. The f2 calculation happens to skip non-finite points and keeps
working, which is worse than failing — two numbers that should agree stop
agreeing, and nothing says why.

The fix is to recognise a nominal schedule: cluster observed times that fall
within a tolerance of one another, take each cluster's median as the canonical
time, then interpolate every profile onto that grid. Interpolation stays inside
each profile's own observed range; beyond it the value is left missing rather
than extrapolated (G3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import config


@dataclass(frozen=True)
class TimeGrid:
    """A canonical sampling schedule recovered from ragged observed times."""

    times_h: np.ndarray
    """Canonical timepoints, ascending."""

    n_raw_times: int
    """How many distinct timestamps appeared in the raw data."""

    n_distinct_vectors: int
    """How many different time vectors the profiles had."""

    max_shift_h: float
    """Largest distance from a raw time to its canonical time."""

    collapsed: bool
    """True when clustering actually merged anything."""

    notes: tuple[str, ...] = field(default=())

    @property
    def n_points(self) -> int:
        return len(self.times_h)


def _tolerance(t: float) -> float:
    """Half-width within which two timestamps count as the same nominal sample.

    Absolute near zero, proportional later: a one-minute slip at the 2-minute
    pull is a different timepoint, while at the 24-hour pull it is nothing.
    """
    return max(
        config.TIME_CLUSTER_ABS_H,
        config.TIME_CLUSTER_REL * abs(t),
    )


def build_time_grid(
    time_vectors: list[np.ndarray],
) -> TimeGrid:
    """Recover the canonical schedule from a set of observed time vectors."""
    all_times = np.sort(
        np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in time_vectors])
    )
    all_times = all_times[np.isfinite(all_times)]
    if all_times.size == 0:
        return TimeGrid(np.array([]), 0, 0, 0.0, False, ("no finite timepoints",))

    distinct_raw = np.unique(all_times)
    signatures = {tuple(np.round(np.unique(np.asarray(v, dtype=float)), 9)) for v in time_vectors}

    # Gap-based clustering alone chains: with jittered samples the 15, 18 and
    # 22 minute pulls form one unbroken run of close values and collapse into a
    # single point, destroying real resolution. So a cluster is also capped at
    # twice the tolerance wide -- a point only joins if it is close to the
    # *cluster*, not merely to its most recent member.
    clusters: list[list[float]] = [[float(distinct_raw[0])]]
    for value in distinct_raw[1:]:
        current = clusters[-1]
        tol = _tolerance(value)
        near_previous = value - current[-1] <= tol
        stays_narrow = value - current[0] <= 2.0 * tol
        if near_previous and stays_narrow:
            current.append(float(value))
        else:
            clusters.append([float(value)])

    canonical = np.array([float(np.median(c)) for c in clusters])
    shifts = [abs(v - float(np.median(c))) for c in clusters for v in c]
    max_shift = float(max(shifts)) if shifts else 0.0
    collapsed = len(canonical) < len(distinct_raw)

    # How many samples a profile actually has. If clustering produced markedly
    # fewer canonical points than that, it merged distinct nominal timepoints and
    # the comparison grid is coarser than the study design -- worth saying out
    # loud rather than leaving in the arithmetic.
    per_profile = [int(np.unique(np.asarray(v, dtype=float)).size) for v in time_vectors]
    typical = int(np.median(per_profile)) if per_profile else 0

    notes: list[str] = []
    if collapsed:
        notes.append(
            f"{len(distinct_raw)} distinct timestamps across {len(signatures)} different "
            f"time vectors were reconciled to a {len(canonical)}-point nominal schedule; "
            f"the largest adjustment was {max_shift * 60:.2f} min. Profiles are "
            "interpolated onto this schedule for comparison. Per-profile metrics are "
            "still computed on each profile's own measured times, so this affects "
            "profile comparisons only."
        )
    elif len(signatures) > 1:
        notes.append(
            f"{len(signatures)} different time vectors, none close enough to reconcile "
            f"within the tolerance. Comparisons use the union of {len(canonical)} "
            "timepoints, and profiles are interpolated where they have coverage."
        )

    if typical and len(canonical) < typical:
        notes.append(
            f"WARNING: the reconciled schedule has {len(canonical)} points but a typical "
            f"profile was sampled {typical} times, so distinct nominal timepoints have "
            "been merged. Profile comparisons are running at lower time resolution than "
            "the study was designed for. If the sampling jitter is smaller than the "
            "spacing between planned pulls, lower TIME_CLUSTER_REL in pipeline/config.py "
            "and re-run."
        )

    return TimeGrid(
        times_h=canonical,
        n_raw_times=len(distinct_raw),
        n_distinct_vectors=len(signatures),
        max_shift_h=max_shift,
        collapsed=collapsed,
        notes=tuple(notes),
    )


def project_onto_grid(
    times_h: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Interpolate one profile onto ``grid``, without extrapolating.

    Grid points outside the profile's observed range come back NaN rather than
    being filled with the nearest value — a profile that stopped at 12 h has no
    opinion about 24 h, and inventing one would be exactly the silent
    extrapolation G3 forbids.
    """
    t = np.asarray(times_h, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    if t.size == 0:
        return np.full(len(grid), np.nan)

    order = np.argsort(t, kind="mergesort")
    t, y = t[order], y[order]

    out = np.interp(grid, t, y, left=np.nan, right=np.nan)
    # np.interp clamps rather than extrapolating; make out-of-range explicit.
    out = np.where((grid < t[0] - 1e-12) | (grid > t[-1] + 1e-12), np.nan, out)
    return out


def paired_finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The points where both profiles have a value. Used for every comparison."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]
