"""f2 similarity with an explicit, defensible point-selection rule.

``f2 = 50 * log10( 100 / sqrt(1 + mean((R_t - T_t)^2)) )``

The statistic is only as meaningful as the timepoints fed into it, and this
database makes that acute: the sampling schedule is deliberately front-loaded,
with 16 of 25 samples inside the first two hours. Passing all 25 points to f2
would weight the first two hours eight times more heavily than the remaining
twenty-two, so two profiles differing only in their plateau would score as
similar while two differing only in burst would not.

The rule applied here follows the regulatory convention and is stated rather than
buried:

* only timepoints where **both** profiles are measured;
* **at most one point beyond 85% release** on either profile, so the plateau
  cannot dominate;
* a minimum of 3 usable points, else no f2 is reported at all.

The selected points travel with the result so any consumer can audit them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import config


@dataclass(frozen=True)
class F2Result:
    """An f2 comparison, with the points it was computed on."""

    value: float
    valid: bool
    n_points: int
    times_used: tuple[float, ...] = field(default=())
    note: str = ""

    @property
    def similar(self) -> bool:
        """True only when f2 is both valid and at or above the threshold."""
        return self.valid and self.value >= config.F2_SIMILAR_THRESHOLD


def select_f2_points(
    time_h: np.ndarray, reference: np.ndarray, test: np.ndarray
) -> np.ndarray:
    """Indices of the timepoints f2 should be computed on."""
    t = np.asarray(time_h, dtype=float)
    r = np.asarray(reference, dtype=float)
    s = np.asarray(test, dtype=float)

    usable = np.isfinite(t) & np.isfinite(r) & np.isfinite(s) & (t > 0)
    idx = np.flatnonzero(usable)
    if idx.size == 0:
        return idx

    order = idx[np.argsort(t[idx], kind="mergesort")]
    plateau = config.F2_PLATEAU_PCT

    kept: list[int] = []
    beyond_plateau = 0
    for i in order:
        past = (r[i] > plateau) or (s[i] > plateau)
        if past:
            beyond_plateau += 1
            if beyond_plateau > 1:
                # Keep exactly one point past the plateau, then stop.
                break
        kept.append(int(i))
    return np.asarray(kept, dtype=int)


def similarity_f2(
    time_h: np.ndarray, reference: np.ndarray, test: np.ndarray
) -> F2Result:
    """Compute f2 between two profiles on a common time grid."""
    idx = select_f2_points(time_h, reference, test)
    if idx.size < config.F2_MIN_POINTS:
        return F2Result(
            value=float("nan"),
            valid=False,
            n_points=int(idx.size),
            note=(
                f"only {idx.size} usable timepoint(s) after applying the "
                f"one-point-past-{config.F2_PLATEAU_PCT:g}% rule; f2 requires at least "
                f"{config.F2_MIN_POINTS} and is not reported"
            ),
        )

    r = np.asarray(reference, dtype=float)[idx]
    s = np.asarray(test, dtype=float)[idx]
    mean_sq = float(np.mean((r - s) ** 2))
    value = 50.0 * np.log10(100.0 / np.sqrt(1.0 + mean_sq))

    return F2Result(
        value=float(value),
        valid=True,
        n_points=int(idx.size),
        times_used=tuple(float(v) for v in np.asarray(time_h, dtype=float)[idx]),
        note=f"computed on {idx.size} points, one at most beyond "
             f"{config.F2_PLATEAU_PCT:g}% release",
    )
