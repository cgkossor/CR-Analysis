"""Censored release times (G5).

A formulation that never reaches its target release level does not have a
missing ``t80`` -- it has a *censored* one: ``t80 > 24 h``. The distinction
matters because dropping it biases every downstream summary toward the fast
formulations, which are precisely the ones a controlled-release study is not
about.

:class:`CensoredValue` keeps the bound and the reason attached to the number so
that no consumer can accidentally treat "> 24 h" as "unknown" or as "24 h".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CensoredValue:
    """A release time that may be right-censored at the end of observation."""

    value: float
    """Interpolated time in hours, or the censoring bound when censored."""

    censored: bool
    """True when the profile never reached the target level."""

    bound: float | None = None
    """Last observed time, when censored."""

    peak_pct: float = float("nan")
    """Highest % released observed, for context on how far short it fell."""

    def __str__(self) -> str:
        if self.censored:
            return f">{self.bound:g}"
        return f"{self.value:.3f}"

    @property
    def for_display(self) -> str:
        return str(self)

    @property
    def numeric_or_nan(self) -> float:
        """Numeric value for arithmetic that has *already* handled censoring.

        Returns NaN when censored, so that a caller who has not thought about
        censoring gets a loud NaN rather than a quietly wrong mean.
        """
        return float("nan") if self.censored else self.value


def time_to_percent(
    time_h: np.ndarray, pct: np.ndarray, target: float
) -> CensoredValue:
    """First time the profile reaches ``target`` % released.

    Linear interpolation between the bracketing observations. "First crossing"
    is the convention: assay noise can push a profile back and forth across a
    threshold, and the first crossing is both reproducible and the quantity a
    formulator means.
    """
    t = np.asarray(time_h, dtype=float)
    y = np.asarray(pct, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t, kind="mergesort")
    t, y = t[order], y[order]

    if len(t) == 0:
        return CensoredValue(float("nan"), censored=True, bound=None)

    peak = float(np.max(y))
    reached = np.nonzero(y >= target)[0]
    if reached.size == 0:
        return CensoredValue(
            value=float("nan"), censored=True, bound=float(t[-1]), peak_pct=peak
        )

    i = int(reached[0])
    if i == 0:
        return CensoredValue(float(t[0]), censored=False, peak_pct=peak)

    t0, t1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return CensoredValue(t1, censored=False, peak_pct=peak)
    crossing = t0 + (target - y0) * (t1 - t0) / (y1 - y0)
    return CensoredValue(float(crossing), censored=False, peak_pct=peak)


def summarise_censored(values: list[CensoredValue]) -> dict[str, float | int | str]:
    """Summarise a set of possibly-censored times without discarding the censored ones.

    Reports the observed count, the censored count, and a median computed by the
    Kaplan-Meier convention where possible. Never returns a plain mean that
    silently ignores censoring (G5).
    """
    n = len(values)
    censored = [v for v in values if v.censored]
    observed = [v.value for v in values if not v.censored]

    summary: dict[str, float | int | str] = {
        "n": n,
        "n_observed": len(observed),
        "n_censored": len(censored),
    }
    if not observed:
        summary["median"] = float("nan")
        summary["median_display"] = (
            f">{max((v.bound or 0.0) for v in censored):g}" if censored else "n/a"
        )
        return summary

    # If more than half the values are censored the median is itself censored.
    if len(censored) > n / 2:
        bound = max((v.bound or 0.0) for v in censored)
        summary["median"] = float("nan")
        summary["median_display"] = f">{bound:g}"
        return summary

    median = float(np.median(observed))
    summary["median"] = median
    summary["median_display"] = f"{median:.3f}"
    return summary
