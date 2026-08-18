"""Kinetic model fits for a single dissolution profile.

Four models, all fitted with fixed starting values and no random restarts, so
two runs give identical numbers (G10):

* **Weibull** ``F(t) = F_inf * (1 - exp(-(t/Td)^beta))`` -- the primary model.
  ``F_inf`` is a *free* parameter, not pinned at 100. Matrix formulations that
  never fully release have an asymptote below 100, and fixing it there biases
  both ``Td`` and ``beta`` on exactly the slow formulations that matter most.
* **Peppas** ``F(t) = k * t^n`` -- fitted on the <=60% released portion only and
  only when at least 3 such points exist (G4). ``n`` is never returned otherwise.
* **Higuchi** ``F(t) = k * sqrt(t)`` and **first-order**
  ``F(t) = F_inf * (1 - exp(-k t))`` -- for model-comparison context (AC2).

Every result carries ``valid`` plus a human-readable ``note`` explaining any
refusal, so a caller can never silently mistake "not fitted" for "fitted badly".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from pipeline import config


@dataclass(frozen=True)
class FitResult:
    """Outcome of fitting one kinetic model to one profile."""

    model: str
    params: dict[str, float] = field(default_factory=dict)
    r_squared: float = float("nan")
    rmse: float = float("nan")
    n_points: int = 0
    valid: bool = False
    note: str = ""

    asymptote_identified: bool = True
    """Weibull only: False when the profile never approaches its fitted plateau,
    so ``f_inf`` and ``td`` are extrapolated rather than estimated."""


def _goodness(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    resid = observed - predicted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(ss_res / len(observed))) if len(observed) else float("nan")
    return r2, rmse


def _clean(time_h: np.ndarray, pct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop non-finite pairs and sort by time. Values are never altered."""
    t = np.asarray(time_h, dtype=float)
    y = np.asarray(pct, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t, kind="mergesort")
    return t[order], y[order]


def weibull(t: np.ndarray, f_inf: float, td: float, beta: float) -> np.ndarray:
    """Weibull release with a free asymptote. ``t`` in hours."""
    safe_t = np.maximum(np.asarray(t, dtype=float), 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return f_inf * (1.0 - np.exp(-((safe_t / td) ** beta)))


def fit_weibull(time_h: np.ndarray, pct: np.ndarray) -> FitResult:
    """Fit the 3-parameter Weibull. Requires at least 4 points."""
    t, y = _clean(time_h, pct)
    if len(t) < 4:
        return FitResult("weibull", n_points=len(t), note="fewer than 4 finite points")

    peak = float(np.max(y))
    if peak <= 0:
        return FitResult("weibull", n_points=len(t), note="no release observed")

    # Deterministic starting values derived from the data itself.
    half = peak / 2.0
    idx = int(np.argmax(y >= half)) if np.any(y >= half) else len(t) - 1
    td0 = max(float(t[idx]), 1e-3)
    # Data-aware ceiling. A flat cap would clamp a profile that genuinely reached
    # 110% -- ordinary for a real assay near plateau -- biasing F_inf and Td down
    # on exactly the formulations that released most fully. The bound still exists
    # to stop a censored profile running its asymptote off to nothing physical.
    ceiling = max(
        config.MAX_PHYSICAL_RELEASE_PCT,
        peak * config.ASYMPTOTE_CEILING_MARGIN,
    )
    p0 = [min(max(peak, 1.0), ceiling), td0, 0.75]
    bounds = ([1.0, 1e-4, 0.05], [ceiling, 1e4, 10.0])

    try:
        popt, _ = curve_fit(weibull, t, y, p0=p0, bounds=bounds, maxfev=20000)
    except (RuntimeError, ValueError) as exc:
        return FitResult("weibull", n_points=len(t), note=f"optimiser failed: {exc}")

    f_inf, td, beta = (float(v) for v in popt)
    r2, rmse = _goodness(y, weibull(t, f_inf, td, beta))

    # A profile that never climbs near its own asymptote does not determine it.
    # F_inf, Td and beta then trade off against one another and the reported Td is
    # an extrapolation. This is the normal state of affairs for a formulation that
    # never reaches 80% release, so it is flagged rather than treated as a failure.
    identified = bool(peak >= config.ASYMPTOTE_IDENTIFIED_FRACTION * f_inf)
    at_bound = bool(f_inf >= ceiling - 1e-6)

    notes: list[str] = []
    if not identified:
        notes.append(
            f"asymptote NOT identified: profile peaks at {peak:.1f}% against a fitted "
            f"F_inf of {f_inf:.1f}%, so F_inf and Td are extrapolated, not estimated"
        )
    if at_bound:
        notes.append(
            f"F_inf pinned at its upper bound ({ceiling:.1f}%); the data do not "
            "constrain the asymptote from below it"
        )

    return FitResult(
        model="weibull",
        params={"f_inf": f_inf, "td": td, "beta": beta},
        r_squared=r2,
        rmse=rmse,
        n_points=len(t),
        valid=True,
        asymptote_identified=identified,
        note="; ".join(notes),
    )


def fit_peppas(time_h: np.ndarray, pct: np.ndarray) -> FitResult:
    """Fit ``F = k t^n`` on the <=60% portion only (G4).

    Returns an invalid result -- with no ``n`` -- when fewer than
    :data:`config.PEPPAS_MIN_POINTS` points fall in that window. The exponent is
    never reported for a fit that violates G4.
    """
    t, y = _clean(time_h, pct)
    window = (t > 0) & (y > 0) & (y <= config.PEPPAS_MAX_PCT)
    tw, yw = t[window], y[window]

    if len(tw) < config.PEPPAS_MIN_POINTS:
        return FitResult(
            "peppas",
            n_points=len(tw),
            note=(
                f"only {len(tw)} point(s) at or below {config.PEPPAS_MAX_PCT:g}% released; "
                f"G4 requires >= {config.PEPPAS_MIN_POINTS}. No k or n reported."
            ),
        )

    # Log-log linearisation: deterministic, closed form, no optimiser.
    slope, intercept = np.polyfit(np.log(tw), np.log(yw), 1)
    k = float(np.exp(intercept))
    n = float(slope)
    r2, rmse = _goodness(yw, k * tw**n)
    return FitResult(
        model="peppas",
        params={"k": k, "n": n},
        r_squared=r2,
        rmse=rmse,
        n_points=len(tw),
        valid=True,
        note=f"fitted on {len(tw)} points at or below {config.PEPPAS_MAX_PCT:g}% released",
    )


def fit_higuchi(time_h: np.ndarray, pct: np.ndarray) -> FitResult:
    """Fit ``F = k sqrt(t)`` through the origin."""
    t, y = _clean(time_h, pct)
    mask = t > 0
    tm, ym = t[mask], y[mask]
    if len(tm) < 2:
        return FitResult("higuchi", n_points=len(tm), note="fewer than 2 positive-time points")

    root = np.sqrt(tm)
    k = float(np.sum(root * ym) / np.sum(root**2))
    r2, rmse = _goodness(ym, k * root)
    return FitResult(
        "higuchi", {"k": k}, r2, rmse, len(tm), valid=True
    )


def fit_first_order(time_h: np.ndarray, pct: np.ndarray) -> FitResult:
    """Fit ``F = F_inf (1 - exp(-k t))``."""
    t, y = _clean(time_h, pct)
    if len(t) < 3:
        return FitResult("first_order", n_points=len(t), note="fewer than 3 finite points")

    def model(tt: np.ndarray, f_inf: float, k: float) -> np.ndarray:
        return f_inf * (1.0 - np.exp(-k * np.maximum(tt, 0.0)))

    peak = float(np.max(y))
    if peak <= 0:
        return FitResult("first_order", n_points=len(t), note="no release observed")

    try:
        popt, _ = curve_fit(
            model, t, y, p0=[max(peak, 1.0), 0.5], bounds=([1.0, 1e-6], [200.0, 1e3]), maxfev=20000
        )
    except (RuntimeError, ValueError) as exc:
        return FitResult("first_order", n_points=len(t), note=f"optimiser failed: {exc}")

    f_inf, k = (float(v) for v in popt)
    r2, rmse = _goodness(y, model(t, f_inf, k))
    return FitResult(
        "first_order", {"f_inf": f_inf, "k": k}, r2, rmse, len(t), valid=True
    )


def fit_all(time_h: np.ndarray, pct: np.ndarray) -> dict[str, FitResult]:
    """Fit every kinetic model to one profile."""
    return {
        "weibull": fit_weibull(time_h, pct),
        "peppas": fit_peppas(time_h, pct),
        "higuchi": fit_higuchi(time_h, pct),
        "first_order": fit_first_order(time_h, pct),
    }
