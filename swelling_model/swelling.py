"""Swelling boundaries for the dynamic models (D1, D2).

Preloaded defaults are the fitted parameters of the drug-free (placebo) HPMC tablets
used in the MS5 revision, so D1/D2 can run on a machine that only has the initial
release-tablet dimensions. Only derived parameters are stored here, never raw
measurements.

The defaults are transferred to a new tablet in "scaled" mode, as in the paper:

    R0, Z0  <- the release tablet's own initial dimensions
    Rf, Zf  <- R0 * (Rf/R0)_source,  Z0 * (Zf/Z0)_source
    kr, kz  <- source rates (optionally thickness-scaled, see ``rate_scaling``)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .model import Swelling, swell_curve


@dataclass(frozen=True)
class SwellingDefaults:
    """Fitted swelling of a placebo HPMC tablet, as dimensionless ratios plus rates."""

    grade: str
    Rf_R0: float
    Zf_Z0: float
    kr_h: float
    kz_h: float
    source_R0_mm: float
    source_Z0_mm: float  # half-height
    source_mass_mg: float
    t_max_h: float
    r2_radial: float
    r2_axial: float
    note: str


#: Primary swelling per grade used for the MS5 tables (fits/swelling_params.csv,
#: PRIMARY_SWELL in nondim_batch_fit.py). Pure HPMC, Carver press, 12.7 mm tooling.
DEFAULT_SWELLING: dict[str, SwellingDefaults] = {
    "K15M": SwellingDefaults(
        "K15M", Rf_R0=1.3561, Zf_Z0=4.6460, kr_h=0.2508, kz_h=0.3096,
        source_R0_mm=6.358, source_Z0_mm=1.395, source_mass_mg=400, t_max_h=12.0,
        r2_radial=0.968, r2_axial=0.970,
        note="400 mg tablet, same geometry as the release tablets (Table 1 of MS5).",
    ),
    "K100M": SwellingDefaults(
        "K100M", Rf_R0=1.4343, Zf_Z0=4.7859, kr_h=0.6977, kz_h=2.2055,
        source_R0_mm=6.375, source_Z0_mm=0.758, source_mass_mg=200, t_max_h=14.0,
        r2_radial=0.945, r2_axial=0.956,
        note="200 mg (half-thickness) tablet; 12-20 h points dropped (erosion).",
    ),
    "K100LV": SwellingDefaults(
        "K100LV", Rf_R0=1.2639, Zf_Z0=3.7479, kr_h=1.4417, kz_h=2.3411,
        source_R0_mm=6.375, source_Z0_mm=0.758, source_mass_mg=200, t_max_h=3.0,
        r2_radial=0.979, r2_axial=0.950,
        note="200 mg tablet; only 0-3 h usable (>= 8 h dropped, the gel erodes). "
             "Rf, Zf are extrapolated asymptotes.",
    ),
}

#: Not used for MS5 results; kept for other grades and for the rate-transfer check.
EXTRA_SWELLING: dict[str, SwellingDefaults] = {
    "K4M": SwellingDefaults(
        "K4M", Rf_R0=1.4440, Zf_Z0=4.9637, kr_h=0.4200, kz_h=1.0484,
        source_R0_mm=6.375, source_Z0_mm=0.758, source_mass_mg=200, t_max_h=8.0,
        r2_radial=0.989, r2_axial=0.981,
        note="200 mg tablet; 12-20 h points dropped. No K4M release data in MS5.",
    ),
    "K15M@200mg": SwellingDefaults(
        "K15M", Rf_R0=1.3273, Zf_Z0=4.2645, kr_h=0.4213, kz_h=1.0649,
        source_R0_mm=6.350, source_Z0_mm=0.725, source_mass_mg=200, t_max_h=8.0,
        r2_radial=0.975, r2_axial=0.971,
        note="Same K15M as the 400 mg default, half the thickness. Shows how much the "
             "rates depend on tablet thickness (see rate_scaling).",
    ),
}

RATE_SCALING = ("none", "axial_L2")


def swelling_for_tablet(
    grade: str,
    R0: float,
    Z0: float,
    *,
    rate_scaling: str = "none",
    defaults: SwellingDefaults | None = None,
    kr_h: float | None = None,
    kz_h: float | None = None,
) -> Swelling:
    """Swelling boundaries for a release tablet with initial radius R0, half-height Z0.

    rate_scaling:
      "none"      source rates used as-is (what MS5 did; reproduces the paper).
      "axial_L2"  kz scaled by (Z0_source/Z0)², i.e. axial hydration time ~ thickness².
                  On K15M this predicts the 400 mg kz from the 200 mg one to within 7 %
                  (0.288 vs 0.310 1/h); "none" is off by 3.4x. kr is left unchanged
                  (the diameter is the same); it still runs ~1.7x fast for a 2x
                  thicker tablet, so treat it as approximate.
    kr_h / kz_h override the rates outright (e.g. from your own swelling data).
    """
    d = defaults or DEFAULT_SWELLING.get(grade) or EXTRA_SWELLING.get(grade)
    if d is None:
        known = sorted(DEFAULT_SWELLING) + sorted(EXTRA_SWELLING)
        raise KeyError(f"no swelling defaults for grade {grade!r}; known: {known}. "
                       "Use S1/S2 (no swelling needed) or fit_swelling() on your own data.")
    if rate_scaling not in RATE_SCALING:
        raise ValueError(f"rate_scaling must be one of {RATE_SCALING}")
    kz = d.kz_h * ((d.source_Z0_mm / Z0) ** 2 if rate_scaling == "axial_L2" else 1.0)
    return Swelling(
        R0=R0, Z0=Z0, Rf=R0 * d.Rf_R0, Zf=Z0 * d.Zf_Z0,
        kr_h=d.kr_h if kr_h is None else kr_h, kz_h=kz if kz_h is None else kz_h,
        source=f"{grade} default ({d.source_mass_mg:g} mg placebo, rates {rate_scaling})",
    )


def fit_swelling_curve(t_h: np.ndarray, X: np.ndarray, X0: float) -> tuple[float, float, float]:
    """Fit k [1/h] and Xf with X0 fixed. Returns (k_h, Xf, R²)."""
    t_h, X = np.asarray(t_h, float), np.asarray(X, float)
    p, _ = curve_fit(
        lambda t, k, Xf: swell_curve(t, X0, Xf, k), t_h, X,
        p0=[0.5, max(X.max(), X0 * 1.01)], bounds=([0.0, X0], [60.0, 100.0]),
    )
    resid = X - swell_curve(t_h, X0, p[1], p[0])
    r2 = 1 - np.sum(resid**2) / np.sum((X - X.mean()) ** 2)
    return float(p[0]), float(p[1]), float(r2)


def fit_swelling(
    df: pd.DataFrame,
    grade: str,
    *,
    mass_mg: float = float("nan"),
    drop_after_h: float | None = None,
) -> SwellingDefaults:
    """Fit measured swelling of one grade into a SwellingDefaults record.

    ``df`` columns: time_h, diameter_mm, height_mm (full height), one row per tablet
    and time. t = 0 rows (dry dimensions) are required: X0 is fixed at their mean.
    Use ``drop_after_h`` to cut late points where the gel has started to erode
    (dimensions shrink, which the saturation curve cannot represent).
    """
    d = df.dropna(subset=["time_h", "diameter_mm", "height_mm"])
    if drop_after_h is not None:
        d = d[d["time_h"] <= drop_after_h]
    st = d.groupby("time_h")[["diameter_mm", "height_mm"]].mean().reset_index()
    if 0 not in st["time_h"].to_numpy() or len(st) < 4:
        raise ValueError("need t = 0 dry dimensions and >= 4 time points")
    t = st["time_h"].to_numpy(float)
    R, Z = st["diameter_mm"].to_numpy() / 2, st["height_mm"].to_numpy() / 2
    kr, Rf, r2r = fit_swelling_curve(t, R, R[0])
    kz, Zf, r2z = fit_swelling_curve(t, Z, Z[0])
    return SwellingDefaults(
        grade, Rf_R0=Rf / R[0], Zf_Z0=Zf / Z[0], kr_h=kr, kz_h=kz,
        source_R0_mm=float(R[0]), source_Z0_mm=float(Z[0]), source_mass_mg=mass_mg,
        t_max_h=float(t.max()), r2_radial=r2r, r2_axial=r2z, note="fitted from user data",
    )
