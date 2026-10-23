"""Analytical 2D release model for a swelling cylinder (Kossor et al., MS5, Pharm Res).

Advection-diffusion in a finite cylinder of radius R(t) and half-height Z(t), perfect
sink on the outer surfaces, uniform initial drug, spatially uniform D(t) = D0 + γt.
The solution is an infinite double series whose time dependence enters only through
two diffusion-time integrals,

    I_r(t) = ∫ D(s) / R(s)² ds,   I_z(t) = ∫ D(s) / Z(s)² ds,

and the fraction released is (separable form of the double sum)

    M(t) = 1 - [Σ_n 4/λ_n² exp(-λ_n² I_r)] [Σ_m 2/μ_m² exp(-μ_m² I_z)]

with λ_n the zeros of J0 and μ_m = (2m+1)π/2. The two integrals are closed form for
the four cases:

    D1  dynamic boundaries (swelling),  D = D0 + γt
    D2  dynamic boundaries (swelling),  D = D0
    S1  static boundaries (R0, Z0),     D = D0 + γt
    S2  static boundaries (R0, Z0),     D = D0

Swelling follows X(t) = Xf + (X0 - Xf)/(1 + k t) independently in r and z.

Units used throughout: mm, s for the physics; the public API takes time in hours.
Z is always the HALF-height (tablet thickness / 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jn_zeros

#: Series terms per direction. Truncation leaves M(0) = 0.608/N (0.3 % at 200); the
#: error is < 0.01 pp after ~15 min of release, so this only matters at t -> 0.
N_TERMS = 200

_LAM2 = jn_zeros(0, N_TERMS) ** 2
_MU2 = ((2 * np.arange(N_TERMS) + 1) * np.pi / 2) ** 2

MODELS = ("D1", "D2", "S1", "S2")
#: (dynamic boundaries?, time-dependent D?)
MODEL_FLAGS = {"D1": (True, True), "D2": (True, False), "S1": (False, True), "S2": (False, False)}
MODEL_LABELS = {
    "D1": "Dynamic boundaries, D = D0 + γt",
    "D2": "Dynamic boundaries, D = D0",
    "S1": "Static boundaries, D = D0 + γt",
    "S2": "Static boundaries, D = D0",
}


@dataclass(frozen=True)
class Geometry:
    """Initial (dry) release-tablet dimensions. Z0 is the half-height."""

    R0: float  # mm
    Z0: float  # mm

    @classmethod
    def from_tablet(cls, diameter_mm: float, thickness_mm: float) -> Geometry:
        return cls(R0=diameter_mm / 2.0, Z0=thickness_mm / 2.0)

    @property
    def beta(self) -> float:
        """Aspect ratio β = Z0/R0 (0.5 = equal radial and axial surface area)."""
        return self.Z0 / self.R0


@dataclass(frozen=True)
class Swelling:
    """Swelling boundaries for D1/D2. Rf/Zf are asymptotes; rates in 1/h."""

    R0: float
    Z0: float
    Rf: float
    Zf: float
    kr_h: float
    kz_h: float
    source: str = ""

    def radius(self, t_h: np.ndarray) -> np.ndarray:
        return swell_curve(t_h, self.R0, self.Rf, self.kr_h)

    def half_height(self, t_h: np.ndarray) -> np.ndarray:
        return swell_curve(t_h, self.Z0, self.Zf, self.kz_h)


def swell_curve(t_h: np.ndarray, X0: float, Xf: float, k_h: float) -> np.ndarray:
    """First-order saturation swelling X(t) = Xf + (X0 - Xf)/(1 + k t)."""
    t_h = np.asarray(t_h, float)
    return Xf + (X0 - Xf) / (1.0 + k_h * t_h)


def series_release(Ir: np.ndarray, Iz: np.ndarray) -> np.ndarray:
    """M = 1 - S_r(I_r) S_z(I_z); identical to the double sum of the paper."""
    Ir = np.atleast_1d(np.asarray(Ir, float))[:, None]
    Iz = np.atleast_1d(np.asarray(Iz, float))[:, None]
    sr = np.sum(4.0 / _LAM2 * np.exp(-_LAM2 * Ir), axis=1)
    sz = np.sum(2.0 / _MU2 * np.exp(-_MU2 * Iz), axis=1)
    return 1.0 - sr * sz


def integral_static(t_s: np.ndarray, D0: float, gamma: float, X0: float) -> np.ndarray:
    """∫ (D0 + γs)/X0² ds = (D0 t + γ t²/2)/X0²."""
    return (D0 * t_s + gamma * t_s**2 / 2.0) / X0**2


def integral_dynamic(
    t_s: np.ndarray, D0: float, gamma: float, X0: float, Xf: float, k_s: float
) -> np.ndarray:
    """∫ (D0 + γs)/X(s)² ds with X(s) = Xf + (X0 - Xf)/(1 + k s). k in 1/s.

    Closed form of the published Eqs. (51)/(52); checked against numerical quadrature
    to 6e-12 (tests/test_swelling_model.py repeats that check).
    """
    t_s = np.asarray(t_s, float)
    if abs(Xf - X0) < 1e-12 or k_s <= 0:
        return integral_static(t_s, D0, gamma, X0)
    quad = gamma * t_s**2 / (2 * Xf**2)
    lin = t_s * (D0 / Xf**2 + 2 * gamma / (Xf**2 * k_s) - 2 * X0 * gamma / (Xf**3 * k_s))
    c_log = -(Xf - X0) * (2 * D0 * Xf * k_s + gamma * (Xf - 3 * X0)) / (Xf**4 * k_s**2)
    c_rat = (Xf - X0) ** 2 * (D0 * k_s * Xf - gamma * X0) / (Xf**4 * k_s**2)
    u = X0 + Xf * k_s * t_s
    return quad + lin + c_log * np.log(X0 / u) + c_rat * (1 / X0 - 1 / u)


def fraction_released(
    model: str,
    t_h: np.ndarray,
    D0: float,
    gamma: float,
    geometry: Geometry,
    swelling: Swelling | None = None,
) -> np.ndarray:
    """Fraction released (0-1) at times t_h [h]. D0 in mm²/s, γ in mm²/s².

    γ is ignored (forced 0) for D2 and S2. D1/D2 need ``swelling``; S1/S2 use
    ``geometry`` only and need no swelling data at all.
    """
    dynamic, time_dep = MODEL_FLAGS[model]
    g = gamma if time_dep else 0.0
    t_s = np.asarray(t_h, float) * 3600.0
    if not dynamic:
        return series_release(
            integral_static(t_s, D0, g, geometry.R0), integral_static(t_s, D0, g, geometry.Z0)
        )
    if swelling is None:
        raise ValueError(f"{model} needs swelling boundaries (see swelling_model.swelling)")
    sw = swelling
    return series_release(
        integral_dynamic(t_s, D0, g, sw.R0, sw.Rf, sw.kr_h / 3600.0),
        integral_dynamic(t_s, D0, g, sw.Z0, sw.Zf, sw.kz_h / 3600.0),
    )


def diffusivity(t_h: np.ndarray, D0: float, gamma: float) -> np.ndarray:
    """Effective D(t) = D0 + γt in mm²/s (unbounded and linear by construction)."""
    return D0 + gamma * np.asarray(t_h, float) * 3600.0


def doubling_time_h(D0: float, gamma: float) -> float:
    """t* = D0/γ [h]: time at which D has doubled. Small t* = γ-dominated release."""
    return float("inf") if gamma <= 0 else D0 / gamma / 3600.0


def alpha_static(D0: float, gamma: float, R0: float) -> float:
    """α = γR0²/D0². Defined for the STATIC models (S1/S2) only."""
    return gamma * R0**2 / D0**2 if D0 > 0 else float("nan")


def time_to_fraction(
    model: str,
    target: float,
    D0: float,
    gamma: float,
    geometry: Geometry,
    swelling: Swelling | None = None,
    t_max_h: float = 500.0,
) -> float:
    """First time [h] the model reaches ``target`` (e.g. 0.8 for t80); NaN if never."""
    t = np.linspace(0.0, t_max_h, 20001)
    m = fraction_released(model, t, D0, gamma, geometry, swelling)
    i = int(np.argmax(m >= target))
    if m[i] < target:
        return float("nan")
    if i == 0:
        return 0.0
    return float(np.interp(target, [m[i - 1], m[i]], [t[i - 1], t[i]]))
