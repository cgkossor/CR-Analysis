"""2D analytical swelling-cylinder release model (D1, D2, S1, S2). See README.md."""

from .delay import (
    DelayFit,
    delay_probe,
    fit_delay,
    fraction_released_k,
    fraction_released_lag,
)
from .fitting import (
    Extrapolation,
    FitResult,
    extrapolate,
    f2,
    fit_all,
    fit_model,
    profile_ci,
    score,
    train_window_scan,
)
from .model import (
    MODEL_LABELS,
    MODELS,
    Geometry,
    Swelling,
    alpha_static,
    diffusivity,
    doubling_time_h,
    fraction_released,
    time_to_fraction,
)
from .swelling import DEFAULT_SWELLING, EXTRA_SWELLING, fit_swelling, swelling_for_tablet

__all__ = [
    "DEFAULT_SWELLING",
    "EXTRA_SWELLING",
    "MODELS",
    "MODEL_LABELS",
    "DelayFit",
    "Extrapolation",
    "FitResult",
    "Geometry",
    "Swelling",
    "alpha_static",
    "delay_probe",
    "diffusivity",
    "doubling_time_h",
    "extrapolate",
    "f2",
    "fit_all",
    "fit_delay",
    "fit_model",
    "fit_swelling",
    "fraction_released",
    "fraction_released_k",
    "fraction_released_lag",
    "profile_ci",
    "score",
    "swelling_for_tablet",
    "time_to_fraction",
    "train_window_scan",
]
