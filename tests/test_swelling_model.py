"""The swelling-cylinder model package (swelling_model/), on synthetic data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from swelling_model import (
    DEFAULT_SWELLING,
    Geometry,
    extrapolate,
    fit_model,
    fit_swelling,
    fraction_released,
    profile_ci,
    swelling_for_tablet,
)
from swelling_model.model import integral_dynamic, swell_curve

GEOM = Geometry.from_tablet(12.7, 2.8)
SW = swelling_for_tablet("K15M", GEOM.R0, GEOM.Z0)
T = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8], float)


def test_release_starts_near_zero_and_is_monotone() -> None:
    t = np.linspace(0, 48, 400)
    for m in ("D1", "D2", "S1", "S2"):
        y = fraction_released(m, t, 3e-6, 2e-9, GEOM, SW)
        assert y[0] < 0.005  # truncation of 200 terms: 0.608/200
        assert np.all(np.diff(y) >= -1e-12)
        assert y[-1] <= 1.0


def test_dynamic_integral_matches_quadrature() -> None:
    D0, g, X0, Xf, k = 3e-6, 2e-9, 1.4, 6.5, 0.31 / 3600
    for t in (60.0, 3600.0, 8 * 3600.0, 30 * 3600.0):
        ref = quad(lambda s: (D0 + g * s) / swell_curve(s, X0, Xf, k) ** 2, 0, t)[0]
        assert integral_dynamic(np.array([t]), D0, g, X0, Xf, k)[0] == pytest.approx(ref, rel=1e-9)


def test_no_swelling_reduces_dynamic_to_static() -> None:
    flat = swelling_for_tablet("K15M", GEOM.R0, GEOM.Z0, kr_h=0.0, kz_h=0.0)
    a = fraction_released("D1", T, 3e-6, 2e-9, GEOM, flat)
    b = fraction_released("S1", T, 3e-6, 2e-9, GEOM)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_gamma_is_ignored_by_constant_d_models() -> None:
    a = fraction_released("S2", T, 5e-6, 9e-9, GEOM)
    b = fraction_released("S2", T, 5e-6, 0.0, GEOM)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("model", ["S1", "D1"])
def test_fit_recovers_known_parameters(model: str) -> None:
    D0, g = 3.0e-6, 2.5e-9
    y = fraction_released(model, T, D0, g, GEOM, SW)
    r = fit_model(model, T, y, GEOM, SW)
    assert pytest.approx(D0, rel=1e-4) == r.D0
    assert r.gamma == pytest.approx(g, rel=1e-4)
    assert r.rmse < 1e-6


def test_profile_ci_brackets_estimate() -> None:
    rng = np.random.default_rng(0)
    y = fraction_released("S1", T, 4e-6, 0.4e-9, GEOM) + rng.normal(0, 0.01, len(T))
    r = fit_model("S1", T, y, GEOM)
    lo, hi = profile_ci(r, T, y, "gamma")
    assert lo <= r.gamma <= hi


def test_extrapolation_from_short_window() -> None:
    rng = np.random.default_rng(3)
    t_all = np.array([0.5, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24], float)
    true = fraction_released("D1", t_all, 3e-6, 2.5e-9, GEOM, SW)
    rows = [dict(tablet=k, time_h=t, frac=v + rng.normal(0, 0.01))
            for k in range(6) for t, v in zip(t_all, true, strict=True)]
    pt = pd.DataFrame(rows)
    mean = pt.groupby("time_h")["frac"].mean()
    ex = extrapolate("D1", mean.index.to_numpy(), mean.to_numpy(), GEOM, SW, cutoff_h=8.0,
                     t_pred_h=t_all, per_tablet=pt, n_boot=40)
    assert ex.test is not None and ex.test["RMSE"] < 0.03
    assert np.all(ex.lo <= ex.pred + 1e-9) and np.all(ex.pred <= ex.hi + 1e-9)


def test_defaults_cover_the_three_grades() -> None:
    assert {"K15M", "K100M", "K100LV"} <= set(DEFAULT_SWELLING)
    sw = swelling_for_tablet("K100M", 6.4, 1.5)
    assert sw.Rf > sw.R0 and sw.Zf > sw.Z0
    scaled = swelling_for_tablet("K100M", 6.4, 1.5, rate_scaling="axial_L2")
    assert scaled.kz_h < sw.kz_h  # thicker than the 200 mg source -> slower axial swelling


def test_fit_swelling_recovers_rates() -> None:
    t = np.array([0, 0.5, 1, 2, 4, 6, 8, 12], float)
    df = pd.DataFrame(dict(time_h=t, diameter_mm=2 * swell_curve(t, 6.35, 8.6, 0.25),
                           height_mm=2 * swell_curve(t, 1.4, 6.5, 0.31)))
    d = fit_swelling(df, "K15M")
    assert d.kr_h == pytest.approx(0.25, rel=1e-4)
    assert d.Zf_Z0 == pytest.approx(6.5 / 1.4, rel=1e-4)
