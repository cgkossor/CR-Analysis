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


def test_mw_eq_blend_and_binary() -> None:
    from swelling_model.trends import composition_from_columns, mw_eq

    assert mw_eq({"K15M": 60}) == pytest.approx(0.6 * 134)
    assert mw_eq({"K100LV": 15, "K100M": 15}) == pytest.approx(0.15 * 29 + 0.15 * 267)
    row = pd.Series(dict(drug_load_pct=40.0))
    assert composition_from_columns(row, "K15M") == {"K15M": 60.0}
    blend = pd.Series(dict(wt_pct_K15M=20.0, wt_pct_K100LV=10.0, drug_load_pct=40.0))
    assert composition_from_columns(blend, "K15M") == {"K15M": 20.0, "K100LV": 10.0}


def test_parameter_trends_recovers_monotone_gamma() -> None:
    from swelling_model.trends import parameter_trends

    mw = np.array([20.0, 40, 80, 120, 200])
    fits = pd.DataFrame(dict(api="A", model="S1", formulation=[f"F{i}" for i in range(5)],
                             mw_eq=mw, D0_e6=5.0, gamma_e9=10 / mw))
    tr = parameter_trends(fits, models=("S1",)).set_index("param")
    assert tr.loc["gamma", "spearman_rho"] == pytest.approx(-1.0)
    assert tr.loc["gamma", "loglog_slope"] == pytest.approx(-1.0)
    assert tr.loc["D0", "median"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------------
# Early-delay variants (swelling_model/delay.py)
# ---------------------------------------------------------------------------------

def test_delay_limits_recover_base_model() -> None:
    from swelling_model import fraction_released_k, fraction_released_lag

    base = fraction_released("D1", T, 3e-6, 2.5e-9, GEOM, SW)
    assert np.allclose(fraction_released_lag("D1", T, 3e-6, 2.5e-9, 0.0, GEOM, SW), base)
    fast = fraction_released_k("D1", T, 3e-6, 2.5e-9, 1e4, GEOM, SW)
    assert np.max(np.abs(fast - base)) < 1e-4
    lag = fraction_released_lag("D1", T, 3e-6, 2.5e-9, 0.5, GEOM, SW)
    after = T > 0.5
    assert np.all(lag[~after] == 0.0)
    assert np.allclose(lag[after], fraction_released("D1", T[after] - 0.5, 3e-6, 2.5e-9, GEOM, SW))


def test_release_step_matches_slow_dissolution_limit() -> None:
    """With very fast diffusion, M -> 1 - exp(-kt) (release is dissolution-limited)."""
    from swelling_model import fraction_released_k

    m = fraction_released_k("S1", T, 1e-1, 0.0, 0.3, GEOM)
    assert np.max(np.abs(m - (1 - np.exp(-0.3 * T)))) < 2e-3


def test_release_step_is_monotone_and_slower_than_base() -> None:
    from swelling_model import fraction_released_k

    tt = np.linspace(0, 12, 25)
    m = fraction_released_k("D1", tt, 3e-6, 2.5e-9, 2.0, GEOM, SW)
    assert np.all(np.diff(m) >= -1e-9)
    assert np.all(m <= fraction_released("D1", tt, 3e-6, 2.5e-9, GEOM, SW) + 1e-9)


@pytest.mark.parametrize("kind,x", [("k", 3.0), ("lag", 0.3)])
def test_fit_delay_recovers_known_parameters(kind: str, x: float) -> None:
    from swelling_model import fit_delay
    from swelling_model.delay import predict_delay

    noise = 0.004 * np.sin(np.arange(len(T)) * 2.3)       # deterministic, ~0.3 pp
    y = predict_delay("D1", kind, T, 4e-6, 2e-9, x, GEOM, SW) + noise
    r = fit_delay("D1", kind, T, y, GEOM, SW, profile=("x",))
    assert r.p == pytest.approx([4.0, 2.0, x], rel=0.3)
    lo, hi = r.prof_ci["x"]
    assert lo <= x <= hi


def test_delay_probe_reads_lactose_trend() -> None:
    from swelling_model import delay_probe

    rows = []
    for api in ("A", "B"):
        for grade in ("K15M", "K100M"):
            for lac in (0, 10, 20, 30, 40):
                rows.append(dict(model="D1+k", api=api, grade=grade, lactose_wt_pct=lac,
                                 delay_min=30 * 10 ** (-lac / 40), D0_e6=5.0 if api == "A" else 0.2,
                                 gamma_e9=1 + lac / 10))
    out = delay_probe(pd.DataFrame(rows))
    sp = out["spearman"]
    assert np.allclose(sp[sp.param == "delay_min"].rho_vs_lactose, -1.0)
    ols = out["ols"].set_index(["param", "term"])
    assert ols.loc[("delay_min", "lactose_wt_pct"), "coef"] == pytest.approx(-1 / 40)
    assert ols.loc[("delay_min", "C(api)[T.B]"), "p"] > 0.05
