"""The dashboard must reproduce Python's predictions exactly.

The formulator tool re-implements the fitted surface in JavaScript so the page
works offline with no Python runtime. That duplication is the risk: if the two
implementations drift, every number in the tool becomes quietly wrong while the
page continues to render perfectly. This test runs the browser's own module
under Node against the exported ``data.js`` and compares it to the Python
prediction path.

Skipped, not silently passed, when Node is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from pipeline.analysis import Analysis, predict_profile, run_analysis
from pipeline.export.data_js import PRECISION, build_payload, write_data_js
from pipeline.io.load import load_database

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

#: Compositions probed on both sides. Chosen to sit inside the tested hull and
#: to span grades, including the extremes where the interaction bites hardest.
PROBES = [
    (10.0, 20.0, 70.0, 2.0),
    (35.0, 40.0, 25.0, 3.60206),
    (60.0, 20.0, 20.0, 5.0),
    (22.5, 49.5, 28.0, 3.60206),
    (35.0, 60.0, 5.0, 5.0),
    (47.5, 37.0, 15.5, 2.0),
]

TIMES = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0]


def _database() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Analysis]:
    db = load_database(_database())
    analysis = run_analysis(db)
    out = tmp_path_factory.mktemp("export") / "data.js"
    write_data_js(build_payload(analysis), out)
    return out, analysis


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_javascript_matches_python(exported: tuple[Path, Analysis]) -> None:
    data_js, analysis = exported

    script = f"""
    const fs = require('fs');
    global.window = global;
    eval(fs.readFileSync({json.dumps(str(data_js))}, 'utf8'));
    const M = require({json.dumps(str(ROOT / "dashboard" / "model.js"))});
    const levels = window.CR_DATA.design_points.map(p => p.log10_visc);
    const probes = {json.dumps(PROBES)};
    const times = {json.dumps(TIMES)};
    const out = probes.map(p =>
      M.predictProfile(window.CR_DATA.surfaces, levels, p[0], p[1], p[2], p[3], times)
    );
    process.stdout.write(JSON.stringify(out));
    """
    result = subprocess.run(
        [str(NODE), "-e", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    js = json.loads(result.stdout)

    # data.js rounds every coefficient to PRECISION decimals so that two runs
    # produce byte-identical output (G10). The browser therefore evaluates
    # slightly rounded coefficients, and exact bit-equality is not the claim.
    # The claim is that the difference is bounded by that deliberate rounding and
    # is negligible beside the cross-validated error the prediction is reported
    # with -- roughly four orders of magnitude smaller.
    param_tol = 10.0 ** -(PRECISION - 1)
    profile_tol_pct = 0.01
    cv_error = float(analysis.cross_validation.profile_rmse_pct)
    assert profile_tol_pct < cv_error / 50.0, (
        "the parity tolerance must stay far below the reported prediction error, "
        "otherwise this test could pass while the tool was materially wrong"
    )

    for probe, js_result in zip(PROBES, js, strict=True):
        api, hpmc, lactose, log10_visc = probe
        py = predict_profile(analysis, api, hpmc, lactose, log10_visc, np.array(TIMES))

        for key in ("log10_td", "weibull_beta", "weibull_f_inf"):
            assert js_result["params"][key] == pytest.approx(
                py["parameters"][key], abs=param_tol
            ), f"{key} diverged at {probe}"
        assert np.allclose(js_result["curve"], py["profile"], atol=profile_tol_pct), (
            f"profile diverged at {probe} by more than {profile_tol_pct}% released"
        )


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_predicted_profiles_are_monotonic_and_bounded(exported: tuple[Path, Analysis]) -> None:
    """AC4 makes this a bug, not a warning -- so it is asserted, not reported."""
    data_js, _ = exported
    script = f"""
    const fs = require('fs');
    global.window = global;
    eval(fs.readFileSync({json.dumps(str(data_js))}, 'utf8'));
    const M = require({json.dumps(str(ROOT / "dashboard" / "model.js"))});
    const levels = window.CR_DATA.design_points.map(p => p.log10_visc);
    const times = []; for (let t = 0; t <= 24; t += 0.25) times.push(t);
    const out = [];
    for (let a = 10; a <= 60; a += 5)
      for (let h = 20; h <= 60; h += 5) {{
        const l = 100 - a - h;
        if (l < 5 || l > 70) continue;
        for (const lv of [2.0, 3.60206, 5.0])
          out.push(M.predictProfile(window.CR_DATA.surfaces, levels, a, h, l, lv, times).curve);
      }}
    process.stdout.write(JSON.stringify(out));
    """
    result = subprocess.run(
        [str(NODE), "-e", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, f"node failed: {result.stderr}"
    curves = json.loads(result.stdout)
    assert curves, "no curves generated"

    for curve in curves:
        values = np.asarray(curve, dtype=float)
        assert np.all(np.diff(values) >= -1e-9), "predicted profile is not monotonic"
        assert values.min() >= 0.0 and values.max() <= 100.0, "predicted profile out of bounds"


def test_dashboard_makes_no_network_calls() -> None:
    """G7: the page must work from file:// with no server and no internet."""
    banned = ("fetch(", "XMLHttpRequest", "WebSocket", "cdn.", "https://", "http://")
    for name in ("index.html", "app.js", "model.js", "styles.css"):
        path = ROOT / "dashboard" / name
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            # Comments may legitimately discuss the prohibition itself.
            if stripped.startswith(("*", "//", "/*", "<!--")):
                continue
            if "www.w3.org/2000/svg" in line:
                continue
            for token in banned:
                assert token not in line, f"{name}:{lineno} contains {token!r}: {stripped}"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_t50_definition_matches_between_python_and_javascript() -> None:
    """t50 means time to 50% of the DOSE in both implementations.

    The Weibull median ``Td*(ln 2)^(1/beta)`` is the time to half of ``F_inf``,
    which is a different quantity whenever the formulation does not release its
    full dose -- precisely the slow formulations the inverse design is asked
    about. The two definitions differ by tens of percent there, so this is
    pinned rather than assumed.
    """
    from pipeline.profiles.censoring import time_to_percent
    from pipeline.profiles.fits import weibull as py_weibull

    cases = [(6.0, 0.68, 92.0), (10.0, 0.55, 78.0), (2.0, 0.9, 100.0), (20.0, 0.5, 45.0)]
    script = f"""
    const M = require({json.dumps(str(ROOT / "dashboard" / "model.js"))});
    const cases = {json.dumps(cases)};
    process.stdout.write(JSON.stringify(
      cases.map(c => M.t50From(c[0], c[1], c[2]))
    ));
    """
    result = subprocess.run(
        [str(NODE), "-e", script], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    js = json.loads(result.stdout)

    grid = np.linspace(0.0, 240.0, 24001)
    for (td, beta, f_inf), js_t50 in zip(cases, js, strict=True):
        curve = py_weibull(grid, f_inf, td, beta)
        py = time_to_percent(grid, curve, 50.0)
        if f_inf <= 50.0:
            assert py.censored, "a formulation below 50% release has no t50"
            assert js_t50 is None, "JavaScript must return null, not a finite time"
        else:
            assert js_t50 is not None
            assert js_t50 == pytest.approx(py.value, rel=2e-3), (
                f"t50 diverged for Td={td}, beta={beta}, F_inf={f_inf}"
            )
