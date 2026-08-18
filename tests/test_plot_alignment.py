"""Every plotted series must carry its own x values.

The bug this guards against was invisible on the placeholder database and wrong
on real data. Replicates were exported as values only, so the dashboard indexed
them positionally against the canonical grid. While both happened to be 25
points the plot looked correct; once ragged sampling times were reconciled the
grid grew longer than a replicate's sample count and the curve was drawn across
only the first N grid slots, compressing a 24-hour profile into a few hours.

Positional alignment between two independently-derived vectors is the defect.
These tests assert x and y always travel together, and that a mismatch is
refused rather than truncated.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis import run_analysis
from pipeline.export.data_js import build_payload
from pipeline.io.load import load_database

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _source() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture(scope="module")
def ragged_payload() -> dict:
    """A workbook whose profiles are sampled on their own jittered schedules."""
    src = _source()
    frame = pd.read_excel(src, sheet_name="Dissolution")
    design = pd.read_excel(src, sheet_name="Design", header=None)

    rng = np.random.default_rng(11)
    parts = []
    for _, group in frame.groupby("ID"):
        g = group.copy()
        offset = float(rng.uniform(-0.9, 0.9))
        g["Min_1"] = g["Min_1"].where(g["Min_1"] == 0, g["Min_1"] + offset)
        parts.append(g)
    jittered = pd.concat(parts, ignore_index=True)

    tmp = Path(tempfile.mkdtemp()) / "ragged.xlsx"
    with pd.ExcelWriter(tmp) as writer:
        jittered.to_excel(writer, sheet_name="Dissolution", index=False)
        design.to_excel(writer, sheet_name="Design", index=False, header=False)

    return build_payload(run_analysis(load_database(tmp)))


class TestSeriesCarryTheirOwnX:
    def test_every_replicate_has_matching_x_and_y(self, ragged_payload: dict) -> None:
        for profile in ragged_payload["profiles"]:
            for rep in profile["replicates"]:
                assert "times_h" in rep, "replicate exported without its own time vector"
                assert len(rep["times_h"]) == len(rep["pct"]), (
                    f"case {profile['case']}/{profile['grade']} rep {rep['replicate']}: "
                    f"{len(rep['times_h'])} times vs {len(rep['pct'])} values"
                )

    def test_mean_profile_matches_the_canonical_grid(self, ragged_payload: dict) -> None:
        grid = len(ragged_payload["grid_h"])
        for profile in ragged_payload["profiles"]:
            assert len(profile["mean_pct"]) == grid
            assert len(profile["sd_pct"]) == grid

    def test_replicates_span_the_full_observed_window(self, ragged_payload: dict) -> None:
        """The visible symptom: curves compressed into the first few hours."""
        last_grid_point = ragged_payload["grid_h"][-1]
        for profile in ragged_payload["profiles"]:
            for rep in profile["replicates"]:
                span = max(rep["times_h"])
                assert span > 0.8 * last_grid_point, (
                    f"case {profile['case']}/{profile['grade']} rep {rep['replicate']} "
                    f"spans only {span:.2f} h of a {last_grid_point:.2f} h window"
                )

    def test_the_grid_and_replicates_genuinely_differ_here(
        self, ragged_payload: dict
    ) -> None:
        """Without this the suite could pass on a fixture that never reproduces it."""
        grid = len(ragged_payload["grid_h"])
        lengths = {
            len(rep["pct"])
            for profile in ragged_payload["profiles"]
            for rep in profile["replicates"]
        }
        assert lengths, "no replicates in the fixture"
        # Either the counts differ, or the times do -- both break positional indexing.
        times_differ = any(
            not np.allclose(rep["times_h"], ragged_payload["grid_h"])
            for profile in ragged_payload["profiles"]
            for rep in profile["replicates"]
            if len(rep["times_h"]) == grid
        )
        assert lengths != {grid} or times_differ, (
            "fixture is not actually ragged; this test would pass vacuously"
        )


class TestAxisLimitsComeFromConfig:
    def test_exported_for_the_dashboard(self, ragged_payload: dict) -> None:
        meta = ragged_payload["meta"]
        for key in (
            "plot_min_time_h",
            "plot_max_time_h",
            "plot_min_release_pct",
            "plot_max_release_pct",
        ):
            assert key in meta, f"{key} not exported; dashboard would hard-code it"

    def test_release_axis_leaves_room_for_overshoot(self, ragged_payload: dict) -> None:
        """Real assays reach 105-110%; an axis capped at 100 would hide it."""
        assert ragged_payload["meta"]["plot_max_release_pct"] > 100.0


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_chart_refuses_mismatched_series() -> None:
    """A length mismatch must be reported, not silently truncated."""
    script = r"""
    const fs = require('fs');
    let errors = [];
    global.console = Object.assign({}, console, { error: (m) => errors.push(String(m)) });
    global.window = global;
    global.document = {
      createElementNS: () => ({ setAttribute(){}, appendChild(){}, style:{} }),
      createElement: () => ({ setAttribute(){}, appendChild(){}, style:{}, dataset:{} }),
      getElementById: () => null,
      querySelectorAll: () => [],
      addEventListener: () => {},
    };
    const src = fs.readFileSync(process.argv[1], 'utf8');
    const m = src.match(/function Chart\([\s\S]*?Chart\.prototype\.line = function[\s\S]*?\n  \};/);
    if (!m) { console.log(JSON.stringify({err:'could not extract Chart'})); process.exit(0); }
    eval('var svgEl = function(){return {setAttribute(){},appendChild(){}}};' + m[0]);
    const c = new Chart(100, 100);
    c.scales([0, 10], [0, 100]);
    c.line([1, 2, 3], [10, 20], '#000');
    process.stdout.write(JSON.stringify({ errors }));
    """
    result = subprocess.run(
        [str(NODE), "-e", script, str(ROOT / "dashboard" / "app.js")],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload.get("errors"), "mismatched series drew silently instead of erroring"
    assert "mis-aligned" in payload["errors"][0]
