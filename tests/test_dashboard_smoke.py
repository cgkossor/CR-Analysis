"""The dashboard must actually render, not merely parse.

A runtime error inside ``boot()`` leaves a blank page while every static check
still passes, so the page is loaded in a real DOM and every tab is clicked.

Requires ``jsdom``. It is not vendored into the repo -- the dashboard itself
ships no dependencies (G7) and a test-only Node package has no business living
next to it. Install with ``npm install jsdom`` anywhere on NODE_PATH, or run
``npm install jsdom`` in the repo root. The test skips, loudly, when it is
absent; it never passes by default.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "dashboard"
NODE = shutil.which("node")

_SMOKE = r"""
const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs"), path = require("path");
const dash = process.argv[1];
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));
const dom = new JSDOM(fs.readFileSync(path.join(dash, "index.html"), "utf8"),
  { runScripts: "dangerously", virtualConsole: vc });
const w = dom.window;
for (const f of ["data.js", "model.js", "doe.js", "formulator.js",
                  "disintegration.js", "audit.js", "app.js"]) {
  try { w.eval(fs.readFileSync(path.join(dash, f), "utf8")); }
  catch (e) { errors.push(f + " threw: " + e.message); }
}
try { w.document.dispatchEvent(new w.Event("DOMContentLoaded")); }
catch (e) { errors.push("boot threw: " + e.message); }
const d = w.document;
const tabs = {};
for (const b of Array.from(d.querySelectorAll("#tabs button"))) {
  try {
    b.dispatchEvent(new w.Event("click", { bubbles: true }));
    const name = b.getAttribute("data-target");
    const panel = d.querySelector('.panel[data-tab="' + name + '"]');
    tabs[name] = { chars: panel.textContent.trim().length,
                   svg: panel.querySelectorAll("svg").length };
  } catch (e) { errors.push("tab threw: " + e.message); }
}
process.stdout.write(JSON.stringify({
  tabs,
  nTabs: d.querySelectorAll("#tabs button").length,
  provenanceShown: !d.getElementById("provenance").hidden,
  cvBadge: (d.getElementById("cv-badge").textContent || "").replace(/\s+/g, " ").trim(),
  g1Gate: d.body.innerHTML.indexOf("Insufficient data") >= 0,
  rows: d.querySelectorAll("tbody tr").length,
  tooltips: d.querySelectorAll("abbr.tip").length,
  adminText: (d.getElementById("ad-text") || {}).value || "",
  errors,
}));
"""

EXPECTED_TABS = {
    "explorer",
    "metrics",
    "design",
    "doe",
    "surfaces",
    "equivalence",
    "stress",
    "diagnostics",
    "formulator",
    "disintegration",
    "guidelines",
    "admin",
}


def _have_jsdom() -> bool:
    if NODE is None:
        return False
    probe = subprocess.run(
        [NODE, "-e", "require.resolve('jsdom')"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    return probe.returncode == 0


@pytest.fixture(scope="module")
def rendered() -> dict:
    if NODE is None:
        pytest.skip("node is not installed")
    if not (DASHBOARD / "data.js").exists():
        pytest.skip("dashboard/data.js not generated; run python -m pipeline.run first")
    if not _have_jsdom():
        pytest.skip("jsdom not installed (npm install jsdom) - dashboard render not verified")

    result = subprocess.run(
        [NODE, "-e", _SMOKE, str(DASHBOARD)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        cwd=ROOT,
    )
    assert result.returncode == 0, f"smoke harness failed: {result.stderr[:2000]}"
    return json.loads(result.stdout)


def test_page_boots_without_errors(rendered: dict) -> None:
    assert rendered["errors"] == [], f"dashboard raised: {rendered['errors']}"


def _expected_tabs() -> set[str]:
    """Disintegration is optional: its tab appears only when data.js carries it."""
    data = (DASHBOARD / "data.js").read_text(encoding="utf-8")
    has_dt = '"disintegration": {' in data
    return EXPECTED_TABS if has_dt else EXPECTED_TABS - {"disintegration"}


def test_every_tab_renders_content(rendered: dict) -> None:
    expected = _expected_tabs()
    assert rendered["nTabs"] == len(expected)
    assert set(rendered["tabs"]) == expected
    for name, info in rendered["tabs"].items():
        # "Handles the single-API case without empty panels" is an explicit
        # acceptance criterion, so an empty panel is a failure, not a nuance.
        assert info["chars"] > 400, f"tab {name} rendered almost nothing ({info['chars']} chars)"


def test_charts_render_where_expected(rendered: dict) -> None:
    for name in ("explorer", "design", "doe", "surfaces", "equivalence",
                 "stress", "formulator"):
        assert rendered["tabs"][name]["svg"] >= 1, f"tab {name} rendered no chart"


def test_provenance_banner_is_shown_for_synthetic_data(rendered: dict) -> None:
    data = (DASHBOARD / "data.js").read_text(encoding="utf-8")
    if '"is_synthetic": true' in data:
        assert rendered["provenanceShown"], (
            "database declares itself synthetic but the page shows no provenance banner"
        )


def test_cross_validated_error_is_always_visible(rendered: dict) -> None:
    """G6: no prediction is displayed without its error estimate."""
    assert "%" in rendered["cvBadge"], f"CV badge missing: {rendered['cvBadge']!r}"
    assert "LOFO" in rendered["cvBadge"]


def test_diagnostics_tab_surfaces_the_verdict(rendered: dict) -> None:
    """The run's health has to be visible in the tool, not only in a file."""
    assert rendered["tabs"]["diagnostics"]["chars"] > 1000, (
        "Diagnostics tab rendered almost nothing"
    )


def test_admin_tab_produces_a_shareable_report(rendered: dict) -> None:
    """The copyable audit must be fenced, integer-only, and report every tab rendered."""
    from pipeline.audit import REPORT_LINE as LINE

    lines = rendered["adminText"].splitlines()
    assert lines and lines[0].startswith("=== CR-AUDIT"), "admin report missing"
    assert lines[-1].startswith("=== END CR-AUDIT")
    bad = [ln for ln in lines if not LINE.match(ln)]
    assert not bad, f"admin report lines outside the audit grammar: {bad[:5]}"
    assert "W14 tabs_failed=0" in lines or any(
        ln.endswith(" tabs_failed=0") for ln in lines
    ), "a dashboard tab failed to render"


def test_statistics_carry_definitions(rendered: dict) -> None:
    """A number a reader cannot interpret is not evidence (issues 8 and 10)."""
    assert rendered["tooltips"] > 10, (
        f"only {rendered['tooltips']} tooltips on the page; statistics should "
        "carry their definitions"
    )


def test_solubility_claims_are_gated_with_one_api(rendered: dict) -> None:
    """G1: with a single API loaded, the cross-API section must render as gated."""
    data = json.loads(
        (DASHBOARD / "data.js").read_text(encoding="utf-8").split("=", 1)[1].rsplit(";", 1)[0]
    )
    if len(data["quality"]["apis"]) < 2:
        assert rendered["g1Gate"], (
            "only one API is loaded but no data-sufficiency gate was rendered"
        )


_INTERACTION = r"""
const { JSDOM, VirtualConsole } = require("jsdom");
const fs = require("fs"), path = require("path");
const dash = process.argv[1];
const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", e => errors.push("jsdomError: " + e.message));
vc.on("error", (...a) => errors.push("console.error: " + a.join(" ")));
const dom = new JSDOM(fs.readFileSync(path.join(dash, "index.html"), "utf8"),
  { runScripts: "dangerously", virtualConsole: vc });
const w = dom.window;
for (const f of ["data.js", "model.js", "doe.js", "formulator.js",
                  "disintegration.js", "audit.js", "app.js"]) {
  w.eval(fs.readFileSync(path.join(dash, f), "utf8"));
}
w.document.dispatchEvent(new w.Event("DOMContentLoaded"));
const d = w.document;
const chart = d.querySelector("#ex-chart svg");
const overlay = chart.querySelector('rect[pointer-events="all"]');
overlay.dispatchEvent(new w.MouseEvent("mousemove", { clientX: 200, clientY: 120, bubbles: true }));
const labels = Array.from(chart.querySelectorAll("text"))
  .filter(t => t.getAttribute("opacity") === "1");
const marker = chart.querySelector('circle[opacity="1"]');
overlay.dispatchEvent(new w.MouseEvent("click", { clientX: 200, clientY: 120, bubbles: true }));
const dimmed = Array.from(chart.querySelectorAll("path"))
  .filter(p => p.getAttribute("opacity") === "0.12").length;
process.stdout.write(JSON.stringify({
  overlay: !!overlay,
  bands: chart.querySelectorAll('path[stroke="none"]').length,
  label: labels.length ? labels[labels.length - 1].textContent : null,
  marker: !!marker,
  dimmedAfterClick: dimmed,
  errors,
}));
"""


@pytest.fixture(scope="module")
def interaction() -> dict:
    if NODE is None:
        pytest.skip("node is not installed")
    if not (DASHBOARD / "data.js").exists():
        pytest.skip("dashboard/data.js not generated")
    if not _have_jsdom():
        pytest.skip("jsdom not installed (npm install jsdom)")
    result = subprocess.run(
        [NODE, "-e", _INTERACTION, str(DASHBOARD)],
        capture_output=True, text=True, timeout=300, check=False, cwd=ROOT,
    )
    assert result.returncode == 0, f"interaction harness failed: {result.stderr[:1500]}"
    return json.loads(result.stdout)


def test_hover_identifies_a_curve(interaction: dict) -> None:
    """Thirty overlaid profiles are a texture until you can name one."""
    assert interaction["errors"] == []
    assert interaction["overlay"], "no hover target attached to the chart"
    label = interaction["label"]
    assert label, "hovering produced no identifying label"
    assert "case" in label and "%" in label and "h" in label, (
        f"label should name the curve and read off its value; got {label!r}"
    )
    assert interaction["marker"], "no point marker shown at the hovered position"


def test_clicking_pins_a_curve(interaction: dict) -> None:
    assert interaction["dimmedAfterClick"] > 0, (
        "clicking did not isolate a curve; the others should dim"
    )


def test_replicate_spread_is_drawn_as_a_band(interaction: dict) -> None:
    assert interaction["bands"] > 0, "no +/-1 SD band rendered"


def test_doe_tab_leads_with_plots_and_takeaways(rendered: dict) -> None:
    """Issues 5, 6 and 9: the classical DoE analysis has to be visible and read.

    A tab full of coefficient tables is what prompted "lackluster"; the point of
    this one is that a finding is stated before any table appears.
    """
    doe = rendered["tabs"]["doe"]
    assert doe["chars"] > 2000, f"DoE tab rendered {doe['chars']} chars"
    assert doe["svg"] >= 4, (
        f"only {doe['svg']} charts on the DoE tab; contour, interaction, traces, "
        "Pareto and half-normal should all render"
    )


def test_formulator_offers_goals_not_raw_weights(rendered: dict) -> None:
    """Issue 13: balancing three components and tuning weights by hand is the
    problem, not the interface. The tool fixes the drug load and picks the rest."""
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert 'id="iv-goal"' in html, "no goal selector; weights are still the interface"
    assert 'id="iv-api"' in html, "drug load is not the fixed input"
    for retired in ('id="iv-w-t50"', 'id="iv-w-p24"'):
        assert retired not in html, f"{retired} still present; raw weight sliders remain"


def test_equivalence_is_framed_as_substitution(rendered: dict) -> None:
    """Issue 12: 'pick a group, get a plot' is not the question anyone asks."""
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    assert 'id="eq-guidance"' in html, "no substitution guidance"
    assert 'id="eq-map"' in html, "no design-space map of where freedom exists"


def test_disintegration_tab_renders_when_the_sheet_exists(tmp_path: Path) -> None:
    """With a Disintegration sheet the tab appears, draws, and feeds the DoE tab."""
    if NODE is None or not _have_jsdom():
        pytest.skip("Node.js with jsdom is needed to render the dashboard")
    workbooks = sorted(ROOT.glob("*.xlsx"))
    if not workbooks:
        pytest.skip("no database workbook present")

    from pipeline.analysis import run_analysis
    from pipeline.diagnostics import collect
    from pipeline.disintegration.synthetic import generate
    from pipeline.export.data_js import write_data_js
    from pipeline.io.load import load_database
    from pipeline.run import _disintegration, _payload, _stress

    source = generate(workbooks[0], tmp_path / "with_dt.xlsx")
    analysis = run_analysis(load_database(source))
    stress = _stress(analysis)
    dt = _disintegration(analysis, str(source))
    assert dt is not None
    payload, _ = _payload(analysis, stress, collect(analysis, stress), str(source), dt)

    dash = tmp_path / "dashboard"
    dash.mkdir()
    for f in DASHBOARD.iterdir():
        if f.suffix in (".html", ".js", ".css") and f.name != "data.js":
            shutil.copy(f, dash / f.name)
    write_data_js(payload, dash / "data.js")

    result = subprocess.run(
        [NODE, "-e", _SMOKE, str(dash)],
        capture_output=True, text=True, timeout=300, check=False, cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr[:2000]
    out = json.loads(result.stdout)
    assert out["errors"] == [], out["errors"]
    assert set(out["tabs"]) == EXPECTED_TABS
    tab = out["tabs"]["disintegration"]
    assert tab["chars"] > 400 and tab["svg"] >= 1
    assert any(r["key"] == "dt_h" for r in payload["doe"]["responses"])
    assert "disintegration_in_doe_tab=1" in out["adminText"]
