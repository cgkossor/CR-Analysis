"""The manuscript questions: answered, honestly labelled, and reproducible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline import manuscript
from pipeline.analysis import Analysis, run_analysis
from pipeline.io.load import load_database

ROOT = Path(__file__).resolve().parents[1]


def _workbook() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture(scope="module")
def analysis() -> Analysis:
    return run_analysis(load_database(_workbook()))


@pytest.fixture(scope="module")
def with_dt(tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    from pipeline.disintegration.synthetic import generate
    from pipeline.run import _disintegration

    source = generate(_workbook(), tmp_path_factory.mktemp("dt") / "with_dt.xlsx")
    a = run_analysis(load_database(source))
    dt = _disintegration(a, str(source))
    assert dt is not None
    return a, dt


def test_every_question_answers_with_a_status(analysis: Analysis) -> None:
    ms = manuscript.build(analysis)
    assert [q.id for q in ms.questions] == ["q1", "q2", "q3", "q4", "q5"]
    for q in ms.questions:
        assert q.question.endswith("?") and q.answer


def test_lever_shares_partition_the_spread(analysis: Analysis) -> None:
    ms = manuscript.build(analysis)
    assert ms.decompositions
    for d in ms.decompositions:
        total = d.share_composition + d.share_grade + d.share_interaction
        assert total == pytest.approx(1.0)
        assert min(d.share_composition, d.share_grade, d.share_interaction) >= 0


def test_solubility_is_gated_with_one_api(analysis: Analysis) -> None:
    """G1: no solubility claim from a single API."""
    q5 = manuscript.build(analysis).questions[4]
    assert q5.status == "gated"
    assert not q5.claims
    assert "Insufficient data" in q5.answer


def test_disintegration_questions_need_the_data(analysis: Analysis) -> None:
    ms = manuscript.build(analysis, None)
    assert ms.questions[2].status == "unavailable"
    assert ms.questions[3].status == "unavailable"
    assert ms.shear == {}


def test_shear_proxy_never_claims_more_than_directional(with_dt) -> None:  # type: ignore[no-untyped-def]
    """A proxy from two different apparatus is never 'supported'."""
    a, dt = with_dt
    ms = manuscript.build(a, dt)
    q3 = ms.questions[2]
    assert q3.status == "directional"
    assert all(c.status != "supported" for c in q3.claims)
    assert any("proxy" in c for c in q3.caveats)
    assert ms.shear and {v["tier"] for v in ms.shear.values()} <= {"low", "mid", "high"}
    assert ms.questions[3].status != "unavailable"


def test_payload_is_json_safe_and_deterministic(with_dt) -> None:  # type: ignore[no-untyped-def]
    a, dt = with_dt
    first = json.dumps(manuscript.payload(manuscript.build(a, dt)), allow_nan=False,
                       sort_keys=True)
    second = json.dumps(manuscript.payload(manuscript.build(a, dt)), allow_nan=False,
                        sort_keys=True)
    assert first == second


def test_storyline_leads_with_a_headline_claim(analysis: Analysis) -> None:
    ms = manuscript.build(analysis)
    ranked = ms.storyline.claims
    assert ranked
    supported = [r for r in ranked if r.claim.status == "supported"]
    if supported:
        assert supported[0].headline
        assert ms.storyline.lead.startswith(f"Lead with {supported[0].question.upper()}")


def test_render_writes_the_figures_each_question_names(with_dt, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    a, dt = with_dt
    ms = manuscript.build(a, dt)
    out = tmp_path / "manuscript"
    out.mkdir()
    (out / "OLD_stale.png").write_bytes(b"")
    records = manuscript.render(ms, a, dt, out)
    captions = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    ids = {c["id"] for c in captions}
    named = {i for q in ms.questions for i in q.figure_ids}
    assert named == ids == {r.id for r in records}
    assert not (out / "OLD_stale.png").exists(), "a stale figure survived the render"
    for c in captions:
        assert (out / c["png"]).exists()


def test_report_lists_every_question(analysis: Analysis, tmp_path: Path) -> None:
    ms = manuscript.build(analysis)
    path = tmp_path / "storyline.md"
    manuscript.write_report(ms, path)
    text = path.read_text(encoding="utf-8")
    for q in ms.questions:
        assert f"## {q.id.upper()}." in text
