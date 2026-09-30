"""Manuscript mode: the research questions the paper answers.

Each question module reads results the pipeline already computed and returns
a :class:`~pipeline.manuscript.model.QuestionResult`: the question, a short
answer, the claims behind it with a support status, tables, and the journal
figures that show it. :func:`build` is pure and deterministic (it feeds the
dashboard payload, which the determinism check reproduces); :func:`render`
draws the figures into ``figures/manuscript/`` with their captions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.manuscript import disintegration_link, grade_swap, levers, solubility
from pipeline.manuscript import shear as shear_proxy
from pipeline.manuscript import storyline as story
from pipeline.manuscript.model import STATUS_LABEL, QuestionResult

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.disintegration.analysis import DisintegrationAnalysis
    from pipeline.figures.publication import FigureRecord


@dataclass(frozen=True)
class Manuscript:
    questions: tuple[QuestionResult, ...]
    storyline: story.Storyline
    #: Shear proxy per "case|grade", for the Formulator's shortlist.
    shear: dict[str, dict[str, Any]]
    decompositions: list[levers.Decomposition] = field(repr=False)
    swaps: list[grade_swap.Swap] = field(repr=False)
    shear_points: list[shear_proxy.ShearPoint] = field(repr=False)


def build(analysis: Analysis, dt: DisintegrationAnalysis | None = None) -> Manuscript:
    """Answer every question from the analysis (and disintegration, when present)."""
    q1, decs = levers.answer(analysis)
    q2, sw = grade_swap.answer(analysis)
    q3, pts = shear_proxy.answer(analysis, dt)
    q4 = disintegration_link.answer(dt)
    q5 = solubility.answer(analysis)
    questions = (q1, q2, q3, q4, q5)
    return Manuscript(
        questions=questions,
        storyline=story.build(analysis, questions),
        shear=shear_proxy.tiers(pts),
        decompositions=decs, swaps=sw, shear_points=pts,
    )


def render(
    ms: Manuscript,
    analysis: Analysis,
    dt: DisintegrationAnalysis | None,
    out_dir: Path,
) -> list[FigureRecord]:
    """Draw every question's figure into ``out_dir`` and write its captions."""
    from pipeline.figures import publication as pub
    from pipeline.figures.render import banner_for

    out_dir.mkdir(parents=True, exist_ok=True)
    # Figures from an earlier run must not pass for this run's.
    for stale in list(out_dir.glob("*.png")) + list(out_dir.glob("*.svg")):
        stale.unlink()
    banner = banner_for(analysis)
    records: list[FigureRecord] = []
    records += levers.render(analysis, ms.decompositions, out_dir, banner)
    records += grade_swap.render(analysis, ms.swaps, out_dir, banner)
    records += shear_proxy.render(ms.shear_points, out_dir, banner)
    records += disintegration_link.render(dt, out_dir, banner)
    pub.write_captions(records, out_dir, "Manuscript figures")
    return records


def _claim(c: Any) -> dict[str, Any]:
    return {"text": c.text, "effect": c.effect, "uncertainty": c.uncertainty,
            "status": c.status, "status_label": STATUS_LABEL[c.status],
            "signal_to_noise": c.signal_to_noise}


def payload(ms: Manuscript) -> dict[str, Any]:
    """The dashboard's view of the manuscript questions (JSON-safe)."""
    return {
        "questions": [
            {
                "id": q.id, "short": q.short, "question": q.question, "answer": q.answer,
                "status": q.status, "status_label": STATUS_LABEL[q.status],
                "claims": [_claim(c) for c in q.claims],
                "tables": [{"title": t.title, "columns": list(t.columns),
                            "rows": [list(r) for r in t.rows], "note": t.note}
                           for t in q.tables],
                "figure_ids": list(q.figure_ids),
                "caveats": list(q.caveats),
                "readiness": dict(q.readiness),
            }
            for q in ms.questions
        ],
        "storyline": {
            "lead": ms.storyline.lead,
            "claims": [{"question": r.question, **_claim(r.claim)}
                       for r in ms.storyline.claims],
            "redundancy": list(ms.storyline.redundancy),
            "readiness": list(ms.storyline.readiness),
            "data": dict(ms.storyline.data),
        },
        "shear": {"label": shear_proxy.PROXY_LABEL, "by_formulation": ms.shear},
    }


def write_report(ms: Manuscript, path: Path) -> None:
    """``reports/storyline.md``: every question, its answer and its claims."""
    lines = ["# Manuscript storyline", "", ms.storyline.lead, "",
             "## Claims, strongest first", "",
             "| Q | claim | effect | uncertainty | status |", "|---|---|---|---|---|"]
    for r in ms.storyline.claims:
        c = r.claim
        lines.append(f"| {r.question.upper()} | {c.text} | {c.effect} | {c.uncertainty} | "
                     f"{STATUS_LABEL[c.status]} |")
    lines += ["", "## Responses that tell the same story", ""]
    for g in ms.storyline.redundancy:
        lines.append(f"- **{g['representative']}** stands for {', '.join(g['members'])}")
    for q in ms.questions:
        lines += ["", f"## {q.id.upper()}. {q.question}", "",
                  f"*{STATUS_LABEL[q.status]}.* {q.answer}", ""]
        for t in q.tables:
            lines += [f"**{t.title}**", "", "| " + " | ".join(t.columns) + " |",
                      "|" + "---|" * len(t.columns)]
            for row in t.rows:
                lines.append("| " + " | ".join("—" if v is None else str(v) for v in row)
                             + " |")
            if t.note:
                lines += ["", t.note]
            lines.append("")
        for cav in q.caveats:
            lines.append(f"- {cav}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
