"""Storyline diagnostics: which story can the paper tell?

A paper leads with its strongest supported finding. This gathers every claim
the questions made, orders them by how well the data backs them, shows which
responses say the same thing (so the paper need not report all of them), and
lists how much data each question rests on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipeline.manuscript.model import STATUS_LABEL, Claim, QuestionResult, num

if TYPE_CHECKING:
    from pipeline.analysis import Analysis

_RANK = {"supported": 0, "directional": 1, "not_supported": 2, "gated": 3, "unavailable": 4}


@dataclass(frozen=True)
class RankedClaim:
    question: str
    claim: Claim
    headline: bool


@dataclass(frozen=True)
class Storyline:
    claims: tuple[RankedClaim, ...]
    redundancy: tuple[dict[str, Any], ...]
    readiness: tuple[dict[str, Any], ...]
    data: dict[str, Any]
    lead: str


def build(analysis: Analysis, questions: tuple[QuestionResult, ...]) -> Storyline:
    # A question's first claim is its headline; the rest qualify it. Within a
    # status, headlines come first, in question order, so the lead is never a
    # side remark that happens to have the largest ratio.
    ranked = [RankedClaim(q.id, c, i == 0) for q in questions for i, c in enumerate(q.claims)]
    order = {q.id: n for n, q in enumerate(questions)}
    ranked.sort(key=lambda r: (_RANK[r.claim.status], not r.headline, order[r.question],
                               -(r.claim.signal_to_noise or 0.0)))

    groups = []
    for g in analysis.response_space.groups:
        groups.append({
            "representative": g.representative,
            "members": list(g.members),
            "max_abs_correlation": num(g.max_abs_correlation, 2),
            "justification": g.justification,
        })

    dp = analysis.design_points
    censored = int((dp["censoring_status"] != "none").sum()) if "censoring_status" in dp else 0
    data = {
        "formulations": len(dp),
        "replicates": int(analysis.quality.n_replicates),
        "censored_formulations": censored,
        "censored_fraction": num(censored / len(dp), 3) if len(dp) else None,
        "synthetic": bool(analysis.quality.is_synthetic),
        "cv_profile_rmse_pct": num(analysis.cross_validation.profile_rmse_pct, 2),
    }
    readiness = tuple(
        {"question": q.id, "short": q.short, "status": q.status,
         "status_label": STATUS_LABEL[q.status], "figures": len(q.figure_ids),
         **q.readiness}
        for q in questions
    )
    top = next((r for r in ranked if r.claim.status == "supported"), None)
    if top is None:
        lead = ("No claim is yet supported outright; lead with the directional findings "
                "and say what data would settle them.")
    else:
        lead = f"Lead with {top.question.upper()}: {top.claim.text}"
    if data["synthetic"]:
        lead += " (This database is synthetic, so none of this is a finding yet.)"
    return Storyline(tuple(ranked), tuple(groups), readiness, data, lead)
