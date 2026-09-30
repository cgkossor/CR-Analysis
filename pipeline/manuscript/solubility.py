"""Q5: how does API solubility change the release mechanism and the curves?

This needs more than one API. G1 goes further: no solubility claim is made
until there are at least two APIs in each solubility class, because with one
API per class solubility is indistinguishable from everything else that
differs between the two molecules. Until then the question renders as gated,
with what it is waiting for. The cross-API comparison step fills it in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.manuscript.model import QuestionResult

if TYPE_CHECKING:
    from pipeline.analysis import Analysis

#: G1: APIs needed in each solubility class before a solubility claim.
MIN_APIS_PER_CLASS = 2

QUESTION = "How does API solubility change the release mechanism and the dissolution curves?"


def answer(analysis: Analysis) -> QuestionResult:
    apis = tuple(analysis.quality.apis)
    return QuestionResult(
        id="q5", short="Solubility effect", question=QUESTION,
        answer=(
            f"Insufficient data: {len(apis)} API loaded ({', '.join(apis)}). A solubility "
            f"claim needs at least {MIN_APIS_PER_CLASS} APIs in each solubility class "
            "(G1). With fewer, solubility cannot be separated from every other difference "
            "between the molecules, such as particle size, wetting or dose. Once more APIs "
            "are run on the same design, this section compares release mechanism (Weibull β) "
            "and curves across them. Even then the result stays labelled as confounded with "
            "molecule identity and directional."
        ),
        status="gated",
        caveats=(
            "Cross-API effects that do not involve solubility, such as whether the "
            "composition and grade levers transfer, need only two APIs and are reported "
            "separately once available.",
        ),
        readiness={"apis": len(apis), "apis_needed_per_class": MIN_APIS_PER_CLASS},
    )
