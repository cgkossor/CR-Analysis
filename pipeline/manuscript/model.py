"""What a manuscript question returns.

Each question is answered from results the pipeline already computed, and says
how strongly the data backs the answer. The status vocabulary is deliberately
small so a reader can sort claims before reading them:

* ``supported``: the effect is clear of its uncertainty by the rule the
  question states.
* ``directional``: the sign is consistent but the size is not established, or
  the measure is a proxy.
* ``not_supported``: the data do not show it.
* ``gated``: the question needs data this database does not have (G1).
* ``unavailable``: an optional input, such as disintegration, was not supplied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["supported", "directional", "not_supported", "gated", "unavailable"]

STATUS_LABEL: dict[str, str] = {
    "supported": "Supported",
    "directional": "Directional",
    "not_supported": "Not supported",
    "gated": "Gated (needs more data)",
    "unavailable": "Unavailable",
}


@dataclass(frozen=True)
class Claim:
    """One sentence the paper could make, with what stands behind it."""

    text: str
    effect: str
    uncertainty: str
    status: Status
    #: Effect over its uncertainty, where the question has a natural ratio.
    signal_to_noise: float | None = None


@dataclass(frozen=True)
class Table:
    """A small table for the dashboard and the report."""

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    note: str = ""


@dataclass(frozen=True)
class QuestionResult:
    """The answer to one research question."""

    id: str
    short: str
    question: str
    answer: str
    status: Status
    claims: tuple[Claim, ...] = ()
    tables: tuple[Table, ...] = ()
    #: Ids of the figures the question draws, as they appear in captions.json.
    figure_ids: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    #: How much data the answer rests on, for the figure-readiness checklist.
    readiness: dict[str, Any] = field(default_factory=dict)


def num(value: Any, dp: int = 3) -> float | None:
    """A finite float rounded for export, or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, dp)
