"""Formulation goals as presets, with the weights behind them still visible.

Raw weight sliders are a poor interface for a judgement. Asked to set a weight on
t50 against one on completeness, most people move them until the answer looks
right, which is optimisation by wishful thinking. Worse, the sliders imply a
precision the data cannot support: two candidates half a desirability point apart
are not distinguished by anything measurable.

So the interface is a goal -- "hit this t50, completeness secondary" -- and the
weights it implies are shown rather than hidden. AC12 requires the weights be
exposed; it does not require them to be the primary control.

Each goal names the responses it cares about, the direction of each, and how much
each matters. Desirability follows Derringer and Suich: every response is mapped
to 0-1, then combined as a weighted geometric mean, so a candidate that misses any
one goal entirely scores zero rather than averaging its way to respectability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Direction = Literal["target", "maximise", "minimise"]


@dataclass(frozen=True)
class ResponseGoal:
    """What we want from one response."""

    response: str
    direction: Direction
    weight: float
    target: float | None = None
    tolerance: float | None = None
    note: str = ""


@dataclass(frozen=True)
class Goal:
    """A named formulation objective."""

    key: str
    label: str
    description: str
    components: tuple[ResponseGoal, ...]
    prompts: tuple[str, ...] = ()

    @property
    def weight_summary(self) -> str:
        return ", ".join(
            f"{c.response} x{c.weight:g}" for c in self.components
        )


#: The goals offered in the formulator tool. Deliberately few: a long list of
#: near-identical objectives is another way of asking the user to tune weights.
GOALS: tuple[Goal, ...] = (
    Goal(
        key="match_t50",
        label="Hit a target release time",
        description=(
            "Get t50 onto the target, with completeness as a secondary concern. "
            "The usual starting point when a release window is specified."
        ),
        components=(
            ResponseGoal(
                "t50", "target", 3.0,
                note="Primary. Deviation either side is penalised equally.",
            ),
            ResponseGoal(
                "pct_24h", "maximise", 1.0,
                note="Secondary. Among formulations that hit the time, prefer the "
                     "one that releases most of the dose.",
            ),
        ),
        prompts=("Target t50 (h)",),
    ),
    Goal(
        key="match_window",
        label="Match a release window",
        description=(
            "Hit specified % released at two timepoints at once -- the form a "
            "dissolution specification is usually written in."
        ),
        components=(
            ResponseGoal(
                "pct_4h", "target", 2.0,
                note="Early control point. Guards against dose dumping.",
            ),
            ResponseGoal(
                "pct_12h", "target", 2.0,
                note="Late control point. Fixes the overall rate.",
            ),
            ResponseGoal(
                "pct_24h", "maximise", 1.0,
                note="Completeness, as a tie-breaker.",
            ),
        ),
        prompts=("Target % at 4 h", "Target % at 12 h"),
    ),
    Goal(
        key="slowest_complete",
        label="Slowest release that still finishes",
        description=(
            "Extend release as far as possible while still reaching near-complete "
            "release by 24 h. Useful when the constraint is a once-daily dose."
        ),
        components=(
            ResponseGoal(
                "t50", "maximise", 2.0,
                note="Push release later.",
            ),
            ResponseGoal(
                "pct_24h", "target", 3.0, target=95.0, tolerance=10.0,
                note="But not so far that the dose is left in the tablet. This is "
                     "the constraint that stops the search running away.",
            ),
        ),
        prompts=(),
    ),
    Goal(
        key="fastest_controlled",
        label="Fastest release that is still controlled",
        description=(
            "The other end: release as quickly as possible while keeping a "
            "controlled profile rather than immediate release."
        ),
        components=(
            ResponseGoal(
                "pct_24h", "maximise", 2.0,
                note="Get the dose out.",
            ),
            ResponseGoal(
                "pct_1h", "target", 3.0, target=20.0, tolerance=15.0,
                note="But keep the first hour modest. Without this the search "
                     "simply returns the lowest polymer content available, which "
                     "is not a controlled-release formulation.",
            ),
        ),
        prompts=(),
    ),
)


def as_payload() -> list[dict[str, object]]:
    """Serialise the goals for the dashboard."""
    return [
        {
            "key": g.key,
            "label": g.label,
            "description": g.description,
            "prompts": list(g.prompts),
            "weights": [
                {
                    "response": c.response,
                    "direction": c.direction,
                    "weight": c.weight,
                    "target": c.target,
                    "tolerance": c.tolerance,
                    "note": c.note,
                }
                for c in g.components
            ],
        }
        for g in GOALS
    ]
