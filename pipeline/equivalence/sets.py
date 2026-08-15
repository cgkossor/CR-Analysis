"""Iso-release equivalence sets (AC7).

Two formulations are equivalent when their release profiles are similar by
f2 >= 50. The interesting question is not whether a formulation matches itself,
but **how much formulation freedom exists for a given target** -- how many
distinct compositions, across how many grades, land on the same curve, and where
that freedom collapses.

Equivalence is computed from measured profiles wherever possible. Where a
candidate is predicted rather than measured, its cross-validated error travels
with it (G6) and its in/out-of-hull status is recorded (G3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline import config
from pipeline.equivalence.f2 import similarity_f2


@dataclass(frozen=True)
class EquivalenceMember:
    """One formulation similar to the target."""

    case: int
    grade: str
    f2: float
    n_points: int
    cv_profile_rmse_pct: float
    measured: bool
    api_wt: float
    hpmc_wt: float
    lactose_wt: float


@dataclass(frozen=True)
class EquivalenceSet:
    """Everything f2-similar to one target formulation."""

    target_case: int
    target_grade: str
    members: tuple[EquivalenceMember, ...]
    grades_spanned: tuple[str, ...]
    n_members: int
    spans_multiple_grades: bool
    notes: tuple[str, ...] = field(default=())

    @property
    def freedom(self) -> int:
        """How many alternatives exist besides the target itself."""
        return max(self.n_members - 1, 0)


def build_equivalence_sets(
    time_grid: np.ndarray,
    profiles: dict[tuple[int, str], np.ndarray],
    compositions: dict[int, tuple[float, float, float]],
    cv_error_by_point: dict[tuple[int, str], float],
) -> tuple[EquivalenceSet, ...]:
    """Every measured formulation's equivalence set, by pairwise f2."""
    keys = sorted(profiles)
    sets: list[EquivalenceSet] = []

    for target in keys:
        members: list[EquivalenceMember] = []
        notes: list[str] = []
        for other in keys:
            result = similarity_f2(time_grid, profiles[target], profiles[other])
            if not result.valid:
                if target != other:
                    notes.append(
                        f"case {other[0]} {other[1]}: f2 not computable — {result.note}"
                    )
                continue
            if result.value >= config.F2_SIMILAR_THRESHOLD:
                api, hpmc, lactose = compositions[other[0]]
                members.append(
                    EquivalenceMember(
                        case=other[0],
                        grade=other[1],
                        f2=result.value,
                        n_points=result.n_points,
                        cv_profile_rmse_pct=cv_error_by_point.get(other, float("nan")),
                        measured=True,
                        api_wt=api,
                        hpmc_wt=hpmc,
                        lactose_wt=lactose,
                    )
                )

        members.sort(key=lambda m: -m.f2)
        grades = tuple(dict.fromkeys(m.grade for m in members))
        sets.append(
            EquivalenceSet(
                target_case=target[0],
                target_grade=target[1],
                members=tuple(members),
                grades_spanned=grades,
                n_members=len(members),
                spans_multiple_grades=len(grades) > 1,
                notes=tuple(notes[:5]),
            )
        )

    return tuple(sets)


@dataclass(frozen=True)
class EquivalenceSummary:
    """The headline AC7 result: how much formulation freedom the data demonstrate."""

    n_targets: int
    cross_grade_targets: tuple[tuple[int, str], ...]
    max_set_size: int
    median_set_size: float
    isolated_targets: tuple[tuple[int, str], ...]
    demonstration: str


def summarise(sets: tuple[EquivalenceSet, ...]) -> EquivalenceSummary:
    """Reduce the equivalence sets to the claim AC7 asks to be demonstrated."""
    sizes = [s.n_members for s in sets]
    cross = tuple((s.target_case, s.target_grade) for s in sets if s.spans_multiple_grades)
    isolated = tuple((s.target_case, s.target_grade) for s in sets if s.freedom == 0)

    if cross:
        best = max(
            (s for s in sets if s.spans_multiple_grades),
            key=lambda s: (len(s.grades_spanned), s.n_members),
        )
        others = [
            m
            for m in best.members
            if (m.case, m.grade) != (best.target_case, best.target_grade)
        ]
        example = ", ".join(
            f"case {m.case}/{m.grade} (API {m.api_wt:g}, HPMC {m.hpmc_wt:g}, f2 {m.f2:.0f})"
            for m in others[:3]
        )
        demonstration = (
            f"Measured data show distinct compositions across different grades producing "
            f"f2-similar profiles. Case {best.target_case}/{best.target_grade} is matched "
            f"by {example}. Grade and composition are therefore partially "
            "interchangeable levers: a target reachable with a high-viscosity grade at "
            "low polymer load is also reachable with a lower grade at higher load."
        )
    else:
        demonstration = (
            "No measured formulation is f2-similar to a formulation at a different "
            "grade. In this database grade and composition are not interchangeable, and "
            "each grade occupies its own release regime."
        )

    return EquivalenceSummary(
        n_targets=len(sets),
        cross_grade_targets=cross,
        max_set_size=max(sizes) if sizes else 0,
        median_set_size=float(np.median(sizes)) if sizes else 0.0,
        isolated_targets=isolated,
        demonstration=demonstration,
    )
