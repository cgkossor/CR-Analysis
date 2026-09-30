"""Q2: when can the HPMC grade be interchanged?

A grade swap is a real formulation question: a grade goes out of stock, or a
lower-viscosity grade processes better. For each composition the measured mean
profiles in two grades are compared by f2, the regulatory similarity test. A
pair at or above the f2 threshold is interchangeable at that composition.

The question then is where in composition space swaps work. Each comparison is
related to the composition's HPMC and API content, so the answer is a region
("above about X wt% HPMC") rather than a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from pipeline import config
from pipeline.equivalence.f2 import similarity_f2
from pipeline.manuscript.model import Claim, QuestionResult, Status, Table, num

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.figures.publication import FigureRecord

FIGURE_ID = "M2_grade_swap"


@dataclass(frozen=True)
class Swap:
    """One composition, two grades, and whether they match."""

    case: int
    grade_a: str
    grade_b: str
    api_wt: float
    hpmc_wt: float
    f2: float
    valid: bool

    @property
    def similar(self) -> bool:
        return self.valid and self.f2 >= config.F2_SIMILAR_THRESHOLD


def grades_by_viscosity(analysis: Analysis) -> list[str]:
    dp = analysis.design_points.drop_duplicates("grade").sort_values("viscosity_cp")
    return [str(g) for g in dp["grade"]]


def swaps(analysis: Analysis) -> list[Swap]:
    """Every same-composition grade pair with measured profiles in both grades."""
    grades = grades_by_viscosity(analysis)
    comp = analysis.design_points.drop_duplicates("case").set_index("case")
    out: list[Swap] = []
    for case in sorted(comp.index):
        for ga, gb in combinations(grades, 2):
            a = analysis.observed_profiles.get((int(case), ga))
            b = analysis.observed_profiles.get((int(case), gb))
            if a is None or b is None:
                continue
            res = similarity_f2(analysis.time_grid, a, b)
            out.append(Swap(int(case), ga, gb, float(comp.at[case, "api_wt"]),
                            float(comp.at[case, "hpmc_wt"]), float(res.value), res.valid))
    return out


def _threshold_hpmc(pair: list[Swap]) -> float | None:
    """The HPMC content above which every swap in this pair works, if one exists."""
    ok = sorted(s.hpmc_wt for s in pair if s.similar)
    bad = sorted(s.hpmc_wt for s in pair if s.valid and not s.similar)
    if not ok or not bad:
        return None
    if min(ok) > max(bad):
        return (min(ok) + max(bad)) / 2
    return None


def answer(analysis: Analysis) -> tuple[QuestionResult, list[Swap]]:
    """Q2, from measured profiles only (no model)."""
    sw = swaps(analysis)
    question = "When can one HPMC grade be swapped for another without changing release?"
    valid = [s for s in sw if s.valid]
    if not valid:
        return QuestionResult(
            id="q2", short="Grade interchange", question=question,
            answer="No composition has profiles in two grades that f2 can compare.",
            status="not_supported",
        ), sw

    grades = grades_by_viscosity(analysis)
    pairs = list(combinations(grades, 2))
    rows = []
    for case in sorted({s.case for s in sw}):
        these = {(s.grade_a, s.grade_b): s for s in sw if s.case == case}
        first = next(iter(these.values()))
        row: list[object] = [case, num(first.api_wt, 1), num(first.hpmc_wt, 1)]
        for pa in pairs:
            s = these.get(pa)
            row.append(None if s is None or not s.valid else num(s.f2, 0))
        rows.append(tuple(row))
    table = Table(
        title="f2 between grades at the same composition",
        columns=("case", "API wt%", "HPMC wt%", *[f"{a} vs {b}" for a, b in pairs]),
        rows=tuple(rows),
        note=f"f2 ≥ {config.F2_SIMILAR_THRESHOLD:g} means the two grades are interchangeable "
             f"at that composition. Blank: fewer than {config.F2_MIN_POINTS} comparable points.",
    )

    claims: list[Claim] = []
    n_ok = sum(s.similar for s in valid)
    claims.append(Claim(
        text=(f"The grade can be swapped in {n_ok} of {len(valid)} same-composition "
              "comparisons." if n_ok else
              "No grade can be swapped for another at any tested composition."),
        effect=f"median f2 {np.median([s.f2 for s in valid]):.0f}",
        uncertainty=f"f2 on measured means; threshold {config.F2_SIMILAR_THRESHOLD:g}",
        status="supported",
    ))
    for ga, gb in pairs:
        pair = [s for s in valid if (s.grade_a, s.grade_b) == (ga, gb)]
        if len(pair) < 4:
            continue
        ok = [s for s in pair if s.similar]
        rho = float(stats.spearmanr([s.hpmc_wt for s in pair], [s.f2 for s in pair]).statistic)
        cut = _threshold_hpmc(pair)
        status: Status
        if cut is not None:
            text = f"{ga} and {gb} are interchangeable above about {cut:.0f} wt% HPMC."
            status = "supported"
        elif ok:
            text = (f"{ga} and {gb} are interchangeable at {len(ok)} of {len(pair)} "
                    "compositions, with no clean HPMC threshold.")
            status = "directional"
        else:
            text = f"{ga} and {gb} are not interchangeable at any tested composition."
            status = "supported"
        claims.append(Claim(
            text=text,
            effect=f"{len(ok)}/{len(pair)} similar; ρ(HPMC wt%, f2) = {rho:+.2f}",
            uncertainty=f"n = {len(pair)} compositions",
            status=status,
            signal_to_noise=num(abs(rho) * np.sqrt(len(pair) - 1), 2),
        ))

    adjacent = [(grades[i], grades[i + 1]) for i in range(len(grades) - 1)]
    adj_ok = sum(s.similar for s in valid if (s.grade_a, s.grade_b) in adjacent)
    far_ok = sum(s.similar for s in valid if (s.grade_a, s.grade_b) not in adjacent)
    answer_text = (
        f"{n_ok} of {len(valid)} same-composition grade pairs pass f2 ≥ "
        f"{config.F2_SIMILAR_THRESHOLD:g}: {adj_ok} between neighbouring grades and "
        f"{far_ok} between the extremes. "
        + " ".join(c.text for c in claims[1:])
    ).strip()
    return QuestionResult(
        id="q2", short="Grade interchange", question=question, answer=answer_text,
        status="supported", claims=tuple(claims), tables=(table,),
        figure_ids=(FIGURE_ID,),
        caveats=(
            "f2 compares mean profiles; it does not test batch-to-batch variation.",
            "A swap judged here holds at a tested composition. Between compositions, use "
            "the Formulator's Substitute tab, which carries the model's error.",
        ),
        readiness={"comparisons": len(valid), "not_comparable": len(sw) - len(valid)},
    ), sw


def render(
    analysis: Analysis, sw: list[Swap], out: Path, banner: str | None
) -> list[FigureRecord]:
    """One panel per grade pair: composition map, filled where the swap works."""
    from pipeline.figures import publication as pub

    grades = grades_by_viscosity(analysis)
    pairs = [p for p in combinations(grades, 2)
             if any((s.grade_a, s.grade_b) == p and s.valid for s in sw)]
    if not pairs:
        return []
    fig, axes = pub.new_figure(pub.DOUBLE, 2.6, ncols=len(pairs), sharex=True, sharey=True)
    axes = list(np.atleast_1d(axes))
    for i, (ax, (ga, gb)) in enumerate(zip(axes, pairs, strict=True)):
        pair = [s for s in sw if (s.grade_a, s.grade_b) == (ga, gb) and s.valid]
        for s in pair:
            ax.scatter(s.api_wt, s.hpmc_wt, s=34,
                       facecolor=pub.OKABE_ITO[3] if s.similar else "white",
                       edgecolor=pub.OKABE_ITO[3] if s.similar else pub.INK,
                       linewidth=0.8, zorder=3)
            ax.annotate(f"{s.f2:.0f}", (s.api_wt, s.hpmc_wt), xytext=(4, 3),
                        textcoords="offset points", fontsize=6, color=pub.MUTED)
        pub.header_note(ax, f"{ga} vs {gb}")
        ax.margins(x=0.12, y=0.12)
        ax.set_xlabel("API (wt%)")
        if i == 0:
            ax.set_ylabel("HPMC (wt%)")
        pub.auto_minor(ax)
    pub.label_panels(axes)
    file = pub.save(fig, out, FIGURE_ID, banner=banner)
    caption = (
        "Where the HPMC grade can be interchanged. Each panel compares two grades at every "
        "tested composition (API and HPMC content; lactose is the balance). Filled points: "
        f"the measured mean profiles are similar (f2 ≥ {config.F2_SIMILAR_THRESHOLD:g}); "
        "open points: they are not. Numbers are the f2 values."
    )
    return [pub.FigureRecord(FIGURE_ID, "Q2", 2, caption, file)]
