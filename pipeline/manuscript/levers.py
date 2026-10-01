"""Q1: composition or HPMC grade, which lever controls release, and do they interact?

The design crosses every composition (case) with every grade, so the question
can be answered without a model: a two-way decomposition of the case x grade
cell means splits each response's spread into what the composition explains,
what the grade explains, and what neither explains alone (their interaction).
Shares are eta-squared on the cell means, so they sum to one and do not depend
on how the DoE model was parameterised.

The F-tests divide by the within-cell replicate variance. Replicates are
vessels from one compression batch, so that variance is analytical
repeatability, not batch-to-batch variation, and the p-values are optimistic.
The shares are the effect sizes to report; the p-values only say whether an
effect is clear of measurement noise at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.manuscript.model import Claim, QuestionResult, Status, Table, num

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.figures.publication import FigureRecord

#: Responses decomposed, in the order they are shown: release through time,
#: then the summary time scale and the curve shape (mechanism).
RESPONSES: tuple[tuple[str, str], ...] = (
    ("pct_2h", "% at 2 h"),
    ("pct_4h", "% at 4 h"),
    ("pct_8h", "% at 8 h"),
    ("pct_12h", "% at 12 h"),
    ("pct_24h", "% at 24 h"),
    ("mdt_h", "MDT"),
    ("log10_td", "log Td"),
    ("weibull_beta", "Weibull β"),
)

#: Responses that grow as release slows (the rest are % released).
TIME_SCALES = frozenset({"mdt_h", "log10_td"})

#: A lever "dominates" when it explains at least half the spread and at least
#: twice what the other lever explains.
DOMINANT_SHARE = 0.5
DOMINANT_RATIO = 2.0
#: Interaction share above which non-additivity is worth a sentence.
MATERIAL_INTERACTION = 0.10

FIGURE_ID = "M1_levers"


@dataclass(frozen=True)
class Decomposition:
    """Two-way decomposition of one response over case x grade."""

    key: str
    label: str
    n_cases: int
    n_grades: int
    #: Cases and grades in the data, before cells were dropped for a full grid.
    n_cases_total: int
    n_grades_total: int
    n_reps: float
    share_composition: float
    share_grade: float
    share_interaction: float
    p_composition: float
    p_grade: float
    p_interaction: float
    f_composition: float
    rho_hpmc: float
    rho_api: float


def decompose(analysis: Analysis, key: str, label: str) -> Decomposition | None:
    """Split one response across composition, grade and their interaction."""
    reps = analysis.replicates
    if key not in reps:
        return None
    frame = reps[["case", "grade", key]].dropna()
    cells = frame.groupby(["case", "grade"])[key].agg(["mean", "var", "count"]).reset_index()
    full = cells.pivot(index="case", columns="grade", values="mean")
    means = complete_grid(full)
    a, b = means.shape
    if a < 3 or b < 2:
        return None
    y = means.to_numpy(dtype=float)
    grand = y.mean()
    row = y.mean(axis=1, keepdims=True)
    col = y.mean(axis=0, keepdims=True)
    ss_case = b * float(((row - grand) ** 2).sum())
    ss_grade = a * float(((col - grand) ** 2).sum())
    ss_int = float(((y - row - col + grand) ** 2).sum())
    total = ss_case + ss_grade + ss_int
    if total <= 0:
        return None

    kept = cells[cells["case"].isin(means.index)]
    n_rep = float(kept["count"].mean())
    df_err = float((kept["count"] - 1).clip(lower=0).sum())
    ms_err = float(((kept["count"] - 1) * kept["var"].fillna(0)).sum() / df_err) if df_err else 0.0

    def test(ss: float, df: float) -> tuple[float, float]:
        # Cell-mean sums of squares scale by n per cell to the replicate scale.
        if ms_err <= 0 or df <= 0 or df_err <= 0:
            return float("nan"), float("nan")
        f = (ss * n_rep / df) / ms_err
        return f, float(stats.f.sf(f, df, df_err))

    f_c, p_c = test(ss_case, a - 1)
    _, p_g = test(ss_grade, b - 1)
    _, p_i = test(ss_int, (a - 1) * (b - 1))

    comp = analysis.design_points.drop_duplicates("case").set_index("case")
    comp = comp.loc[means.index]
    rho_h = stats.spearmanr(comp["hpmc_wt"], row.ravel()).statistic
    rho_a = stats.spearmanr(comp["api_wt"], row.ravel()).statistic
    return Decomposition(
        key=key, label=label, n_cases=a, n_grades=b,
        n_cases_total=int(full.shape[0]), n_grades_total=int(full.shape[1]), n_reps=n_rep,
        share_composition=ss_case / total, share_grade=ss_grade / total,
        share_interaction=ss_int / total, p_composition=p_c, p_grade=p_g,
        p_interaction=p_i, f_composition=f_c,
        rho_hpmc=float(rho_h), rho_api=float(rho_a),
    )


def complete_grid(means: pd.DataFrame) -> pd.DataFrame:
    """The largest complete case x grade block of a table with missing cells.

    The decomposition needs every case in every grade. Dropping cases until the
    grid is full can throw away nearly everything when one grade was run for
    only a few cases; dropping that grade instead keeps the rest. Every subset
    of at least two grades is tried (there are only a handful) and the one that
    keeps the most cells wins, ties going to more grades.
    """
    best = means.iloc[0:0, 0:0]
    best_key = (-1, -1)
    cols = list(means.columns)
    for k in range(len(cols), 1, -1):
        for subset in combinations(cols, k):
            block = means[list(subset)].dropna(axis=0, how="any")
            key = (block.shape[0] * block.shape[1], block.shape[1])
            if key > best_key:
                best, best_key = block, key
    return best


def _dominant(d: Decomposition) -> str | None:
    c, g = d.share_composition, d.share_grade
    if c >= DOMINANT_SHARE and c >= DOMINANT_RATIO * g:
        return "composition"
    if g >= DOMINANT_SHARE and g >= DOMINANT_RATIO * c:
        return "grade"
    return None


def answer(analysis: Analysis) -> tuple[QuestionResult, list[Decomposition]]:
    """Q1, from the case x grade cell means."""
    decs = [d for key, label in RESPONSES if (d := decompose(analysis, key, label))]
    question = ("Which lever controls release, composition (API / HPMC / lactose) or "
                "HPMC viscosity grade, and do the two interact?")
    if not decs:
        return QuestionResult(
            id="q1", short="Composition vs grade", question=question,
            answer="The design does not cross enough cases with grades to separate "
                   "the two levers.",
            status="not_supported", figure_ids=(),
        ), []

    rows = []
    for d in decs:
        rows.append((
            d.label, num(100 * d.share_composition, 1), num(100 * d.share_grade, 1),
            num(100 * d.share_interaction, 1), _dominant(d) or "neither",
            num(d.rho_hpmc, 2), num(d.rho_api, 2),
        ))
    table = Table(
        title="Share of the case x grade spread explained (%)",
        columns=("response", "composition", "grade", "interaction", "dominant lever",
                 "ρ with HPMC wt%", "ρ with API wt%"),
        rows=tuple(rows),
        note="Eta-squared on the case x grade cell means. ρ is the Spearman correlation "
             "of the composition's grade-averaged response with its HPMC or API content.",
    )

    time_decs = [d for d in decs if d.key != "weibull_beta"]
    comp_led = [d for d in time_decs if _dominant(d) == "composition"]
    grade_led = [d for d in time_decs if _dominant(d) == "grade"]
    mean_c = float(np.mean([d.share_composition for d in time_decs])) if time_decs else 0.0
    mean_g = float(np.mean([d.share_grade for d in time_decs])) if time_decs else 0.0
    mean_i = float(np.mean([d.share_interaction for d in time_decs])) if time_decs else 0.0

    # A decomposition on part of the design describes that part only.
    partial = any(d.n_cases < d.n_cases_total or d.n_grades < d.n_grades_total for d in decs)
    used = min(decs, key=lambda d: d.n_cases * d.n_grades)
    coverage = (f"{used.n_cases} of {used.n_cases_total} compositions x {used.n_grades} of "
                f"{used.n_grades_total} grades")

    claims: list[Claim] = []
    status: Status
    if len(comp_led) > len(time_decs) / 2:
        lead, other = "composition", "grade"
        status = "supported"
    elif len(grade_led) > len(time_decs) / 2:
        lead, other = "grade", "composition"
        status = "supported"
    else:
        lead, other = ("composition", "grade") if mean_c >= mean_g else ("grade", "composition")
        status = "directional"
    lead_share = mean_c if lead == "composition" else mean_g
    other_share = mean_g if lead == "composition" else mean_c
    claims.append(Claim(
        text=f"{lead.capitalize()} is the stronger lever on release rate.",
        effect=f"{100 * lead_share:.0f}% vs {100 * other_share:.0f}% of the spread "
               f"(mean over {len(time_decs)} release responses)",
        uncertainty=f"dominant in {len(comp_led if lead == 'composition' else grade_led)} "
                    f"of {len(time_decs)} responses; {coverage}",
        status="directional" if partial else status,
        signal_to_noise=num(lead_share / other_share, 2) if other_share > 0 else None,
    ))

    material = [d for d in time_decs if d.share_interaction >= MATERIAL_INTERACTION]
    clear = [d for d in material if np.isfinite(d.p_interaction) and d.p_interaction < 0.05]
    lever = analysis.lever_effects
    fold = ""
    if len(lever) >= 2 and np.isfinite(lever["fold_change_td"]).all():
        first, last = lever.iloc[0], lever.iloc[-1]
        fold = (f"; +10 wt% HPMC multiplies Td by ×{first['fold_change_td']:.2f} at "
                f"{first['grade']} but ×{last['fold_change_td']:.2f} at {last['grade']}")
    claims.append(Claim(
        text=("The levers interact: the grade changes how much a composition change moves "
              "release." if material else "The levers act close to additively."),
        effect=f"interaction {100 * mean_i:.0f}% of the spread on average{fold}",
        uncertainty=f"≥{100 * MATERIAL_INTERACTION:.0f}% in {len(material)} of "
                    f"{len(time_decs)} responses; clear of replicate noise in {len(clear)}",
        status=("supported" if len(clear) > len(time_decs) / 2
                else "directional" if material else "not_supported"),
    ))

    # Signed as correlation with release speed: % released rises with speed,
    # MDT and Td fall. Mixing the raw signs would cancel them out.
    speed = [(-1.0 if d.key in TIME_SCALES else 1.0) * d.rho_hpmc
             for d in time_decs if np.isfinite(d.rho_hpmc)]
    if speed:
        med = float(np.median(speed))
        claims.append(Claim(
            text=("Across compositions, more HPMC slows release." if med < 0 else
                  "Across compositions, more HPMC does not slow release."),
            effect=f"median ρ between HPMC wt% and release speed {med:+.2f}",
            uncertainty=f"n = {time_decs[0].n_cases} compositions, {len(speed)} responses",
            status="supported" if abs(med) >= 0.7 else "directional",
        ))

    beta = next((d for d in decs if d.key == "weibull_beta"), None)
    if beta is not None:
        blead = _dominant(beta)
        claims.append(Claim(
            text=("The curve shape (Weibull β, the release mechanism) is set mainly by "
                  f"{blead}." if blead else
                  "Neither lever alone sets the curve shape (Weibull β)."),
            effect=f"composition {100 * beta.share_composition:.0f}%, grade "
                   f"{100 * beta.share_grade:.0f}%, interaction "
                   f"{100 * beta.share_interaction:.0f}%",
            uncertainty="β is fitted per vessel; its spread includes fit error",
            status="directional",
        ))

    answer_text = (
        f"{lead.capitalize()} explains {100 * lead_share:.0f}% of the spread in release "
        f"on average and {other} {100 * other_share:.0f}%. "
        + (f"Their interaction adds {100 * mean_i:.0f}%, so a composition change does "
           "not move release by the same amount in every grade."
           if material else
           f"Their interaction is small ({100 * mean_i:.0f}%), so the two act close to "
           "additively.")
    )
    return QuestionResult(
        id="q1", short="Composition vs grade", question=question, answer=answer_text,
        status=claims[0].status, claims=tuple(claims), tables=(table,),
        figure_ids=(FIGURE_ID,),
        caveats=(
            "p-values use within-batch replicate variance and are optimistic; the shares "
            "are the effect sizes.",
            f"Composition is {used.n_cases_total} mixtures of three components, so "
            "'composition' bundles API, HPMC and lactose together. The ρ columns say which "
            "component carries it.",
        ) + ((
            f"Cells are missing from the case x grade grid, so the decomposition uses the "
            f"largest complete block: {coverage}. It describes that block, not the whole "
            "design, and is held at directional.",
        ) if partial else ()),
        readiness={"formulations": int(used.n_cases * used.n_grades),
                   "cases_used": used.n_cases, "cases_total": used.n_cases_total,
                   "grades_used": used.n_grades, "grades_total": used.n_grades_total,
                   "replicates_per_cell": num(decs[0].n_reps, 1),
                   "responses": len(decs)},
    ), decs


def render(
    analysis: Analysis, decs: list[Decomposition], out: Path, banner: str | None
) -> list[FigureRecord]:
    """(A) stacked shares per response; (B) the HPMC lever's size in each grade."""
    from pipeline.figures import publication as pub

    if not decs:
        return []
    fig, (ax_a, ax_b) = pub.new_figure(pub.DOUBLE, 2.9, ncols=2,
                                       gridspec_kw={"width_ratios": [1.6, 1]})
    labels = [d.label for d in decs]
    y = np.arange(len(decs))[::-1]
    parts = (
        ("Composition", [d.share_composition for d in decs], pub.OKABE_ITO[0]),
        ("Grade", [d.share_grade for d in decs], pub.OKABE_ITO[1]),
        ("Interaction", [d.share_interaction for d in decs], pub.OKABE_ITO[2]),
    )
    left = np.zeros(len(decs))
    for name, vals, colour in parts:
        v = 100 * np.asarray(vals)
        ax_a.barh(y, v, left=left, color=colour, height=0.66, label=name,
                  edgecolor="white", linewidth=0.5)
        left += v
    ax_a.set_yticks(y, labels)
    ax_a.set_xlim(0, 100)
    ax_a.set_xlabel("Share of case × grade spread (%)")
    pub.categorical(ax_a, "y")
    ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False)

    lever = analysis.lever_effects
    lever = lever[np.isfinite(lever["fold_change_td"])]
    if lever.empty:
        return []
    grades = list(lever["grade"])
    xs = np.arange(len(grades))
    colours = [pub.grade_style(g, i).colour for i, g in enumerate(grades)]
    ax_b.bar(xs, lever["fold_change_td"], color=colours, width=0.6)
    ax_b.axhline(1.0, color=pub.INK, lw=0.6, ls="--")
    ax_b.set_xticks(xs, grades)
    ax_b.set_ylabel("Td multiplier for +10 wt% HPMC")
    ax_b.set_ylim(0, max(float(lever["fold_change_td"].max()) * 1.15, 1.2))
    pub.categorical(ax_b, "x")
    pub.label_panels([ax_a, ax_b])
    file = pub.save(fig, out, FIGURE_ID, banner=banner)

    caption = (
        "Composition versus HPMC grade as levers on release. (A) Share of the spread in "
        "each response across the case × grade design explained by composition, by "
        "grade, and by their interaction (eta-squared on cell means). (B) Multiplier on "
        "the Weibull time scale Td from replacing 10 wt% lactose with HPMC at the design "
        "centroid, per grade; a smaller bar means the HPMC lever is weaker in that grade."
    )
    return [pub.FigureRecord(FIGURE_ID, "Q1", 1, caption, file)]

