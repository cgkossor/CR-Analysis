"""Generated interpretation for each table and plot.

Two kinds of text, deliberately separated:

**Takeaway** -- what *this* data shows, generated from the numbers. It names
effects, sizes and directions, so it changes when the data changes. A sentence
that would read the same on any dataset is not a takeaway.

**Method note** -- what the technique is and how to read it. Fixed text, because
the method does not change with the data.

Both stay honest about limits. Where a result is fragile the takeaway says so
rather than reporting the headline and leaving the caveat to a footnote nobody
reads.
"""

from __future__ import annotations

import numpy as np

from pipeline.doe.anova import AnovaTable
from pipeline.doe.effects import EffectRanking, InteractionProfile, Trace
from pipeline.doe.responses import ResponseData

METHOD_NOTES: dict[str, str] = {
    "anova": (
        "Analysis of variance splits the variation in the response into the part "
        "the model explains and the part it does not, then asks whether each term "
        "explains more than chance would. Adjusted (Type III) sums of squares test "
        "every term as though it were entered last, so the answer does not depend "
        "on the order the terms happen to appear in — which matters here because a "
        "mixture design makes the columns correlated by construction. "
        "P below 0.05 is the usual threshold, but significance is not importance: "
        "read the Adj SS column to see how much of the variation a term actually "
        "accounts for."
    ),
    "model_summary": (
        "S is the typical size of a residual, in the response's own units. R-sq is "
        "the fraction of variation explained and never falls when terms are added. "
        "R-sq(adj) penalises extra terms. R-sq(pred) comes from leaving each point "
        "out and predicting it, so it is the only one of the three that says "
        "anything about a formulation the model has not seen. A large gap between "
        "adj and pred means the model is fitting noise."
    ),
    "pareto": (
        "The Pareto chart ranks terms by the size of their effect divided by its "
        "own standard error, so terms measured in different units can be compared. "
        "Bars past the reference line are significant at 5%. The second, further "
        "line is the Bonferroni threshold: the bar has to clear it to be "
        "significant once you account for having tested every term at once."
    ),
    "half_normal": (
        "The half-normal plot needs no error estimate at all. Terms that do nothing "
        "scatter along a straight line through the origin; terms that do something "
        "fall off it to the right. This makes it more trustworthy than the Pareto "
        "when there are few residual degrees of freedom, because it does not depend "
        "on estimating the noise."
    ),
    "cox_trace": (
        "On a mixture you cannot change one component while holding the others "
        "fixed — they sum to 100%, so something must absorb the change. A Cox "
        "response trace varies one component and lets the others take up the "
        "difference in their existing proportions, which is what actually happens "
        "when you reformulate. A steep trace means the response is sensitive to "
        "that component; a flat one means it is not."
    ),
    "interaction": (
        "Each line shows how the response changes with the component, drawn "
        "separately for each HPMC grade. Parallel lines mean the two levers act "
        "independently — the effect of one does not depend on the other. Lines that "
        "fan apart mean they interact, and the size of the fan is the size of the "
        "interaction."
    ),
    "contour": (
        "Contours join compositions predicted to give the same response, with "
        "lactose making up the balance to 100%. Closely spaced contours mean the "
        "response changes quickly there. The blank region lies outside the "
        "compositions actually tested, so no prediction is offered — the tool will "
        "not extrapolate into it."
    ),
    "surface_3d": (
        "The same fitted surface as the contour plot, seen in relief. Useful for "
        "spotting curvature and saddle regions that contour spacing can disguise."
    ),
}


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "not computable"
    return f"{value:,.{digits}g}"


def anova_takeaway(table: AnovaTable, response: ResponseData) -> str:
    """What the ANOVA says about this response, in a sentence or three."""
    if table.residual_df <= 0:
        return (
            f"The model for {response.spec.label.lower()} is saturated — "
            f"{table.n_obs} design points for {table.n_terms} terms — so nothing "
            "can be tested. Reduce the model before reading anything into it."
        )

    groups = [r for r in table.rows if r.is_group]
    groups_sorted = sorted(groups, key=lambda r: -r.adj_ss)
    significant = [r for r in groups_sorted if r.significant]

    parts: list[str] = []
    if significant:
        top = significant[0]
        share = top.adj_ss / table.total_ss if table.total_ss > 0 else float("nan")
        p_text = (
            "< 0.0001" if top.p_value < 1e-4 else f"= {top.p_value:.4f}"
        )
        if np.isfinite(share) and share >= 0.10:
            parts.append(
                f"**{top.source}** is the largest single source of variation in "
                f"{response.spec.label.lower()}, accounting for {share:.0%} of the "
                f"total on its own and significant at p {p_text}."
            )
        else:
            # Adjusted sums of squares measure what a term adds *last*. On a
            # mixture the columns overlap heavily, so every term can add little
            # individually while the model as a whole explains almost everything.
            # Calling the largest of several small contributions "dominant" would
            # be badly misleading.
            parts.append(
                f"No single source stands out for {response.spec.label.lower()}: the "
                f"largest, {top.source.lower()}, adds only {share:.0%} of the total "
                f"variation once the other terms are present (p {p_text}). That is "
                "characteristic of a mixture design rather than a weak model — the "
                "components sum to a constant, so the terms overlap and each one adds "
                "little that the others have not already accounted for. Judge the "
                "model as a whole and read the effect plots for direction."
            )
        interaction = next(
            (r for r in significant if "grade" in r.source.lower()), None
        )
        if interaction is not None and interaction is not top:
            ip = (
                "< 0.0001" if interaction.p_value < 1e-4
                else f"= {interaction.p_value:.4f}"
            )
            parts.append(
                f"The {interaction.source.lower()} term is also significant "
                f"(p {ip}), so the composition effect depends on which grade is "
                "used — the two levers are not additive."
            )
        elif interaction is not None and interaction is top:
            parts.append(
                "That the interaction outweighs composition alone is the substantive "
                "result: how much polymer to add is not a question that can be "
                "answered without naming the grade."
            )
    else:
        parts.append(
            f"No source reaches significance for {response.spec.label.lower()}. "
            "Either the factors genuinely do not move this response, or the design "
            "lacks the power to show it."
        )

    parts.append(
        f"The model explains {table.r_squared:.1%} of the variation "
        f"(R-sq(adj) {table.adj_r_squared:.1%}), and predicts held-out points with "
        f"R-sq(pred) {table.pred_r_squared:.1%}. Typical residual {_fmt(table.s)} "
        f"{response.spec.units}."
    )

    if np.isfinite(table.adj_r_squared) and np.isfinite(table.pred_r_squared):
        gap = table.adj_r_squared - table.pred_r_squared
        if gap > 0.2:
            parts.append(
                f"**The {gap:.2f} gap between adjusted and predicted R-sq means this "
                "model describes the data it was fitted to better than it predicts "
                "new formulations.** Treat its predictions with caution."
            )

    if response.n_missing:
        parts.append(response.coverage_note)

    return " ".join(parts)


def effects_takeaway(ranking: EffectRanking, response: ResponseData) -> str:
    significant = [e for e in ranking.effects if e.significant]
    if not significant:
        return (
            f"No individual term reaches significance for "
            f"{response.spec.label.lower()}."
        )
    strongest = significant[0]
    beyond_bonferroni = [
        e for e in significant if np.isfinite(ranking.bonferroni_t)
        and e.abs_t >= ranking.bonferroni_t
    ]
    text = (
        f"{len(significant)} of {len(ranking.effects)} terms are significant, led by "
        f"`{strongest.term}` (|t| = {strongest.abs_t:.1f}). "
    )
    if beyond_bonferroni:
        text += (
            f"{len(beyond_bonferroni)} clear the stricter Bonferroni line, so they "
            "survive having tested every term at once."
        )
    else:
        text += (
            "None clears the stricter Bonferroni line, so read these as suggestive "
            "rather than established once the number of tests is accounted for."
        )
    if ranking.note:
        text += " " + ranking.note
    return text


def trace_takeaway(traces: list[Trace], response: ResponseData) -> str:
    """Which component moves this response most, over its tested range."""
    spans: list[tuple[str, float]] = []
    for tr in traces:
        ys = np.asarray(tr.y_values, dtype=float)
        finite = ys[np.isfinite(ys)]
        if finite.size:
            spans.append((tr.factor, float(finite.max() - finite.min())))
    if not spans:
        return ""
    spans.sort(key=lambda t: -abs(t[1]))
    top, top_span = spans[0]
    others = ", ".join(f"{n} {abs(s):.3g}" for n, s in spans[1:])
    return (
        f"Across its tested range, **{top.upper()}** moves "
        f"{response.spec.label.lower()} most — a span of {abs(top_span):.3g} "
        f"{response.spec.units} — against {others}. Each trace varies one component "
        "and lets the others take up the difference in their existing proportions, "
        "which is what happens when you reformulate."
    )


def interaction_takeaway(profile: InteractionProfile) -> str:
    return profile.interpretation


def contour_takeaway(
    z_ranges: list[tuple[str, float, float]], response: ResponseData
) -> str:
    """How much the achievable response differs between grades."""
    if not z_ranges:
        return ""
    widest = max(z_ranges, key=lambda t: t[2] - t[1])
    narrowest = min(z_ranges, key=lambda t: t[2] - t[1])
    text = (
        f"Within the tested compositions, {widest[0]} spans "
        f"{widest[1]:.3g}–{widest[2]:.3g} {response.spec.units} of "
        f"{response.spec.label.lower()} — the widest range of any grade — while "
        f"{narrowest[0]} spans only {narrowest[1]:.3g}–{narrowest[2]:.3g}. "
    )
    if widest[0] != narrowest[0]:
        text += (
            f"Choosing {widest[0]} therefore buys more formulation latitude than "
            f"{narrowest[0]} does, before any composition is chosen."
        )
    return text
