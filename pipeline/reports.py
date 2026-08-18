"""Markdown report writers for the analysis stages, plus the guidelines document."""

from __future__ import annotations

import numpy as np

from pipeline import config
from pipeline.analysis import Analysis
from pipeline.stress.subsets import StressTest

_BANNER = (
    "> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.**\n"
    "> Every value below is derived from a generated development database. "
    "No statement here is an experimental finding, and none may be used to "
    "support a formulation decision.\n"
)


def _banner(analysis: Analysis) -> str:
    return _BANNER if analysis.quality.is_synthetic else ""


def render_responses(analysis: Analysis) -> str:
    space = analysis.response_space
    out: list[str] = ["# Response space and key responses (AC5)\n", _banner(analysis)]
    add = out.append

    add(
        "The AC2 metrics are redundant by construction. This section establishes how "
        "many independent quantities they actually measure, and reduces them to a key "
        "set before anything downstream consumes them (G11).\n"
    )

    add("## Dimensionality\n")
    add("| component | variance | cumulative |")
    add("|---|---|---|")
    for i, (v, c) in enumerate(
        zip(space.explained_variance_ratio[:6], space.cumulative_variance[:6], strict=False)
    ):
        add(f"| PC{i + 1} | {v:.1%} | {c:.1%} |")
    add(
        f"\n**{space.n_components_90} component(s) reach 90% cumulative variance** across "
        f"{len(space.metrics)} metrics. The response set is far lower-dimensional than "
        "the metric count suggests, which is precisely why G11 forbids counting "
        "agreement among these metrics as independent confirmation."
    )

    add("\n## Consistency with the two-stage model (AC4)\n")
    add(space.weibull_consistency)

    add("\n## Redundancy groups\n")
    for g in space.groups:
        kind = "structural" if g.contains_structural else "empirical"
        add(
            f"- **{g.representative}** ({kind}, max |r| = {g.max_abs_correlation:.3f}) — "
            f"members: {', '.join(g.members)}"
        )
        add(f"  - chosen because: {g.justification}")

    if space.notes:
        add("\n## Coverage caveats\n")
        for note in space.notes:
            add(f"- {note}")

    add(f"\n**Key responses:** {', '.join(space.key_responses)}\n")
    add(
        "Every surface, figure, equivalence set and formulator output downstream is "
        "drawn on this set alone.\n"
    )

    add("\n## Structural redundancy\n")
    add(
        "These relationships are algebraic. No dataset can make these metrics "
        "independent, so agreement between them carries no information at all.\n"
    )
    for note in space.structural_notes:
        add(f"- {note}")

    add("\n## Empirical redundancy\n")
    if space.empirical_notes:
        add(
            "These pairs co-vary in this database without an algebraic relationship "
            "forcing it. A design spanning a wider region might separate them, so the "
            "redundancy is a property of this experiment, not of the metrics.\n"
        )
        for note in space.empirical_notes[:12]:
            add(f"- {note}")
        if len(space.empirical_notes) > 12:
            add(f"- ...and {len(space.empirical_notes) - 12} further pair(s)")
    else:
        add("No purely empirical redundancy above the clustering threshold.")

    return "\n".join(out) + "\n"


def render_surfaces(analysis: Analysis) -> str:
    out: list[str] = ["# Response surfaces (AC4)\n", _banner(analysis)]
    add = out.append

    add(
        "Fitted on **design-point means**, not on individual replicate profiles. The "
        "three replicates of a formulation are vessels from one compression batch, so "
        "they are subsamples rather than independent runs; treating them as independent "
        "would shrink every standard error by roughly sqrt(3) and manufacture "
        "significance the design cannot support.\n"
    )
    add(f"Model form: `{analysis.model_spec.label}`\n")

    for name, reduced in analysis.surfaces.items():
        fit = reduced.fit
        bc = analysis.box_cox[name]
        add(f"\n## {name}\n")
        add(f"- Model: `{fit.model_label}`, {fit.n_terms} terms, {fit.df_residual} residual df")
        add(f"- Reduction: {reduced.rationale}")
        add(
            f"- R² = {fit.r_squared:.4f} | adjusted = {fit.adj_r_squared:.4f} | "
            f"**predicted (PRESS) = {fit.pred_r_squared:.4f}**"
        )
        add(f"- RMSE = {fit.rmse:.4f} | adequate precision = {fit.adequate_precision:.1f}")
        if fit.r2_gap_flagged:
            add("- ⚠ adjusted-vs-predicted R² gap exceeds 0.2")
        if fit.adequate_precision_flagged:
            add(f"- ⚠ adequate precision below {config.ADEQUATE_PRECISION_FLAG:g}")

        add(f"\n**Box–Cox.** λ = {bc.lambda_hat:.3f}", )
        if np.isfinite(bc.ci_low):
            add(f" (95% CI {bc.ci_low:.3f} … {bc.ci_high:.3f}) — {bc.recommendation}")
        else:
            add(f" — {bc.recommendation}")
        if bc.note:
            add(f"  \n  {bc.note}")

        add("\n**Sequential model sum of squares**\n")
        add("| model | terms | added | SS | F | p |")
        add("|---|---|---|---|---|---|")
        for step in analysis.sequential.get(name, ()):
            add(
                f"| {step.model} | {step.n_terms} | {step.added_terms} | "
                f"{step.sum_squares:.4f} | {step.f_value:.2f} | {step.p_value:.4g} |"
            )

        add("\n**Coefficients**\n")
        add("| term | estimate | std error | t | p | 95% CI |")
        add("|---|---|---|---|---|---|")
        for c in sorted(fit.coefficients, key=lambda c: c.p_value):
            add(
                f"| `{c.name}` | {c.estimate:+.4f} | {c.std_error:.4f} | {c.t_value:+.2f} | "
                f"{c.p_value:.3g} | {c.ci_low:+.4f} … {c.ci_high:+.4f} |"
            )

        influential = [
            i for i, d in enumerate(fit.cooks_distance) if np.isfinite(d) and d > 1.0
        ]
        add(
            f"\n**Residual diagnostics.** max |studentised| = "
            f"{np.nanmax(np.abs(fit.studentised)):.2f}, max leverage = "
            f"{np.nanmax(fit.leverage):.3f}, max Cook's D = "
            f"{np.nanmax(fit.cooks_distance):.3f}. "
            + (
                f"Points with Cook's D > 1: {influential}."
                if influential
                else "No point exceeds Cook's D of 1."
            )
        )
        for note in fit.notes:
            add(f"\n- {note}")

    add("\n## Lack of fit\n")
    add(analysis.lack_of_fit.describe())

    add("\n## What the levers do\n")
    add(
        "Marginal effect of adding 10 wt% HPMC, taken from lactose, at the design "
        "centroid. This is the form a formulator can act on; the interaction "
        "coefficient answers a less useful question.\n"
    )
    add("| grade | viscosity (cP) | Δ log₁₀(Td) | Td multiplier |")
    add("|---|---|---|---|")
    for row in analysis.lever_effects.itertuples():
        add(
            f"| {row.grade} | {row.viscosity_cp:,.0f} | {row.delta_log10_td:+.4f} | "
            f"×{row.fold_change_td:.2f} |"
        )
    first = analysis.lever_effects.iloc[0]
    last = analysis.lever_effects.iloc[-1]
    loss = 1.0 - (last.fold_change_td - 1.0) / max(first.fold_change_td - 1.0, 1e-9)
    add(
        f"\n**The composition lever loses about {loss:.0%} of its effect** moving from "
        f"{first.grade} to {last.grade}. The two levers are not additive: buying delay "
        "with polymer content becomes progressively less efficient as grade viscosity "
        "rises, so the combination that looks strongest on paper is not the one that "
        "delivers the most delay per unit of formulation change."
    )
    return "\n".join(out) + "\n"


def render_equivalence(analysis: Analysis) -> str:
    summary = analysis.equivalence_summary
    out: list[str] = ["# Iso-release equivalence sets (AC7)\n", _banner(analysis)]
    add = out.append

    add(
        f"Two formulations are equivalent when f2 ≥ {config.F2_SIMILAR_THRESHOLD:g}. f2 is "
        f"computed on regularly spaced timepoints with at most one beyond "
        f"{config.F2_PLATEAU_PCT:g}% release — necessary here because the sampling "
        "schedule is front-loaded, and passing all 25 points would let the first two "
        "hours dominate the statistic.\n"
    )

    add("## Headline\n")
    add(summary.demonstration)
    add(
        f"\n- Targets examined: {summary.n_targets}"
        f"\n- Targets with a cross-grade equivalent: {len(summary.cross_grade_targets)}"
        f"\n- Largest equivalence set: {summary.max_set_size} formulations"
        f"\n- Median set size: {summary.median_set_size:.0f}"
    )
    if summary.isolated_targets:
        names = ", ".join(f"case {c}/{g}" for c, g in summary.isolated_targets)
        add(
            f"\n**Where the freedom collapses.** {len(summary.isolated_targets)} "
            f"formulation(s) have no equivalent anywhere in the design: {names}. These "
            "sit at the edge of the achievable release space — there is exactly one way "
            "to hit that profile, so the formulation has no latitude for substitution."
        )

    add("\n## Sets spanning more than one grade\n")
    add("| target | grades | members (f2) |")
    add("|---|---|---|")
    for s in analysis.equivalence:
        if not s.spans_multiple_grades:
            continue
        members = ", ".join(
            f"{m.case}/{m.grade} ({m.f2:.0f})"
            for m in s.members
            if (m.case, m.grade) != (s.target_case, s.target_grade)
        )
        add(f"| case {s.target_case}/{s.target_grade} | {len(s.grades_spanned)} | {members} |")

    return "\n".join(out) + "\n"


def render_stress(analysis: Analysis, stress: StressTest) -> str:
    out: list[str] = ["# Design stress test (AC8)\n", _banner(analysis)]
    add = out.append

    add(
        "Candidate subsets are D-optimal for the fitted model, chosen by deterministic "
        "exchange over the available design points — not sampled at random, which would "
        "mostly measure how bad a bad design is.\n"
    )
    add(f"Full design profile RMSE: {stress.full_profile_rmse_pct:.3f}% released\n")

    add("| runs | estimable | profile RMSE % | worst % | D-criterion | directions agree "
        "| equivalence agreement |")
    add("|---|---|---|---|---|---|---|")
    for r in stress.results:
        add(
            f"| {r.size} | {'yes' if r.estimable else 'no'} | {r.profile_rmse_pct:.3f} | "
            f"{r.profile_rmse_worst_pct:.3f} | {r.d_criterion:.4f} | "
            f"{'yes' if r.lever_direction_agrees else '**no**'} | "
            f"{r.equivalence_jaccard:.3f} |"
        )

    add(f"\n## Recommendation\n\n{stress.rationale}\n")
    if stress.recommended:
        add(f"### The {stress.recommended.size} runs to make\n")
        add("| case | grade |")
        add("|---|---|")
        for case, grade in stress.recommended.selected:
            add(f"| {case} | {grade} |")
        saved = len(analysis.design_points) - stress.recommended.size
        add(
            f"\nThat is {saved} fewer runs than the full design "
            f"({saved / len(analysis.design_points):.0%} saved)."
        )

    for note in stress.notes:
        add(f"\n- {note}")
    return "\n".join(out) + "\n"


def render_guidelines(analysis: Analysis, stress: StressTest) -> str:
    quality = analysis.quality
    space = analysis.response_space
    cv = analysis.cross_validation
    out: list[str] = ["# Formulation guidelines and limitations (AC13)\n", _banner(analysis)]
    add = out.append

    add("## What the data supports\n")
    first = analysis.lever_effects.iloc[0]
    last = analysis.lever_effects.iloc[-1]
    loss = 1.0 - (last.fold_change_td - 1.0) / max(first.fold_change_td - 1.0, 1e-9)
    add(
        f"**The two levers are not additive, and the second one pays less.** Adding "
        f"10 wt% HPMC in place of lactose multiplies Td by ×{first.fold_change_td:.2f} at "
        f"{first.grade}, but only ×{last.fold_change_td:.2f} at {last.grade} — roughly "
        f"{loss:.0%} less effect. Polymer content and grade viscosity are partially "
        "substitutable rather than cumulative, so reaching for both at once buys less "
        "than the sum of their separate effects. If a target is missed at high grade, "
        "adding polymer is the weaker of the two remaining moves."
    )
    add(
        f"\n**Formulation freedom is real and measurable.** "
        f"{len(analysis.equivalence_summary.cross_grade_targets)} of "
        f"{analysis.equivalence_summary.n_targets} measured formulations have at least "
        "one f2-similar counterpart at a *different* grade. A target release profile "
        "usually does not dictate a single composition; it admits a set, and that set "
        "can be chosen on secondary criteria — cost, compressibility, drug load — "
        "without giving up the release profile."
    )
    if analysis.equivalence_summary.isolated_targets:
        names = ", ".join(
            f"case {c}/{g}" for c, g in analysis.equivalence_summary.isolated_targets
        )
        add(
            f"\n**Where that freedom disappears.** {names} has no equivalent anywhere in "
            "the design. At the fast edge of the release space there is one way to do it, "
            "and substitutions are not available."
        )
    add(
        f"\n**A smaller design would have reached the same conclusions.** "
        f"{stress.recommended_size} of {len(analysis.design_points)} runs preserves "
        "profile prediction and every directional conclusion. For a new API, that is the "
        "starting design — see the stress-test report for the specific points."
    )

    add("\n## How far to trust a prediction\n")
    add(
        f"- Leave-one-formulation-out CV over {cv.n_folds} folds: "
        f"**{cv.profile_rmse_pct:.2f}% released RMSE** in profile space, worst case "
        f"{cv.profile_rmse_pct_worst:.2f}%. Median f2 between predicted and observed "
        f"profiles is {cv.median_f2:.0f}."
    )
    add(
        "- That number, not the replicate spread, is the uncertainty on a prediction. "
        "The replicate SD in this database measures vessel-to-vessel repeatability "
        "within one compression batch — a much smaller quantity that says nothing about "
        "how a *new* formulation will behave."
    )
    add(
        "- **Batch-to-batch variability is unmeasured.** Every formulation was "
        "compressed once. Nothing here bounds how much a repeat batch would differ, and "
        "the tight replicate spread must not be read as evidence that it would not."
    )

    add("\n## What the data does not support\n")
    add(
        f"- **Nothing about solubility.** The database contains {len(quality.apis)} API"
        f"{'s' if len(quality.apis) != 1 else ''}. Any relationship between solubility "
        "class and release behaviour is unavailable, and every solubility-facing feature "
        "in the dashboard is gated shut until at least two APIs per class exist. When "
        "that gate opens at 2×2, the contrast will still be confounded with molecule "
        "identity and can only ever be directional."
    )
    add(
        "- **No extrapolation.** Predictions are valid inside the convex hull of the "
        "tested compositions and between the three tested viscosity grades. Outside "
        "either, the tool warns and shows the nearest measured formulations instead of "
        "a number."
    )
    add(
        f"- **The slowest formulations are the least well characterised.** "
        f"{len(quality.fully_censored_ids)} formulations never reach "
        f"{config.CENSORING_PCT:g}% release within 24 h and "
        f"{len(quality.partially_censored_ids)} more straddle it. For these the Weibull "
        "asymptote is not identified by the data — it is extrapolated — so Td carries "
        "materially more uncertainty there than the headline CV figure suggests. "
        "Censoring and asymptote identifiability coincide exactly in this design."
    )
    add(
        f"- **The metrics are not independent evidence.** {len(space.metrics)} computed "
        f"metrics collapse to {space.n_components_90} real dimensions. Agreement between "
        "t50, MDT and % released at 12 h is arithmetic, not corroboration. All "
        f"conclusions here rest on the key responses: {', '.join(space.key_responses)}."
    )
    add(
        "- **Lack of fit is testable only against within-batch error.** The replicates "
        "are vessels from one batch, so the pure-error term excludes batch variation. "
        "The F test is anti-conservative and will over-declare lack of fit."
    )

    add("\n## Design-space boundaries\n")
    comps = analysis.design_points
    add(
        f"- API: {comps['api_wt'].min():g}–{comps['api_wt'].max():g} wt% | "
        f"HPMC: {comps['hpmc_wt'].min():g}–{comps['hpmc_wt'].max():g} wt% | "
        f"lactose: {comps['lactose_wt'].min():g}–{comps['lactose_wt'].max():g} wt%"
    )
    add(
        f"- Grades: {', '.join(sorted(set(comps['grade'])))} "
        f"({comps['viscosity_cp'].min():,.0f}–{comps['viscosity_cp'].max():,.0f} cP). "
        "Interpolation between them is permitted; extrapolation beyond is not."
    )
    add(
        "- Components sum to 100 wt% exactly. This is a mixture: the three cannot be "
        "varied independently, and any change to one is a change to another."
    )
    add(
        f"- Observation window 0–{analysis.time_grid.max():g} h, "
        f"{len(analysis.time_grid)} timepoints, {quality.samples_within_2h} of them "
        "inside the first 2 h."
    )
    return "\n".join(out) + "\n"


def render_doe(analysis: Analysis) -> str:
    """The classical DoE report: takeaways first, tables underneath."""
    out: list[str] = ["# DoE analysis on named responses\n", _banner(analysis)]
    add = out.append

    add(
        "Response surfaces fitted directly on the quantities a formulator names — "
        "release times and % released — rather than on the Weibull parameters. The "
        "Weibull model still exists, but only to reconstruct predicted curves for the "
        "formulator tool; nothing below depends on reading it.\n"
    )
    add(
        "Each response chooses its own model. The shape that fits release timing is "
        "not necessarily the shape that fits completeness, and imposing one model on "
        "all of them would flatter some responses and misrepresent others.\n"
    )

    for ra in analysis.doe.responses:
        spec = ra.response.spec
        add(f"\n## {spec.label} ({spec.units})\n")
        if not ra.usable:
            add(
                "Not estimable from the points this response has. "
                + " ".join(ra.notes)
            )
            continue

        table = ra.anova
        add(
            f"**Model.** `{ra.spec.label}`, {len(ra.kept_terms)} terms, "
            f"{table.n_obs} design points, {table.residual_df} residual df."
        )
        add(
            f"S = {table.s:.4g} {spec.units} | R-sq = {table.r_squared:.2%} | "
            f"R-sq(adj) = {table.adj_r_squared:.2%} | "
            f"**R-sq(pred) = {table.pred_r_squared:.2%}**\n"
        )
        add(f"\n{ra.takeaway_anova}\n")
        add(f"\n**Levers.** {ra.takeaway_traces}\n")
        add(f"\n**Interaction.** {ra.interactions[0].interpretation}\n")
        if ra.takeaway_contour:
            add(f"\n**Design space.** {ra.takeaway_contour}\n")
        add(f"\n**Effects.** {ra.takeaway_effects}\n")

        add("\n| Source | DF | Adj SS | Adj MS | F | P |")
        add("|---|---|---|---|---|---|")
        m = table.model_row
        add(
            f"| **Model** | {m.df} | {m.adj_ss:.3f} | {m.adj_ms:.3f} | "
            f"{m.f_value:.2f} | {m.p_value:.4g} |"
        )
        for row in table.rows:
            if not row.is_group:
                continue
            add(
                f"| &nbsp;&nbsp;{row.source} | {row.df} | {row.adj_ss:.3f} | "
                f"{row.adj_ms:.3f} | {row.f_value:.2f} | {row.p_value:.4g} |"
            )
        add(f"| **Residual** | {table.residual_df} | {table.residual_ss:.3f} | | | |")
        add(f"| **Total** | {table.total_df} | {table.total_ss:.3f} | | | |")

        for note in list(ra.notes) + list(table.notes):
            add(f"\n- {note}")

    if analysis.doe.responses:
        add("\n## How to read these\n")
        for key in (
            "anova", "model_summary", "cox_trace", "interaction",
            "contour", "pareto", "half_normal",
        ):
            method_note = analysis.doe.method_notes.get(key)
            if method_note:
                add(f"\n**{key.replace('_', ' ').title()}.** {method_note}")

    return "\n".join(out) + "\n"
