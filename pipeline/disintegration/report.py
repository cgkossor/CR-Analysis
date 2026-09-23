"""The disintegration report: findings first, evidence underneath.

Every sentence that states a finding is built from the computed numbers, with
wording that changes with them. Real data that disagrees with the synthetic
story produces a report that says so, not one that repeats the story.
"""

from __future__ import annotations

import numpy as np

from pipeline.disintegration.analysis import DisintegrationAnalysis
from pipeline.disintegration.grade_effects import BlockModel


def _f(x: float, fmt: str = ".2f") -> str:
    return format(x, fmt) if np.isfinite(x) else "n/a"


def _p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def correlation_sentence(r: DisintegrationAnalysis) -> str:
    c = r.correlation
    ranked = c.ranked
    if not ranked:
        return "Too few matched formulations to correlate disintegration with dissolution."
    best = ranked[0]
    td = c.by_key("td_h")
    within = [w.pearson for w in c.within_grade if np.isfinite(w.pearson)]
    parts = [
        f"Across {best.n} uncensored formulations, disintegration time tracks "
        f"{best.metric.label} most closely (Spearman ρ = {best.spearman:+.2f}, 95% CI "
        f"{_f(best.spearman_ci[0])} to {_f(best.spearman_ci[1])})."
    ]
    if td is not None and within:
        lo, hi = min(within), max(within)
        if lo > abs(td.pearson) + 0.05:
            parts.append(
                f"Against the Weibull time scale Td the pooled correlation is r = "
                f"{td.pearson:.2f}, but *within* each grade it is {lo:.2f}–{hi:.2f}. The "
                "grades sit on separate lines, and pooling them understates how tightly "
                "DT follows dissolution."
            )
        else:
            parts.append(
                f"Against Td, r = {td.pearson:.2f} pooled and {lo:.2f}–{hi:.2f} within "
                "grades, so one line describes all grades about equally well."
            )
    return " ".join(parts)


def grade_sentence(r: DisintegrationAnalysis) -> str:
    g = r.grades
    bd, bt = g.block_dt, g.block_td
    if bd is None or bt is None:
        return "Not enough complete composition blocks to compare grades at matched composition."
    lo, hi = r.grade_order[0], r.grade_order[-1]
    ratio_dt = float(np.exp(bd.effects.get(hi, np.nan) - bd.effects.get(lo, np.nan)))
    ratio_td = float(np.exp(bt.effects.get(hi, np.nan) - bt.effects.get(lo, np.nan)))
    comp = g.compression
    if not np.isfinite(comp):
        verdict = ""
    elif comp < 0.8:
        verdict = (
            f"Grade separates disintegration far *less* than it separates release. The DT "
            f"grade span is {comp:.2f}× the dissolution span, so a DT specification would "
            "under-rank the high-viscosity grades' release retardation."
        )
    elif comp > 1.25:
        verdict = (
            f"Grade separates disintegration *more* than release (span ratio {comp:.2f}). "
            "The gel's survival is more grade-sensitive than its drug release."
        )
    else:
        verdict = (
            f"Grade moves disintegration and release by similar amounts (span ratio {comp:.2f})."
        )
    return (
        f"At identical composition ({g.n_complete_blocks} complete blocks), {hi} tablets "
        f"take ×{_f(ratio_dt)} as long to disintegrate as {lo}, while their dissolution "
        f"time scale is ×{_f(ratio_td)}. {verdict}"
    )


def prediction_sentence(r: DisintegrationAnalysis) -> str:
    by = {m.key: m for m in r.loo}
    rec, dis, both = by.get("recipe"), by.get("dissolution"), by.get("recipe_dissolution")
    if rec is None or dis is None:
        return ""
    s = (
        f"Predicting DT for a formulation left out of the fit, the recipe alone gives "
        f"Q² = {rec.q2:.2f} (typical error ×{rec.fold_error:.2f}) and one dissolution "
        f"number (Td) alone gives Q² = {dis.q2:.2f}."
    )
    if both is not None:
        gain = both.q2 - rec.q2
        s += (
            f" Adding Td to the recipe {'improves' if gain > 0.01 else 'barely changes'} "
            f"this (Q² = {both.q2:.2f})"
            + (". Dissolution carries information about DT that the recipe does not."
               if gain > 0.01 else ". Dissolution adds little beyond what the recipe already "
               "says.")
        )
    return s


def _block_table(dt: BlockModel, td: BlockModel) -> list[str]:
    rows = ["| Contrast | DT ratio (95% Tukey CI) | p | Td ratio (95% Tukey CI) | p |",
            "|---|---|---|---|---|"]
    tdc = {(c.a, c.b): c for c in td.contrasts}
    for c in dt.contrasts:
        t = tdc.get((c.a, c.b))
        tail = (f"×{t.ratio:.2f} ({t.ci[0]:.2f}–{t.ci[1]:.2f}) | {_p(t.p_adj)}"
                if t else "n/a | n/a")
        rows.append(f"| {c.a} / {c.b} | ×{c.ratio:.2f} ({c.ci[0]:.2f}–{c.ci[1]:.2f}) | "
                    f"{_p(c.p_adj)} | {tail} |")
    return rows


def render(r: DisintegrationAnalysis, figures: list[tuple[str, str]] | None = None) -> str:
    out: list[str] = ["# Disintegration vs dissolution and formulation\n"]
    add = out.append
    if r.is_synthetic:
        add("> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.** No value here is a "
            "measurement. The synthetic generator and its true parameters are in the "
            "workbook's `Disintegration_Notes` sheet.\n")

    add("## Findings\n")
    add(f"1. **Correlation.** {correlation_sentence(r)}")
    add(f"2. **Grade at matched composition.** {grade_sentence(r)}")
    pred = prediction_sentence(r)
    if pred:
        add(f"3. **What predicts DT.** {pred}")
    a = r.grades.ancova
    if a is not None:
        add(
            f"4. **One line or several.** The grades share a slope on ln Td "
            f"(p = {_p(a.p_common_slope)} for different slopes) "
            if a.selected == "common slope" else
            f"4. **One line or several.** The grades need their own slopes on ln Td "
            f"(p = {_p(a.p_common_slope)}) "
        )
        out[-1] += (
            f"and differ in offset (p = {_p(a.p_common_offset)}). Pooled slope "
            f"{a.pooled_slope:.2f} against shared within-grade slope {a.shared_slope:.2f}: "
            + ("the pooled line is flattened by the grade offsets, a Simpson-type effect."
               if a.shared_slope - a.pooled_slope > 0.2 else "the pooled line is not distorted.")
        )
    cens = r.censored_points
    if len(cens):
        add(f"\n*{len(cens)} of {len(r.matched)} formulations were still intact at the "
            f"{r.data.test_end_h:g} h test end. They are lower bounds and are excluded from "
            "every model, so conclusions describe the faster-eroding part of the design.*")

    # --- Precision --------------------------------------------------------
    p = r.precision
    add("\n## Replicate precision\n")
    add(f"ICC(1) = {_f(p.icc, '.3f')}: the share of total ln-DT variance that is between "
        "formulations rather than between tablets of one formulation.")
    add(f"Brown–Forsythe test for equal scatter across grades: p = {_p(p.brown_forsythe_p)}.\n")
    add("| Grade | Formulations | Pooled CV | Median CV |")
    add("|---|---|---|---|")
    for gp in p.by_grade:
        add(f"| {gp.grade} | {gp.n_formulations} | {gp.pooled_cv:.1%} | {gp.median_cv:.1%} |")
    if p.flags:
        add("\nDixon's Q flags (reported, not removed):\n")
        for fl in p.flags:
            add(f"- case {fl.case}/{fl.grade}, replicate {fl.replicate}: {fl.dt_h:.2f} h "
                f"(Q = {fl.q:.2f} > {fl.q_critical:.3f})")

    # --- Correlation ------------------------------------------------------
    c = r.correlation
    add("\n## Correlation with the dissolution responses\n")
    add("Design-point means. Pearson is on ln DT against ln(metric) for time-like metrics "
        "and against the raw metric for percentages. 95% intervals from 2,000 "
        "formulation-level bootstrap resamples.\n")
    add("| Metric | n | Pearson r (95% CI) | Spearman ρ (95% CI) | Expected sign |")
    add("|---|---|---|---|---|")
    for x in c.ranked:
        exp = {1: "+", -1: "−", 0: "—"}[x.metric.expected_sign]
        ok = "" if x.sign_as_expected in (None, True) else " ✗"
        add(f"| {x.metric.label} | {x.n} | {x.pearson:+.2f} ({_f(x.pearson_ci[0])}, "
            f"{_f(x.pearson_ci[1])}) | {x.spearman:+.2f} ({_f(x.spearman_ci[0])}, "
            f"{_f(x.spearman_ci[1])}) | {exp}{ok} |")
    add("\n**Deming regression of ln DT on ln Td**. This allows for replicate error on "
        "both axes, with the error ratio estimated from the replicates.\n")
    add("| Fit | n | Deming slope (95% CI) | OLS slope | Error ratio |")
    add("|---|---|---|---|---|")
    for dm in c.deming:
        add(f"| {dm.label} | {dm.n} | {dm.slope:.2f} ({_f(dm.slope_ci[0])}, "
            f"{_f(dm.slope_ci[1])}) | {dm.ols_slope:.2f} | {dm.error_ratio:.2f} |")
    pc = c.partial
    add(f"\nPartial correlation of ln DT with ln Td after removing "
        f"{', '.join(pc.controls)}: r = {_f(pc.r)} (p = {_p(pc.p)}, n = {pc.n}).")

    # --- Grade ------------------------------------------------------------
    g = r.grades
    add("\n## Grade effects at matched composition\n")
    if a is not None:
        add(f"**ANCOVA** of ln DT on ln Td (centred at {a.centre_ln_td:.2f}) by grade, "
            f"{a.n} points, R² = {a.r2:.3f}, selected model: {a.selected}.\n")
        add("| Grade | n | ln DT at mean ln Td (95% CI) | Slope (95% CI) |")
        add("|---|---|---|---|")
        for ln in a.lines:
            add(f"| {ln.grade} | {ln.n} | {ln.offset:.2f} ({ln.offset_ci[0]:.2f}, "
                f"{ln.offset_ci[1]:.2f}) | {ln.slope:.2f} ({ln.slope_ci[0]:.2f}, "
                f"{ln.slope_ci[1]:.2f}) |")
    if g.block_dt is not None and g.block_td is not None:
        add(f"\n**Randomised-block model** `ln y ~ grade + case`: {g.block_dt.n} points in "
            f"{g.block_dt.n_blocks} composition blocks. Grade p = {_p(g.block_dt.p_grade)} "
            f"for DT and {_p(g.block_td.p_grade)} for Td. Ratios are at identical "
            "composition, with Tukey-adjusted intervals.\n")
        out += _block_table(g.block_dt, g.block_td)
        add(f"\nSpan of grade effects, DT over Td: **{_f(g.compression)}**. Median "
            f"per-case ratio of the spread across grades, SD(ln DT)/SD(ln Td): "
            f"**{_f(g.divergence_index)}** over {g.n_complete_blocks} complete cases.")
    if g.lag:
        add("\n**Erosion lag** R = DT / Td by grade:\n")
        add("| Grade | n | Median R | Change in ln R per +10 wt% HPMC (95% CI) |")
        add("|---|---|---|---|")
        for lt in g.lag:
            add(f"| {lt.grade} | {lt.n} | {lt.median_ratio:.2f} | {lt.slope_per_10wt:+.3f} "
                f"({lt.ci[0]:+.3f}, {lt.ci[1]:+.3f}) |")
        add(f"\nln R against the Peppas exponent n: Spearman ρ = {_f(g.lag_vs_peppas_rho)} "
            f"(p = {_p(g.lag_vs_peppas_p)}).")

    # --- DoE --------------------------------------------------------------
    add("\n## DoE on disintegration time\n")
    add("The same Scheffé mixture × log-viscosity fit, model reduction and Type III ANOVA "
        "as the dissolution responses (`doe_analysis.md`), applied to DT.\n")
    for ra in r.doe.responses:
        spec = ra.response.spec
        add(f"\n### {spec.label} ({spec.units})\n")
        if not ra.usable:
            add("Not estimable. " + " ".join(ra.notes))
            continue
        t = ra.anova
        add(f"`{ra.spec.label}`, {len(ra.kept_terms)} terms, {t.n_obs} points, "
            f"{t.residual_df} residual df. R² = {t.r_squared:.2%}, R²(adj) = "
            f"{t.adj_r_squared:.2%}, **R²(pred) = {t.pred_r_squared:.2%}**.\n")
        add(f"{ra.takeaway_anova}\n")
        add(f"**Levers.** {ra.takeaway_traces}\n")
        add(f"**Interaction.** {ra.interactions[0].interpretation}\n")
        add(f"**Effects.** {ra.takeaway_effects}\n")
        add("| Source | DF | Adj SS | F | P |")
        add("|---|---|---|---|---|")
        for row in t.rows:
            if row.is_group:
                add(f"| {row.source} | {row.df} | {row.adj_ss:.3f} | {row.f_value:.2f} | "
                    f"{_p(row.p_value)} |")
        add(f"| Residual | {t.residual_df} | {t.residual_ss:.3f} | | |")
        for note in ra.notes:
            add(f"\n- {note}")

    # --- Prediction -------------------------------------------------------
    if r.loo:
        add("\n## What predicts disintegration\n")
        add("Leave-one-formulation-out on ln DT, with all models evaluated on the same "
            "formulations.\n")
        add("| Model | Parameters | Q² | RMSE (ln) | Typical error |")
        add("|---|---|---|---|---|")
        for m in r.loo:
            add(f"| {m.label} | {m.n_params} | {m.q2:.3f} | {m.rmse_ln:.3f} | "
                f"×{m.fold_error:.2f} |")

    if r.recovery:
        add("\n## Synthetic recovery check\n")
        add("The generator's structural model refitted to its own output. Every true "
            "value should lie inside its interval.\n")
        add("| Parameter | Truth | Estimate | 95% CI | Covered |")
        add("|---|---|---|---|---|")
        for rec in r.recovery:
            add(f"| {rec.name} | {rec.truth:g} | {rec.estimate:.3f} | {rec.ci[0]:.3f} to "
                f"{rec.ci[1]:.3f} | {'yes' if rec.covered else '**no**'} |")

    add("\n## Caveats\n")
    add("- Correlation is between formulation means. It says nothing about whether one "
        "tablet's DT predicts the same tablet's release.")
    add("- Td is used as the dissolution covariate because it is defined for every "
        "formulation. t80 is censored for most high-viscosity ones, and a covariate "
        "missing on one grade would bias the grade comparison.")
    add("- DT is measured with mechanical agitation (basket and discs), dissolution "
        "without. A gap between the two across grades is partly a statement about the "
        "tests, not only about the tablets.")
    if figures:
        add("\n## Figures\n")
        for fid, caption in figures:
            add(f"- **{fid}.** {caption}")
        add("\nRecommended headline candidate for the deck: **DT-03** (same composition, "
            "different grade).")
    # _p() renders small values as "< 0.001"; drop the "=" that precedes it in prose.
    return "\n".join(out).replace("p = < ", "p < ") + "\n"
