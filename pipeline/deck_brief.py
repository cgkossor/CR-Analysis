"""An instruction file for building the results deck in another tool.

The manuscript figures are drawn here, but the slide deck may be assembled
elsewhere, by a person or by an AI agent working where the real data live.
``deck_brief.md`` gives that builder everything it needs and nothing it has
to guess: the design system (fonts, colours, sizes and layouts in PowerPoint
units), then one entry per slide with its title, layout, the figure file to
place, and the slide text, written from this run's results in publication
style (past tense, no first person, hedged where the evidence is only
directional).

The numbers are read from the analysis, never typed in, so the brief is
regenerated with every run and always matches the figures beside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline import config
from pipeline.figures.publication import FigureRecord

if TYPE_CHECKING:
    from pipeline.paper import PaperInput

#: The deck's look, matched to the reference deck. Sizes are PowerPoint points
#: on a 13.33 x 7.5 in (16:9) slide.
DESIGN = """\
## Design specification

Build a 16:9 deck (13.33 x 7.5 in). Keep every slide to the layouts below; do not
add decoration, stock images, icons or animations beyond what is listed.

**Typefaces.** Headings: Source Serif 4, semibold (fallback Georgia). Body, labels,
tables: IBM Plex Sans (fallback Arial).

**Type sizes (pt).** Cover title 48. Slide title 32 (28 on figure slides). Card
heading 18. Body and bullets 14. Eyebrow labels 12, semibold, letter-spaced, upper
case. Footer 12. Big numbers on the results slide 48.

**Colours (hex).**
- Navy (dark backgrounds, headings): #14213D
- Paper (light background): #F6F4EE
- Card fill: #FBFAF6, card border: #DCD8CE (1 pt), corner radius about 0.1 in
- Body text: #1F2A3D; secondary text: #4A5568; footer text: #6B7280
- Accent (eyebrows, links): #0072B2; light accent on navy: #7FB8E0
- Warning text (placeholder notices): #E8A27A on navy, #B4541F on paper
- Table header row: #E9E5DA; banded rows: #FBFAF6

**Margins.** 0.67 in on all sides; the footer sits on one line 0.33 in above the
bottom edge, left-aligned.

**Layouts.**
- **Title**: navy background. Eyebrow line in light accent, title in paper colour,
  one-line subtitle in #C9D3E3, warning footer.
- **Cards**: paper background, slide title, then 2 to 4 cards in a row or a 2 x 2
  grid. Each card: eyebrow or card heading, then 1 to 3 short sentences.
- **Table**: paper background, slide title, one table with a header row and banded
  rows, one sentence below it.
- **Figure, side**: paper background, slide title. Left: the figure on a white
  card (about 60 % of the width, aspect ratio preserved, never stretched or
  cropped). Right: eyebrow "WHAT IT SHOWS" with 2 to 3 bullets, then eyebrow
  "WHY IT MATTERS" with one or two sentences.
- **Figure, wide**: for figures much wider than tall. The figure spans the full
  width on a white card; below it two text columns, "WHAT IT SHOWS" and
  "WHY IT MATTERS".
- **Statement**: navy background, slide title, 3 to 4 big-number cards (number in
  light accent, 48 pt; one sentence under each in #C9D3E3), one sentence below.

**Footer.** Every slide except the cover carries the footer text given for it.
"""

RULES = """\
## Rules for the builder

1. Use the slide text exactly as written. Do not add, round, or reword numbers.
2. Place each figure from the named PNG file in this folder. Keep its aspect
   ratio; never crop, recolour or redraw a figure.
3. Speaker notes are given where useful; put them in the notes pane, not on the slide.
4. If a figure file listed here is missing, leave a labelled empty frame and say so;
   never substitute another figure.
5. Text in square brackets is a placeholder for a method detail not held in the
   data (for example the dissolution apparatus); keep the brackets until it is filled.
"""


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _api_list(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _hedge(status: str, strong: str, soft: str) -> str:
    """Wording by evidence: firm when supported, tentative when directional."""
    return strong if status == "supported" else soft


def _lever_sentence(p: PaperInput) -> tuple[str, str]:
    decs = [d for d in p.manuscript.decompositions if d.key != "weibull_beta"]
    if not decs:
        return "", "not_supported"
    c = float(np.mean([d.share_composition for d in decs]))
    g = float(np.mean([d.share_grade for d in decs]))
    i = float(np.mean([d.share_interaction for d in decs]))
    status = p.manuscript.questions[0].status
    lead = "HPMC grade" if g >= c else "composition"
    verb = _hedge(status, "accounted for", "appeared to account for")
    return (
        f"For {p.api}, {lead} {verb} the larger share of the variation in release "
        f"(grade {_pct(g)}, composition {_pct(c)}, interaction {_pct(i)}; mean over "
        f"{len(decs)} release responses)."
    ), status


def _swap_sentence(p: PaperInput) -> str:
    sw = [s for s in p.manuscript.swaps if s.valid]
    ok = [s for s in sw if s.similar]
    if not sw:
        return f"For {p.api}, no grade pair could be compared by f2."
    pairs = sorted({f"{s.grade_a} and {s.grade_b}" for s in ok})
    where = (f", all between {_api_list(pairs)}" if pairs else "")
    return (f"For {p.api}, {len(ok)} of {len(sw)} same-blend grade comparisons met "
            f"f2 ≥ {config.F2_SIMILAR_THRESHOLD:g}{where}.")


def _beta_range(p: PaperInput) -> str:
    dp = p.analysis.design_points
    if "weibull_beta_mean" not in dp:
        return ""
    b = dp["weibull_beta_mean"].to_numpy(dtype=float)
    b = b[np.isfinite(b)]
    if not b.size:
        return ""
    return f"{p.api}: β {b.min():.2f} to {b.max():.2f}"


def _fit_sentence(p: PaperInput, key: str) -> str:
    ra = p.analysis.doe.by_key(key)
    if ra is None or not ra.usable:
        return ""
    return (f"{p.api}: R² {ra.fit.r_squared:.2f}, predicted R² (leave-one-out) "
            f"{ra.fit.pred_r_squared:.2f}")


def _dt_sentence(p: PaperInput) -> str:
    q = p.manuscript.questions[3]
    if q.status == "unavailable" or p.disintegration is None:
        return f"No disintegration data were available for {p.api}."
    fit = next((f for f in p.disintegration.correlation.deming if f.label == "all grades"),
               None)
    if fit is None:
        return f"For {p.api}, the pooled disintegration fit could not be estimated."
    verb = _hedge(q.status, "increased", "tended to increase")
    lo, hi = fit.slope_ci
    return (f"For {p.api}, disintegration time {verb} with the dissolution time scale "
            f"(Deming slope {fit.slope:.2f}, 95% CI {lo:.2f} to {hi:.2f}, "
            f"n = {fit.n}).")


def _stress_sentence(p: PaperInput) -> str:
    rec = p.stress.recommended
    n = len(p.analysis.design_points)
    if rec is None:
        return f"For {p.api}, no reduced design met the recommendation criteria."
    return (f"For {p.api}, {rec.size} of {n} runs reproduced the full-design "
            f"conclusions (profile error {rec.profile_rmse_pct:.2f}% against "
            f"{p.stress.full_profile_rmse_pct:.2f}% for all runs).")


#: One line per figure on why it matters; the full background goes in the notes.
WHY_SHORT: dict[str, str] = {
    "Fig2": "The measured profiles are the primary evidence: they show the size of each "
            "effect and the replicate precision before any model is applied.",
    "Fig3": "The dominant lever is the one to adjust first; a large interaction means "
            "rules from one grade do not carry over to another.",
    "Fig4": "A surface that predicts formulations it was not fitted to is the basis for "
            "a formulation design space.",
    "Fig5": "The release mechanism governs robustness: erosion-influenced release tends "
            "to be more sensitive to hydrodynamic conditions than diffusion-controlled "
            "release.",
    "Fig6": "A change of HPMC grade is a post-approval change; f2 ≥ 50 is the "
            "regulatory similarity criterion.",
    "Fig7": "Disintegration testing is fast; if it tracks dissolution it can serve as a "
            "screening test.",
    "Fig8": "The smallest adequate design sets the cost of extending the study to "
            "further APIs.",
}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(x[ok], y[ok]).statistic)


def _grade_trend(p: PaperInput) -> str:
    """Mean MDT by grade, in viscosity order, as measured."""
    dp = p.analysis.design_points
    if "mdt_h_mean" not in dp:
        return ""
    by = dp.groupby("grade").agg(v=("viscosity_cp", "first"), m=("mdt_h_mean", "mean"))
    by = by.sort_values("v")
    if len(by) < 2:
        return ""
    vals = by["m"].to_numpy(dtype=float)
    rising = bool(np.all(np.diff(vals) > 0))
    parts = ", ".join(f"{g} {m:.1f} h" for g, m in zip(by.index, vals, strict=True))
    return (f"{p.api}: mean dissolution time "
            f"{'increased with viscosity grade' if rising else 'did not rise steadily with grade'}"
            f" ({parts}).")


def _hpmc_trend(p: PaperInput) -> str:
    """Within-grade association of HPMC level with mean dissolution time."""
    dp = p.analysis.design_points
    if "mdt_h_mean" not in dp:
        return ""
    rhos = [
        _spearman(g["hpmc_wt"].to_numpy(dtype=float), g["mdt_h_mean"].to_numpy(dtype=float))
        for _, g in dp.groupby("grade")
    ]
    rhos = [r for r in rhos if np.isfinite(r)]
    if not rhos:
        return ""
    med = float(np.median(rhos))
    trend = ("lengthened with HPMC level" if med >= 0.5 else
             "shortened with HPMC level" if med <= -0.5 else
             "showed no consistent trend with HPMC level")
    return (f"{p.api}: within each grade, mean dissolution time {trend} "
            f"(median Spearman ρ {med:+.2f} across grades).")


def _beta_lever(p: PaperInput) -> str:
    d = next((d for d in p.manuscript.decompositions if d.key == "weibull_beta"), None)
    if d is None:
        return ""
    lead = "composition" if d.share_composition >= d.share_grade else "HPMC grade"
    return (f"{p.api}: curve shape (Weibull β) varied more with {lead} (composition "
            f"{_pct(d.share_composition)}, grade {_pct(d.share_grade)}).")


def _beta_trend(p: PaperInput) -> str:
    dp = p.analysis.design_points
    if "weibull_beta_mean" not in dp:
        return ""
    b = dp["weibull_beta_mean"].to_numpy(dtype=float)
    rh = _spearman(dp["hpmc_wt"].to_numpy(dtype=float), b)
    rv = _spearman(dp["log10_visc"].to_numpy(dtype=float), b)
    def word(r: float) -> str:
        if not np.isfinite(r):
            return "could not be assessed"
        return "decreased" if r <= -0.3 else "increased" if r >= 0.3 else "changed little"
    return (f"{p.api}: β {word(rh)} with HPMC level (ρ {rh:+.2f}) and {word(rv)} with "
            f"viscosity grade (ρ {rv:+.2f}).")


def _label(inputs: list[PaperInput], key: str | None) -> str:
    """The response's readable name, lower case, e.g. 'mean dissolution time'."""
    ra = inputs[0].analysis.doe.by_key(key) if key else None
    return ra.response.spec.label.lower() if ra is not None else "the headline response"


def _sampling(p: PaperInput) -> str:
    per = p.analysis.db.profiles.groupby(["id", "replicate"]).size()
    med = int(per.median()) if len(per) else 0
    kind = ("probe-logged" if med >= config.SPIKE_MIN_READINGS else "manually sampled")
    return f"{p.api}: {kind}, median {med} readings per vessel"


def build(inputs: list[PaperInput], records: Sequence[FigureRecord],
          headline_key: str | None) -> str:
    """The brief as Markdown."""
    files = {r.id: r.file for r in records}
    apis = [p.api for p in inputs]
    synthetic = any(p.analysis.quality.is_synthetic for p in inputs)
    a0 = inputs[0].analysis
    n_form = len(a0.design_points)
    n_blend = int(a0.design_points["case"].nunique())
    grades = list(dict.fromkeys(a0.design_points.sort_values("viscosity_cp")["grade"]))
    n_rep = int(a0.quality.n_replicates)
    footer = ("Placeholder data: synthetic, not experimental" if synthetic
              else "Results: " + _api_list(apis))

    slides: list[dict[str, Any]] = []

    def slide(title: str, layout: str, body: str, figure: str | None = None,
              notes: str = "", footer_text: str | None = None) -> None:
        slides.append({"title": title, "layout": layout, "body": body,
                       "figure": figure, "notes": notes,
                       "footer": footer_text if footer_text is not None else footer})

    slide(
        "Composition and HPMC grade as determinants of drug release from hydrophilic "
        "matrix tablets", "Title",
        f"Eyebrow: CONTROLLED-RELEASE MATRIX TABLETS · MIXTURE DESIGN\n\n"
        f"Subtitle: A mixture–process design across {len(apis)} API"
        f"{'s' if len(apis) != 1 else ''} ({_api_list(apis)})",
        footer_text=("Placeholder data: every value derives from a synthetic database."
                     if synthetic else ""),
    )
    slide(
        "Background", "Cards (3 across)",
        "Card 1, heading 'Gel layer': On hydration, HPMC at the tablet surface forms a "
        "viscous gel layer around the dry core.\n\n"
        "Card 2, heading 'Release pathways': Drug is released by diffusion through the gel "
        "and by erosion of the gel; their balance sets the rate and shape of the profile.\n\n"
        "Card 3, heading 'Formulation levers': Release is adjusted through the HPMC level "
        "and viscosity grade, with drug load and filler completing the blend.\n\n"
        "Sentence below the cards: Because the components sum to 100%, no component can be "
        "varied alone; the study was therefore designed as a mixture design crossed with "
        "HPMC grade.",
    )
    slide(
        "Research questions", "Table",
        "Table, columns 'Q' | 'Question' | 'Approach':\n"
        "- Q1 | Which lever controls release: composition or HPMC grade, and do they "
        "interact? | Variance decomposition; mixture models\n"
        "- Q2 | Under which conditions are HPMC grades interchangeable? | f2 similarity "
        "at each blend\n"
        "- Q3 | Which formulations are sensitive to hydrodynamic shear? | "
        "Disintegration-based proxy\n"
        "- Q4 | Does disintegration time reflect dissolution behaviour? | Deming "
        "regression; leave-one-out prediction\n"
        "- Q5 | How does API solubility affect release mechanism? | Cross-API models "
        "(requires two APIs per solubility class)",
    )
    slide(
        "Experimental design", "Figure, side", (
            "WHAT IT SHOWS:\n"
            f"- Three components (API, HPMC, lactose) in {n_blend} blends spanning the "
            "edges, interior and centroid of the constrained region\n"
            f"- Each blend prepared with {_api_list(grades)}: {n_form} formulations, "
            f"{n_rep} vessels each\n"
            "- The same blends were used for every API\n\n"
            "WHY IT MATTERS: The tested region defines where the conclusions apply; "
            "identical blends allow a direct comparison between APIs."),
        files.get("Fig1"),
        notes="Compositions are listed in Table1_design.csv in this folder.",
    )
    slide(
        "Release metrics", "Figure, side", (
            "WHAT IT SHOWS:\n"
            "- Times to 10, 25, 50 and 80% released, and % released at fixed times\n"
            "- Mean dissolution time and phase-wise release rates\n"
            "- Weibull plateau (F∞), time scale (Td) and shape (β)\n\n"
            "WHY IT MATTERS: Each result is stated in one of these metrics, and each "
            "describes a different part of the profile."),
        files.get("Scheme1"),
        notes="Schematic on an illustrative Weibull curve; not measured data.",
    )
    slide(
        "Dissolution testing and data processing", "Cards (2 x 2)",
        "Card 'Dissolution': [Apparatus, rotation speed and medium], "
        f"{config.VESSEL_VOLUME_ML:g} mL per vessel, {n_rep} vessels per formulation, to "
        f"{config.ANALYSIS_WINDOW_H:g} h ({'; '.join(_sampling(p) for p in inputs)}).\n\n"
        "Card 'Release calculation': Release was calculated from concentration and vessel "
        "volume relative to the dose (tablet mass × API fraction). Values were not "
        "rescaled.\n\n"
        "Card 'Data treatment': Readings beyond 24 h were excluded and momentary spikes "
        "removed against a local median; every exclusion was recorded.\n\n"
        "Card 'Disintegration': Disintegration was measured per tablet in a basket-and-"
        "disc apparatus; tablets intact at the end of the test were treated as censored.",
    )
    slide(
        "Statistical analysis", "Cards (4 across)",
        "Card '1 · MODEL': Scheffé mixture polynomials in API, HPMC and lactose, crossed "
        "with HPMC grade as coded log viscosity.\n\n"
        "Card '2 · DEGREE': Linear, quadratic or special cubic, selected by sequential "
        "sums of squares.\n\n"
        "Card '3 · REDUCTION': Backward elimination one term at a time, preserving model "
        "hierarchy.\n\n"
        "Card '4 · VALIDATION': ANOVA, predicted R², and leave-one-formulation-out "
        "cross-validation.\n\n"
        "Sentence below: Fixed-time responses were modelled directly, and the Weibull "
        "parameters were modelled and recombined into predicted profiles.",
        notes="Vessel replicates came from one compression batch; ANOVA p-values are "
              "therefore optimistic, and the cross-validated error is the measure of "
              "predictive accuracy.",
    )

    # Results overview: one card per headline number, for every API.
    levers = " ".join(_lever_sentence(p)[0] for p in inputs)
    cv = ", ".join(f"{p.api} {p.analysis.cross_validation.profile_rmse_pct:.2f}%"
                   for p in inputs)
    slide(
        "Results overview", "Statement",
        "Big-number cards, each with its sentence:\n"
        + "".join(f"- {_lever_sentence(p)[0]}\n" for p in inputs)
        + f"- Cross-validated profile prediction error: {cv}.\n"
        + "".join(f"- {_swap_sentence(p)}\n" for p in inputs)
        + "".join(f"- {_stress_sentence(p)}\n" for p in inputs)
        + "\nUse the leading figure of each sentence as the big number.",
        notes="Every value on this slide is taken from this run's analysis.",
    )

    fig_text = {
        "Fig2": ("Measured release profiles", "Figure, side",
                 [_grade_trend(p) for p in inputs] + [_hpmc_trend(p) for p in inputs]
                 + ["Lines show the replicate mean through every reading; markers with "
                    "±1 SD error bars mark the mean at fixed display times."]),
        "Fig3": ("Contribution of composition and grade", "Figure, wide",
                 [_lever_sentence(p)[0] for p in inputs]
                 + [_beta_lever(p) for p in inputs]),
        "Fig4": ("Response surface and model adequacy", "Figure, side",
                 [f"Response: {_label(inputs, headline_key)}; contours show the "
                  "fitted model, points the measured blends on the same colour scale."]
                 + [s for s in (_fit_sentence(p, headline_key or "") for p in inputs) if s]),
        "Fig5": ("Release mechanism across the design", "Figure, side",
                 [_beta_trend(p) for p in inputs]
                 + ["Range: " + "; ".join(s for s in (_beta_range(p) for p in inputs) if s)
                    + "."]),
        "Fig6": ("Interchangeability of HPMC grades", "Figure, side",
                 [_swap_sentence(p) for p in inputs]),
        "Fig7": ("Disintegration and dissolution", "Figure, side",
                 [_dt_sentence(p) for p in inputs]),
        "Fig8": ("Reduced experimental designs", "Figure, wide",
                 [_stress_sentence(p) for p in inputs]),
    }
    why = {r.id: r.context for r in records}
    for fid, (title, layout, bullets) in fig_text.items():
        if fid not in files:
            continue
        body = ("WHAT IT SHOWS:\n" + "".join(f"- {b}\n" for b in bullets if b)
                + f"\nWHY IT MATTERS: {WHY_SHORT.get(fid, '')}")
        slide(f"{fid.replace('Fig', 'Figure ')} · {title}", layout, body, files[fid],
              notes=why.get(fid, ""))

    gated = any(p.manuscript.questions[4].status == "gated" for p in inputs)
    slide(
        "Conclusions and outlook", "Cards (2 x 2)",
        f"Card 'Formulation levers': {levers}\n\n"
        "Card 'Model': The mixture models predicted unseen formulations with a "
        f"cross-validated error of {cv}.\n\n"
        "Card 'Practical outcome': "
        + " ".join(_swap_sentence(p) for p in inputs) + "\n\n"
        "Card 'Outlook': "
        + ("Additional APIs will allow the effect of API solubility on release mechanism "
           "to be examined; solubility comparisons were not made in this study."
           if gated else
           "The cross-API comparison examined the effect of API solubility; results are "
           "reported as directional, because solubility is confounded with molecule "
           "identity."),
    )

    lines = [
        "# Deck brief: CR matrix tablet results",
        "",
        "This file instructs a slide builder (a person or an AI agent) to produce the "
        "results deck for this run. It was generated by the analysis pipeline from the "
        "same results as the figures in this folder; regenerate it with every run.",
        "",
        ("**This run used synthetic placeholder data. Every number below is illustrative "
         "and must not be presented as a finding.**" if synthetic else
         f"APIs in this run: {_api_list(apis)}."),
        "",
        RULES,
        DESIGN,
        "## Slides",
        "",
    ]
    for n, s in enumerate(slides, start=1):
        lines += [f"### Slide {n}: {s['title']}", "", f"- **Layout:** {s['layout']}"]
        if s["figure"]:
            lines.append(f"- **Figure:** `{s['figure']}`")
        if s["footer"]:
            lines.append(f"- **Footer:** {s['footer']}")
        lines += ["", s["body"].rstrip(), ""]
        if s["notes"]:
            lines += [f"*Speaker notes:* {s['notes']}", ""]
    return "\n".join(lines).rstrip() + "\n"


def write(inputs: list[PaperInput], records: Sequence[FigureRecord], out: Path,
          headline_key: str | None) -> Path:
    path = out / "deck_brief.md"
    path.write_text(build(inputs, records, headline_key), encoding="utf-8", newline="\n")
    return path
