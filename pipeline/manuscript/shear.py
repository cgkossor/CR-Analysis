"""Q3: which formulations are sensitive to hydrodynamic shear?

No dissolution run here varies paddle speed, so shear sensitivity is not
measured directly. What the data do hold is the same tablets under two very
different hydrodynamic regimes: the disintegration test (basket, discs and
vigorous mechanical agitation) and paddle dissolution. A matrix whose gel
holds together under agitation disintegrates late relative to how fast it
releases in the paddle; one that is eroded mechanically falls apart early.

The proxy index is therefore

    shear index = log10(Td / DT)

with Td the Weibull time scale of paddle release and DT the geometric-mean
disintegration time. Higher means the tablet breaks up earlier under
agitation than its paddle release would suggest: more sensitive to shear.

Every place this appears it is labelled a proxy. The two tests differ in
apparatus, medium volume and end point, so the index ranks formulations; it
does not measure a response to rpm. When dissolution at more than one paddle
speed becomes available, it replaces this directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import stats

from pipeline.manuscript.model import Claim, QuestionResult, Table, num

if TYPE_CHECKING:
    from pipeline.analysis import Analysis
    from pipeline.disintegration.analysis import DisintegrationAnalysis
    from pipeline.figures.publication import FigureRecord

FIGURE_ID = "M3_shear_proxy"
PROXY_LABEL = "proxy (disintegration vs paddle dissolution), not a direct shear measurement"
QUESTION = ("Which formulations are sensitive to hydrodynamic shear, and for a target "
            "profile, which option is least sensitive?")


@dataclass(frozen=True)
class ShearPoint:
    case: int
    grade: str
    api_wt: float
    hpmc_wt: float
    viscosity_cp: float
    td_h: float
    dt_h: float
    index: float


def points(dt: DisintegrationAnalysis) -> list[ShearPoint]:
    """The index per formulation, skipping tablets still intact at the end of the test."""
    m = dt.matched
    out: list[ShearPoint] = []
    for _, r in m.iterrows():
        if bool(r.get("dt_censored", False)):
            continue
        td, d = float(r.get("td_h", np.nan)), float(r.get("dt_h", np.nan))
        if not (np.isfinite(td) and np.isfinite(d)) or td <= 0 or d <= 0:
            continue
        out.append(ShearPoint(int(r["case"]), str(r["grade"]), float(r["api_wt"]),
                              float(r["hpmc_wt"]), float(r["viscosity_cp"]), td, d,
                              float(np.log10(td / d))))
    return sorted(out, key=lambda p: (p.case, p.viscosity_cp))


def tiers(pts: list[ShearPoint]) -> dict[str, dict[str, Any]]:
    """Per formulation: the index and its third (low / mid / high sensitivity)."""
    if not pts:
        return {}
    vals = np.array([p.index for p in pts])
    lo, hi = np.quantile(vals, [1 / 3, 2 / 3])
    out: dict[str, dict[str, Any]] = {}
    for p in pts:
        tier = "low" if p.index <= lo else "high" if p.index > hi else "mid"
        out[f"{p.case}|{p.grade}"] = {"index": num(p.index, 3), "tier": tier}
    return out


def answer(
    analysis: Analysis, dt: DisintegrationAnalysis | None
) -> tuple[QuestionResult, list[ShearPoint]]:
    if dt is None:
        return QuestionResult(
            id="q3", short="Shear sensitivity", question=QUESTION,
            answer="Needs disintegration data. Run with --disintegration FILE, or add a "
                   "Disintegration sheet, to compute the shear proxy.",
            status="unavailable",
        ), []
    pts = points(dt)
    if len(pts) < 4:
        return QuestionResult(
            id="q3", short="Shear sensitivity", question=QUESTION,
            answer="Too few formulations disintegrated within the test to rank shear "
                   "sensitivity.",
            status="not_supported", readiness={"formulations": len(pts)},
        ), pts

    ordered = sorted(pts, key=lambda p: p.index)
    table = Table(
        title="Shear index per formulation (higher = more sensitive)",
        columns=("case", "grade", "API wt%", "HPMC wt%", "Td (h)", "DT (h)", "shear index"),
        rows=tuple((p.case, p.grade, num(p.api_wt, 1), num(p.hpmc_wt, 1), num(p.td_h, 2),
                    num(p.dt_h, 2), num(p.index, 2)) for p in reversed(ordered)),
        note="shear index = log10(Td / DT). " + PROXY_LABEL.capitalize() + ".",
    )

    claims: list[Claim] = []
    n = len(pts)
    idx = [p.index for p in pts]
    for label, xs in (("HPMC content", [p.hpmc_wt for p in pts]),
                      ("grade viscosity", [np.log10(p.viscosity_cp) for p in pts]),
                      ("API content", [p.api_wt for p in pts])):
        res = stats.spearmanr(xs, idx)
        rho, p = float(res.statistic), float(res.pvalue)
        if not np.isfinite(rho):
            continue
        direction = "more" if rho > 0 else "less"
        claims.append(Claim(
            text=(f"Higher {label} makes a tablet {direction} shear-sensitive." if p < 0.05
                  else f"No clear shear-sensitivity trend with {label}."),
            effect=f"ρ = {rho:+.2f}",
            uncertainty=f"n = {n}, p = {p:.3f}",
            # A proxy never earns more than directional, however clean the trend.
            status="directional" if p < 0.05 else "not_supported",
            signal_to_noise=num(abs(rho) * np.sqrt(n - 1), 2),
        ))
    # Clearest trend first: it is the question's headline claim.
    claims.sort(key=lambda c: -(c.signal_to_noise or 0.0))
    most, least = ordered[-1], ordered[0]
    answer_text = (
        f"On the disintegration proxy, case {most.case} / {most.grade} is the most "
        f"shear-sensitive and case {least.case} / {least.grade} the least. "
        + " ".join(c.text for c in claims if c.status == "directional")
        + " The Formulator's target-profile shortlist shows this tier for each candidate, "
          "so between options that meet the same profile the less sensitive one can be "
          "chosen."
    )
    return QuestionResult(
        id="q3", short="Shear sensitivity", question=QUESTION, answer=answer_text.strip(),
        status="directional", claims=tuple(claims), tables=(table,),
        figure_ids=(FIGURE_ID,),
        caveats=(
            "This is a " + PROXY_LABEL + ".",
            "Tablets still intact at the end of the disintegration test are excluded.",
            "Confirm with dissolution at two or more paddle speeds before claiming shear "
            "sensitivity in a paper.",
        ),
        readiness={"formulations": n,
                   "excluded_intact": int(len(dt.matched) - n)},
    ), pts


def render(pts: list[ShearPoint], out: Path, banner: str | None) -> list[FigureRecord]:
    """(A) index against HPMC content by grade; (B) index by grade."""
    from pipeline.figures import publication as pub

    if len(pts) < 4:
        return []
    grades = sorted({p.grade for p in pts},
                    key=lambda g: next(p.viscosity_cp for p in pts if p.grade == g))
    fig, (ax_a, ax_b) = pub.new_figure(pub.DOUBLE, 2.8, ncols=2)
    for i, g in enumerate(grades):
        st = pub.grade_style(g, i)
        gp = sorted((p for p in pts if p.grade == g), key=lambda p: p.hpmc_wt)
        ax_a.plot([p.hpmc_wt for p in gp], [p.index for p in gp], ls="none",
                  marker=st.marker, color=st.colour, ms=4, label=g)
        ys = [p.index for p in gp]
        ax_b.scatter(np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys)), ys,
                     s=14, color=st.colour, marker=st.marker)
        if ys:
            ax_b.hlines(np.median(ys), i - 0.25, i + 0.25, color=pub.INK, lw=1)
    for ax in (ax_a, ax_b):
        ax.axhline(0, color=pub.MUTED, lw=0.6, ls="--")
        ax.set_ylabel("Shear index, log$_{10}$(Td / DT)")
    ax_a.set_xlabel("HPMC (wt%)")
    ax_a.legend(frameon=False, loc="best")
    pub.auto_minor(ax_a)
    ax_b.set_xticks(range(len(grades)), grades)
    pub.categorical(ax_b, "x")
    pub.label_panels([ax_a, ax_b])
    file = pub.save(fig, out, FIGURE_ID, banner=banner)
    caption = (
        "Shear-sensitivity proxy. The index log10(Td / DT) compares the Weibull time "
        "scale of paddle dissolution (Td) with the disintegration time under basket-and-"
        "disc agitation (DT); higher values mean a tablet breaks up earlier under "
        "agitation than its paddle release suggests. (A) Index against HPMC content, by "
        "grade. (B) Index by grade; bars are medians. A proxy from two different "
        "apparatus, not a direct measurement of the response to paddle speed."
    )
    return [pub.FigureRecord(FIGURE_ID, "Q3", 3, caption, file)]
