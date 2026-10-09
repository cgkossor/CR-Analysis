"""Q3: which formulations are sensitive to hydrodynamic shear?

No dissolution run here varies paddle speed, so shear sensitivity is not
measured directly. What the data do hold is the same tablets under two very
different hydrodynamic regimes: the disintegration test (basket, discs and
vigorous mechanical agitation) and paddle dissolution. A matrix whose gel
holds together under agitation disintegrates late relative to how fast it
releases in the paddle; one that is eroded mechanically falls apart early.

Slow tablets take longer to disintegrate too, so DT and Td rise together
(the disintegration section fits ln DT = a + b ln Td by Deming regression).
A raw ratio Td / DT would then rank tablets mostly by how slow they are,
whenever b differs from 1. The index is instead the departure from that
pooled fit:

    shear index = log10(predicted DT / measured DT)
                = (a + b ln Td - ln DT) / ln 10

with Td the Weibull time scale of paddle release and DT the geometric-mean
disintegration time. Zero means the tablet breaks up when its release speed
says it should. Positive means it breaks up earlier under agitation than its
paddle release predicts: more sensitive to shear.

One pooled line leaves any grade offset in the index: if a whole grade
disintegrates early for its release speed, every formulation of that grade
scores high. So a second, within-grade index is reported beside it, from one
shared slope with a separate offset per grade (ANCOVA, least squares):

    within-grade index = (a_grade + b ln Td - ln DT) / ln 10

It compares a tablet only with tablets of its own grade, and averages zero
within each grade by construction. The two read together: the pooled index
says how sensitive a formulation is overall, the within-grade index which
compositions are more sensitive than their grade-mates, and the difference
between them (the grade shift) how much of the pooled index is grade alone.

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
    #: Departure from the shared-slope, per-grade-offset line: compares a tablet
    #: only with its grade-mates. NaN when the grade has fewer than two points.
    index_within: float = float("nan")


@dataclass(frozen=True)
class WithinGradeFit:
    """ln DT = offset[grade] + slope x ln Td, one slope shared by every grade."""

    slope: float
    offsets: dict[str, float]


def within_grade_fit(pts: list[ShearPoint]) -> WithinGradeFit | None:
    """Least-squares common-slope fit over grades with at least two points."""
    counts: dict[str, int] = {}
    for p in pts:
        counts[p.grade] = counts.get(p.grade, 0) + 1
    grades = sorted(g for g, n in counts.items() if n >= 2)
    use = [p for p in pts if p.grade in grades]
    if len(use) < len(grades) + 2:
        return None
    x = np.log([p.td_h for p in use])
    y = np.log([p.dt_h for p in use])
    dummies = [[1.0 if p.grade == g else 0.0 for p in use] for g in grades]
    design = np.column_stack([*dummies, x])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    offsets = {g: float(c) for g, c in zip(grades, coef[:-1], strict=True)}
    return WithinGradeFit(float(coef[-1]), offsets)


def grade_shifts(pts: list[ShearPoint]) -> dict[str, float]:
    """Per grade, the mean of pooled minus within-grade index: grade alone."""
    out: dict[str, float] = {}
    for g in sorted({p.grade for p in pts}):
        d = [p.index - p.index_within for p in pts
             if p.grade == g and np.isfinite(p.index_within)]
        if d:
            out[g] = float(np.mean(d))
    return out


def pooled_fit(dt: DisintegrationAnalysis) -> Any:
    """The all-grades Deming fit of ln DT on ln Td, or None."""
    return next((f for f in dt.correlation.deming if f.label == "all grades"), None)


def points(dt: DisintegrationAnalysis) -> tuple[list[ShearPoint], int, int]:
    """The index per formulation, and how many were left out (intact, non-finite).

    Tablets still intact at the end of the test have no DT and are skipped.
    """
    fit = pooled_fit(dt)
    if fit is None:
        return [], 0, 0
    out: list[ShearPoint] = []
    intact = nonfinite = 0
    for _, r in dt.matched.iterrows():
        if bool(r.get("dt_censored", False)):
            intact += 1
            continue
        td, d = float(r.get("td_h", np.nan)), float(r.get("dt_h", np.nan))
        if not (np.isfinite(td) and np.isfinite(d)) or td <= 0 or d <= 0:
            nonfinite += 1
            continue
        index = (fit.intercept + fit.slope * np.log(td) - np.log(d)) / np.log(10.0)
        out.append(ShearPoint(int(r["case"]), str(r["grade"]), float(r["api_wt"]),
                              float(r["hpmc_wt"]), float(r["viscosity_cp"]), td, d,
                              float(index)))
    within = within_grade_fit(out)
    if within is not None:
        from dataclasses import replace

        out = [
            replace(p, index_within=float(
                (within.offsets[p.grade] + within.slope * np.log(p.td_h) - np.log(p.dt_h))
                / np.log(10.0)))
            if p.grade in within.offsets else p
            for p in out
        ]
    return sorted(out, key=lambda p: (p.case, p.viscosity_cp)), intact, nonfinite


def tiers(pts: list[ShearPoint]) -> dict[str, dict[str, Any]]:
    """Per formulation: each index and its third (low / mid / high sensitivity).

    ``tier`` is from the pooled index (overall sensitivity); ``tier_within``
    from the within-grade index (compared with grade-mates only).
    """
    if not pts:
        return {}

    def thirds(vals: np.ndarray) -> tuple[float, float]:
        finite = vals[np.isfinite(vals)]
        if not finite.size:
            return float("nan"), float("nan")
        lo, hi = np.quantile(finite, [1 / 3, 2 / 3])
        return float(lo), float(hi)

    def tier(v: float, cut: tuple[float, float]) -> str | None:
        if not np.isfinite(v) or not np.isfinite(cut[0]):
            return None
        return "low" if v <= cut[0] else "high" if v > cut[1] else "mid"

    pooled = thirds(np.array([p.index for p in pts]))
    within = thirds(np.array([p.index_within for p in pts]))
    return {
        f"{p.case}|{p.grade}": {
            "index": num(p.index, 3), "tier": tier(p.index, pooled),
            "index_within": num(p.index_within, 3) if np.isfinite(p.index_within) else None,
            "tier_within": tier(p.index_within, within),
        }
        for p in pts
    }


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
    pts, intact, nonfinite = points(dt)
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
        columns=("case", "grade", "API wt%", "HPMC wt%", "Td (h)", "DT (h)",
                 "shear index (pooled)", "within grade"),
        rows=tuple((p.case, p.grade, num(p.api_wt, 1), num(p.hpmc_wt, 1), num(p.td_h, 2),
                    num(p.dt_h, 2), num(p.index, 2),
                    num(p.index_within, 2) if np.isfinite(p.index_within) else "–")
                   for p in reversed(ordered)),
        note="shear index = log10(DT predicted from Td / DT measured); positive breaks up "
             "earlier than its release speed predicts. Pooled: one line for all grades. "
             "Within grade: one shared slope with an offset per grade, so a tablet is "
             "compared only with its grade-mates. " + PROXY_LABEL.capitalize() + ".",
    )

    claims: list[Claim] = []
    n = len(pts)
    within_pts = [p for p in pts if np.isfinite(p.index_within)]
    # (prefix, points, which index, factor name, factor). Within a grade the
    # grade viscosity is constant, so only composition can be tested there.
    tests: list[tuple[str, list[ShearPoint], Any, str, Any]] = [
        ("", pts, lambda q: q.index, "HPMC content", lambda q: q.hpmc_wt),
        ("", pts, lambda q: q.index, "grade viscosity",
         lambda q: np.log10(q.viscosity_cp)),
        ("", pts, lambda q: q.index, "API content", lambda q: q.api_wt),
        ("Within a grade: ", within_pts, lambda q: q.index_within, "HPMC content",
         lambda q: q.hpmc_wt),
        ("Within a grade: ", within_pts, lambda q: q.index_within, "API content",
         lambda q: q.api_wt),
    ]
    for prefix, group, index_of, label, factor in tests:
        xs = [factor(q) for q in group]
        if len(group) < 4 or len(set(np.round(xs, 9))) < 2:
            continue  # one grade only: no trend with viscosity to test
        res = stats.spearmanr(xs, [index_of(q) for q in group])
        rho, p = float(res.statistic), float(res.pvalue)
        if not np.isfinite(rho):
            continue
        direction = "more" if rho > 0 else "less"
        body = (f"higher {label} makes a tablet {direction} shear-sensitive." if p < 0.05
                else f"no clear shear-sensitivity trend with {label}.")
        claims.append(Claim(
            text=prefix + body if prefix else body[0].upper() + body[1:],
            effect=f"ρ = {rho:+.2f}",
            uncertainty=f"n = {len(group)}, p = {p:.3f}",
            # A proxy never earns more than directional, however clean the trend.
            status="directional" if p < 0.05 else "not_supported",
            signal_to_noise=num(abs(rho) * np.sqrt(len(group) - 1), 2),
        ))
    # The index should not simply track release speed; if it does, the
    # pooled fit has not removed the speed trend and every ranking inherits it.
    lt = stats.spearmanr([np.log(p.td_h) for p in pts], [p.index for p in pts])
    speed_rho = float(lt.statistic)
    # Clearest trend first: it is the question's headline claim.
    claims.sort(key=lambda c: -(c.signal_to_noise or 0.0))
    most, least = ordered[-1], ordered[0]
    shifts = grade_shifts(pts)
    shift_text = (
        "Grade shift (how much of the pooled index is grade alone, log10): "
        + ", ".join(f"{g} {v:+.2f}" for g, v in shifts.items()) + ". "
        if len(shifts) > 1 else ""
    )
    answer_text = (
        f"On the disintegration proxy, case {most.case} / {most.grade} is the most "
        f"shear-sensitive and case {least.case} / {least.grade} the least. "
        + shift_text
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
            f"Residual check: ρ between the index and ln Td is {speed_rho:+.2f}; near zero "
            "means the index is not just release speed in disguise.",
            "Confirm with dissolution at two or more paddle speeds before claiming shear "
            "sensitivity in a paper.",
        ),
        readiness={"formulations": n, "excluded_intact": intact,
                   "excluded_nonfinite": nonfinite,
                   "index_rho_ln_td": num(speed_rho, 2),
                   "grade_shift": {g: num(v, 3) for g, v in shifts.items()}},
    ), pts


def render(pts: list[ShearPoint], out: Path, banner: str | None,
           fit: Any = None) -> list[FigureRecord]:
    """The two indices and the fits they are measured from, in a 2 x 2 grid.

    (A) ln DT against ln Td with the pooled Deming line (solid) and the
    shared-slope line of each grade (dashed, grade colour); the stick from each
    formulation to the pooled line is its pooled index. (B) Pooled index by
    grade: a grade sitting wholly to one side of zero is the grade shift.
    (C) Pooled index against HPMC content. (D) Within-grade index against HPMC
    content: what is left once each tablet is compared only with its grade-mates.
    """
    from matplotlib.ticker import FuncFormatter

    from pipeline.figures import publication as pub

    if len(pts) < 4:
        return []
    grades = sorted({p.grade for p in pts},
                    key=lambda g: next(p.viscosity_cp for p in pts if p.grade == g))
    within = within_grade_fit(pts)
    fig, axes = pub.new_figure(pub.DOUBLE, 5.2, nrows=2, ncols=2)
    (ax_f, ax_g), (ax_p, ax_w) = axes
    td_all = np.array([p.td_h for p in pts])
    grid = np.geomspace(td_all.min() / 1.2, td_all.max() * 1.2, 50)
    for i, g in enumerate(grades):
        st = pub.grade_style(g, i)
        gp = sorted((p for p in pts if p.grade == g), key=lambda p: p.hpmc_wt)
        td = np.array([p.td_h for p in gp])
        dt = np.array([p.dt_h for p in gp])
        if fit is not None:
            expected = np.exp(fit.intercept + fit.slope * np.log(td))
            ax_f.vlines(td, dt, expected, color=st.colour, lw=0.6, alpha=0.7, zorder=2)
        if within is not None and g in within.offsets:
            span = np.geomspace(td.min() / 1.1, td.max() * 1.1, 20)
            ax_f.plot(span, np.exp(within.offsets[g] + within.slope * np.log(span)),
                      color=st.colour, ls="--", lw=0.9, zorder=3)
        ax_f.plot(td, dt, ls="none", marker=st.marker, color=st.colour, ms=4,
                  markeredgecolor="black", markeredgewidth=0.4, label=g, zorder=4)
        ys = [p.index for p in gp]
        ax_g.scatter(np.full(len(ys), i) + np.linspace(-0.12, 0.12, len(ys)), ys,
                     s=14, color=st.colour, marker=st.marker)
        if ys:
            ax_g.hlines(np.median(ys), i - 0.25, i + 0.25, color=pub.INK, lw=1)
        ax_p.plot([p.hpmc_wt for p in gp], ys, ls="none", marker=st.marker,
                  color=st.colour, ms=4, label=g)
        wp = [p for p in gp if np.isfinite(p.index_within)]
        ax_w.plot([p.hpmc_wt for p in wp], [p.index_within for p in wp], ls="none",
                  marker=st.marker, color=st.colour, ms=4, label=g)
    if fit is not None:
        ax_f.plot(grid, np.exp(fit.intercept + fit.slope * np.log(grid)), color=pub.INK,
                  lw=1.1, label=f"pooled, slope {fit.slope:.2f}", zorder=5)
    if within is not None:
        ax_f.plot([], [], color=pub.MUTED, ls="--", lw=0.9,
                  label=f"per grade, shared slope {within.slope:.2f}")
    ax_f.set_xscale("log")
    ax_f.set_yscale("log")
    for axis in (ax_f.xaxis, ax_f.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax_f.set_xlabel("Weibull Td (h)")
    ax_f.set_ylabel("Disintegration time, DT (h)")
    ax_f.legend(frameon=False, loc="upper left", fontsize=6)
    for ax, what in ((ax_g, "pooled"), (ax_p, "pooled"), (ax_w, "within grade")):
        ax.axhline(0, color=pub.MUTED, lw=0.6, ls="--")
        ax.set_ylabel(f"Shear index, {what}")
    for ax in (ax_p, ax_w):
        ax.set_xlabel("HPMC (wt%)")
        pub.auto_minor(ax)
    # Same scale on C and D, so the share of the pooled index that is grade
    # alone shows as the spread that disappears between them.
    lo = min(ax_p.get_ylim()[0], ax_w.get_ylim()[0])
    hi = max(ax_p.get_ylim()[1], ax_w.get_ylim()[1])
    for ax in (ax_p, ax_w):
        ax.set_ylim(lo, hi)
    ax_g.set_xticks(range(len(grades)), grades)
    pub.categorical(ax_g, "x")
    pub.label_panels([ax_f, ax_g, ax_p, ax_w])
    file = pub.save(fig, out, FIGURE_ID, banner=banner)
    line = (f"ln DT = {fit.intercept:.2f} + {fit.slope:.2f} ln Td, n = {fit.n}"
            if fit is not None else "the pooled fit")
    shifts = grade_shifts(pts)
    shift = ("; grade shift (mean of pooled minus within-grade index): "
             + ", ".join(f"{g} {v:+.2f}" for g, v in shifts.items())
             if len(shifts) > 1 else "")
    shared = (f"shared slope {within.slope:.2f}" if within is not None
              else "not estimable")
    caption = (
        "Shear-sensitivity proxy, measured two ways. (A) Disintegration time against the "
        "paddle-dissolution time scale Td on log axes. Solid line: the pooled Deming fit "
        f"across all grades ({line}). Dashed lines: one line per grade with a slope shared "
        f"by all grades ({shared}). The stick from each formulation to the pooled line is "
        "its pooled shear index: below the line, it breaks up earlier under basket-and-disc "
        "agitation than its release speed predicts (positive index). (B) Pooled index by "
        "grade; bars are medians. A grade lying wholly to one side of zero carries a grade "
        f"offset into the pooled index{shift}. (C) Pooled index against HPMC content. "
        "(D) Within-grade index (departure from the grade's own dashed line) against HPMC "
        "content: each tablet compared only with its grade-mates; it averages zero within "
        "each grade. C and D share a scale. Tablets still intact at the end of the "
        "disintegration test have no DT and are not shown. A proxy from two different "
        "apparatus, not a direct measurement of the response to paddle speed."
    )
    return [pub.FigureRecord(FIGURE_ID, "Q3", 3, caption, file)]
