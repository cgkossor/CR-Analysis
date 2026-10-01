"""Q4: does disintegration time predict dissolution?

Disintegration is quick and cheap; paddle dissolution takes a day per run. If
DT tracked the dissolution time scale closely, it could screen formulations.
The disintegration section already fits the relationship (Deming regression
of ln DT on ln Td, which allows for error in both) and cross-validates
prediction models. This question reads those results and states how far the
link can be trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pipeline.manuscript.model import Claim, QuestionResult, Table, num

if TYPE_CHECKING:
    from pipeline.disintegration.analysis import DisintegrationAnalysis
    from pipeline.figures.publication import FigureRecord

FIGURE_ID = "M4_dt_vs_dissolution"
QUESTION = "Does disintegration time predict dissolution, well enough to screen with it?"

#: Leave-one-out Q2 needed before DT is called a usable screen.
Q2_SCREEN = 0.5
Q2_TREND = 0.2


def answer(dt: DisintegrationAnalysis | None) -> QuestionResult:
    if dt is None:
        return QuestionResult(
            id="q4", short="Disintegration vs dissolution", question=QUESTION,
            answer="Needs disintegration data. Run with --disintegration FILE, or add a "
                   "Disintegration sheet.",
            status="unavailable",
        )
    corr = dt.correlation
    claims: list[Claim] = []
    pooled = next((f for f in corr.deming if f.label == "all grades"), None)
    if pooled is not None:
        lo, hi = pooled.slope_ci
        clear = np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)
        half = (hi - lo) / 2 if np.isfinite(lo) and np.isfinite(hi) else np.nan
        claims.append(Claim(
            text=("DT rises with the dissolution time scale across all grades."
                  if clear and pooled.slope > 0 else
                  "DT does not follow the dissolution time scale across grades."),
            effect=f"Deming slope of ln DT on ln Td = {pooled.slope:.2f}",
            uncertainty=f"95% CI {lo:.2f} to {hi:.2f}, n = {pooled.n}",
            status="supported" if clear else "not_supported",
            signal_to_noise=num(abs(pooled.slope) / half, 2) if half and half > 0 else None,
        ))
    within = [w for w in corr.within_grade if np.isfinite(w.spearman)]
    # "Within each grade, not only between them" needs at least two grades.
    if len(within) >= 2:
        consistent = sum(w.spearman > 0 for w in within)
        claims.append(Claim(
            text=("The link holds within each grade, not only between grades."
                  if consistent == len(within) else
                  "Within a grade the link is inconsistent; much of it is the grade itself."),
            effect=", ".join(f"{w.grade} ρ = {w.spearman:+.2f}" for w in within),
            uncertainty="n per grade " + ", ".join(str(w.n) for w in within),
            status="supported" if consistent == len(within) else "directional",
        ))
    by_key = {m.key: m for m in dt.loo}
    best = max(dt.loo, key=lambda m: m.q2 if np.isfinite(m.q2) else -np.inf, default=None)
    diss = by_key.get("dissolution_grade") or by_key.get("dissolution")
    if diss is not None and np.isfinite(diss.q2):
        claims.append(Claim(
            # The cross-validated model predicts ln DT from ln Td: dissolution
            # to disintegration. Shared information in that direction is what
            # makes DT a candidate screen; it is not a validated DT-to-release
            # prediction.
            text=("DT and dissolution share enough information that DT is a candidate "
                  "screen for release rate."
                  if diss.q2 >= Q2_SCREEN else
                  "DT and dissolution share a trend, but not enough to screen with."
                  if diss.q2 >= Q2_TREND else
                  "Dissolution does not predict DT out of sample, so DT cannot screen "
                  "for it."),
            effect=f"leave-one-out Q² = {diss.q2:.2f} ({diss.label})",
            uncertainty=f"RMSE {diss.rmse_ln:.2f} in ln DT, n = {diss.n}",
            status=("supported" if diss.q2 >= Q2_SCREEN else
                    "directional" if diss.q2 >= Q2_TREND else "not_supported"),
        ))

    table = Table(
        title="Leave-one-out prediction of ln DT",
        columns=("model", "parameters", "Q²", "RMSE (ln DT)", "n"),
        rows=tuple((m.label, m.n_params, num(m.q2, 2), num(m.rmse_ln, 3), m.n)
                   for m in dt.loo),
        note="Q² near 1 predicts well; at or below 0 it predicts no better than the mean.",
    )
    # The headline is the screening claim when it exists: it is what the
    # question asks.
    headline = next((c for c in claims if "screen" in c.text), claims[0] if claims else None)
    status = headline.status if headline else "not_supported"
    if headline is not None and claims[0] is not headline:
        claims.remove(headline)
        claims.insert(0, headline)
    answer_text = " ".join(c.text for c in claims)
    if best is not None and np.isfinite(best.q2):
        answer_text += f" The best cross-validated model is '{best.label}' (Q² {best.q2:.2f})."
    return QuestionResult(
        id="q4", short="Disintegration vs dissolution", question=QUESTION,
        answer=answer_text.strip(), status=status, claims=tuple(claims), tables=(table,),
        figure_ids=(FIGURE_ID,),
        caveats=(
            "The two tests use different apparatus and end points; a correlation does not "
            "make them interchangeable for a specification.",
            f"{corr.n_censored_excluded} tablets still intact at the end of the test are "
            "left out of the correlation.",
        ),
        readiness={"matched": corr.n_matched, "censored_excluded": corr.n_censored_excluded},
    )


def render(dt: DisintegrationAnalysis | None, out: Path, banner: str | None
           ) -> list[FigureRecord]:
    """ln DT against ln Td by grade, with the pooled Deming line."""
    from pipeline.figures import publication as pub

    if dt is None:
        return []
    m = dt.matched
    m = m[~m["dt_censored"].astype(bool)] if "dt_censored" in m else m
    if len(m) < 3:
        return []
    fig, ax = pub.new_figure(pub.SINGLE, 2.8)
    for i, g in enumerate(dt.grade_order):
        sub = m[m["grade"] == g]
        st = pub.grade_style(str(g), i)
        ax.plot(sub["td_h"], sub["dt_h"], ls="none", marker=st.marker, color=st.colour,
                ms=4, label=str(g))
    pooled = next((f for f in dt.correlation.deming if f.label == "all grades"), None)
    if pooled is None:
        return []
    xs = np.linspace(float(m["td_h"].min()), float(m["td_h"].max()), 50)
    ax.plot(xs, np.exp(pooled.intercept + pooled.slope * np.log(xs)), color=pub.INK, lw=1)
    pub.log_axis(ax, "x")
    pub.log_axis(ax, "y")
    ax.set_xlabel("Dissolution time scale Td (h)")
    ax.set_ylabel("Disintegration time DT (h)")
    ax.legend(frameon=False, loc="upper left")
    file = pub.save(fig, out, FIGURE_ID, banner=banner)
    caption = (
        "Disintegration time against the Weibull dissolution time scale, both on log "
        f"axes, by grade. The line is the pooled Deming fit (slope {pooled.slope:.2f}, "
        f"95% CI {pooled.slope_ci[0]:.2f} to {pooled.slope_ci[1]:.2f}), which allows for "
        "error in both measurements. Tablets intact at the end of the test are omitted."
    )
    return [pub.FigureRecord(FIGURE_ID, "Q4", 4, caption, file)]
