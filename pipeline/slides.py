"""Ready-made slide images for the results deck.

``Slide_model_approach.png`` explains the modelling in one slide: what goes
in, the two model routes (one model per release metric, and one model of the
whole curve through its Weibull parameters), how they are checked, and what
they are used for, in plain words. ``Slide_model_equations.png`` follows it:
the same slide with the two routes written out as equations. Both are drawn
in the deck's palette at 16:9, so a slide builder can place them as they are
or rebuild them from the same layout. They show no data, so they are the same
for every run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY = "#14213D"
PAPER = "#F6F4EE"
CARD = "#FBFAF6"
RULE = "#DCD8CE"
MUTED = "#4A5568"
FOOT = "#6B7280"
ACCENT = "#0072B2"
ACCENT2 = "#D55E00"
LIGHT = "#7FB8E0"


def _installed(*names: str) -> str:
    """The first of these typefaces this machine has, chosen quietly."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in names if n in have), names[-1])


SERIF = _installed("Source Serif 4", "Georgia", "DejaVu Serif")
SANS = _installed("IBM Plex Sans", "Arial", "DejaVu Sans")

#: The response model, written out for the slide (mathtext). A, H, L are the
#: API, HPMC and lactose fractions; v is the coded grade.
SCHEFFE_LINES = (
    r"$\hat{y} = \beta_A A + \beta_H H + \beta_L L"
    r" + \beta_{AH}AH + \beta_{AL}AL + \beta_{HL}HL$",
    r"$\qquad + (\gamma_A A + \gamma_H H + \gamma_L L)\,v"
    r" + (\delta_A A + \delta_H H + \delta_L L)\,v^2$",
)
WEIBULL_LINE = r"$F(t) = F_\infty\left[1 - \exp\left(-(t/T_d)^{\beta}\right)\right]$"


def _card(ax: Any, x: float, y: float, w: float, h: float, eyebrow: str, title: str,
          lines: list[str], colour: str = ACCENT, math: tuple[str, ...] = (),
          note: str = "") -> None:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=CARD, edgecolor=RULE, linewidth=1.2))
    ax.text(x + 0.012, y + h - 0.03, eyebrow, color=colour, fontsize=10.5,
            fontweight="bold", family=SANS, va="top")
    ax.text(x + 0.012, y + h - 0.075, title, color=NAVY, fontsize=14,
            fontweight="bold", family=SANS, va="top")
    # The body starts below however many lines the title takes; equations
    # follow the body, and a one-line note follows the equations.
    title_lines = len(title.splitlines())
    body_top = y + h - 0.08 - 0.045 * title_lines
    ax.text(x + 0.012, body_top, "\n".join(lines), color=MUTED, fontsize=11, family=SANS,
            va="top", linespacing=1.45)
    eq_top = body_top - 0.036 * len(lines) - 0.012
    for k, expr in enumerate(math):
        ax.text(x + 0.012, eq_top - 0.055 * k, expr, color=NAVY, fontsize=13, va="top")
    if note:
        ax.text(x + 0.012, eq_top - 0.055 * len(math) - 0.008, note, color=MUTED,
                fontsize=9.5, family=SANS, va="top")


def _arrow(ax: Any, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                                 color=NAVY, linewidth=1.4))


def model_process(out: Path, equations: bool = True) -> Path:
    """Draw the modelling-approach slide and return its path.

    With ``equations`` the model cards show the polynomial and the Weibull
    function; without, they say in words what each model does.
    """
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150)
    fig.patch.set_facecolor(PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.92, "Modelling approach: from blends to predicted release",
            color=NAVY, fontsize=28, fontweight="bold", family=SERIF, va="center")

    top, h = 0.81, 0.61
    _card(ax, 0.05, top - h, 0.14, h, "1 · INPUTS", "What defines a\nformulation",
          ["Composition:", "API (A), HPMC (H),", "lactose (L) fractions,",
           "A + H + L = 1", "", "HPMC grade, as", "coded log viscosity v:",
           "K100LV = -1", "K4M = 0", "K100M = +1"])
    _arrow(ax, 0.195, top - h / 2, 0.215, top - h / 2)
    _card(ax, 0.22, top - h, 0.14, h, "2 · DATA", "What the models\nlearn from",
          ["33 formulations", "(11 blends x 3", "grades)", "", "Each the mean of",
           "3 vessels", "", "Release metrics and", "Weibull fits from", "each profile"])
    _arrow(ax, 0.365, top - 0.14, 0.385, top - 0.10)
    _arrow(ax, 0.365, top - h + 0.14, 0.385, top - h + 0.10)

    hh = 0.32
    if equations:
        a_lines = ["Scheffé polynomial crossed with grade (no intercept):"]
        b_lines = ["Each vessel's profile is fitted with the Weibull function:"]
        a_note = "Terms the data do not support are dropped."
        b_note = ("log Td, β and F∞ each get the polynomial in 3A, "
                  "then rebuild a full curve.")
    else:
        a_lines = ["Each release metric is modelled separately as a smooth",
                   "function of composition and HPMC grade.", "",
                   "Blending effects of each component, how pairs of",
                   "components interact, and how grade changes each effect.", "",
                   "Terms the data do not support are dropped."]
        b_lines = ["Each vessel's profile is summarised by three Weibull",
                   "parameters: time scale Td, shape β and plateau F∞.", "",
                   "Each parameter is modelled like a metric in 3A;",
                   "together they rebuild the full predicted curve."]
        a_note = b_note = ""
    _card(ax, 0.39, top - hh, 0.37, hh, "3A · ONE MODEL PER METRIC",
          "t50, t80, % at 1-24 h, MDT", a_lines, colour=ACCENT,
          math=SCHEFFE_LINES if equations else (), note=a_note)
    _card(ax, 0.39, top - h, 0.37, h - hh - 0.015, "3B · ONE MODEL OF THE CURVE",
          "Weibull Td, β, F∞", b_lines, colour=ACCENT2,
          math=(WEIBULL_LINE,) if equations else (), note=b_note)
    _arrow(ax, 0.765, top - h / 2, 0.785, top - h / 2)

    _card(ax, 0.79, top - h, 0.16, h, "4 · CHECKS", "Is the model\ntrustworthy?",
          ["Terms kept only if", "supported (backward", "elimination)", "",
           "R²: fit to the 33", "Q²: predicts a left-", "out formulation", "",
           "Leave-one-out curve", "error on every", "prediction"])

    ax.add_patch(FancyBboxPatch((0.05, 0.075), 0.90, 0.09,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=NAVY, edgecolor=NAVY))
    ax.text(0.065, 0.12, "5 · USES", color=LIGHT, fontsize=11, fontweight="bold",
            family=SANS, va="center")
    uses = ("Response maps  ·  predicted vs measured  ·  release mechanism (β)  ·  "
            "target-profile design  ·  reduced designs for new APIs")
    ax.text(0.145, 0.12, uses, color=PAPER, fontsize=12, family=SANS, va="center")

    ax.text(0.05, 0.035, "Predictions are made only inside the tested blends; outside "
            "them the tools refuse rather than extrapolate.", color=FOOT, fontsize=10.5,
            family=SANS, va="center")
    ax.text(0.95, 0.035, "Schematic; no data shown", color=FOOT, fontsize=10.5,
            family=SANS, va="center", ha="right")

    out.mkdir(parents=True, exist_ok=True)
    path = out / ("Slide_model_equations.png" if equations else "Slide_model_approach.png")
    fig.savefig(path, dpi=150, facecolor=PAPER, metadata={"Software": None})
    plt.close(fig)
    return path
