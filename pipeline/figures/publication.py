"""Journal-style figure conventions, shared by every static figure in the repo.

One module owns the look so no two figures can drift apart: a closed box with
inward, mirrored major and minor ticks; no grid; no in-plot titles (the takeaway
belongs in the caption or on the slide); bold (A)/(B) panel letters; sans-serif
type at journal sizes; and a grade palette that still reads in greyscale because
every grade also carries its own marker and line style.

Widths follow the common single / one-and-a-half / double column convention, so
a figure drawn at ``SINGLE`` drops into a journal column at 100 % without its
type shrinking below 7 pt.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.transforms import ScaledTranslation

from pipeline import config

#: Figure widths, inches.
SINGLE = 3.5
ONEHALF = 5.5
DOUBLE = 7.2

#: Printed on every figure drawn from a database that declares itself synthetic.
SYNTHETIC_BANNER = "SYNTHETIC PLACEHOLDER DATA — NOT EXPERIMENTAL"

#: When True, ``save`` refuses a figure that carries an axes title or suptitle.
#: The tests switch it on; titles belong in captions.
STRICT = False

#: Okabe–Ito, the colour-blind-safe qualitative set.
OKABE_ITO = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",
)

#: Neutral ink for reference lines, identity lines and annotations.
INK = "#333333"
MUTED = "#8C8C8C"
HIGHLIGHT = "#D55E00"


@dataclass(frozen=True)
class SeriesStyle:
    """Colour, marker and line style for one series."""

    colour: str
    marker: str
    linestyle: str

    def line(self) -> dict[str, Any]:
        return {"color": self.colour, "marker": self.marker, "linestyle": self.linestyle}


#: Grades are ordered by viscosity; the hues stay close to the dashboard's.
GRADE_STYLE: dict[str, SeriesStyle] = {
    "K100LV": SeriesStyle("#0072B2", "o", "-"),
    "K4M": SeriesStyle("#E69F00", "s", "--"),
    "K100M": SeriesStyle("#CC79A7", "^", "-."),
}
_FALLBACK_MARKERS = ("D", "v", "P", "X", "h")


def grade_style(grade: str, index: int = 0) -> SeriesStyle:
    """The style for a grade, with a deterministic fallback for unknown grades."""
    known = GRADE_STYLE.get(grade)
    if known is not None:
        return known
    return SeriesStyle(
        OKABE_ITO[3 + index % 5], _FALLBACK_MARKERS[index % len(_FALLBACK_MARKERS)], ":"
    )


def series_colour(index: int) -> str:
    """Colour for a series that is not a grade (components, models, ...)."""
    return OKABE_ITO[index % len(OKABE_ITO)]


def apply_style() -> None:
    """Set the journal rcParams. Idempotent."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "axes.linewidth": 0.6,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "axes.labelpad": 3.0,
            "axes.unicode_minus": True,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.6,
            "errorbar.capsize": 2.0,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "figure.autolayout": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def new_figure(
    width: float,
    height: float | None = None,
    *,
    aspect: float = 0.75,
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kw: Any,
) -> tuple[Figure, Any]:
    """A figure with constrained layout. ``height`` defaults to width x aspect."""
    apply_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height if height is not None else width * aspect),
        layout="constrained",
        **subplot_kw,
    )
    return fig, axes


def _dpi_trans(ax: Axes) -> Any:
    fig = ax.figure
    assert fig is not None
    return fig.dpi_scale_trans


def panel_label(ax: Axes, letter: str) -> None:
    """A bold "(A)" just outside the top-left corner of the axes."""
    offset = ScaledTranslation(-24 / 72, 3 / 72, _dpi_trans(ax))
    # 3-D axes take (x, y, z, s) in text(); text2D is their 2-D equivalent.
    draw = getattr(ax, "text2D", ax.text)
    draw(
        0.0,
        1.0,
        f"({letter})",
        transform=ax.transAxes + offset,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def label_panels(axes: Iterable[Axes], start: str = "A") -> None:
    """Letter every axes in reading order."""
    for i, ax in enumerate(axes):
        panel_label(ax, chr(ord(start) + i))


def corner_note(ax: Axes, text: str, loc: str = "upper left") -> None:
    """A short in-axes label (e.g. the grade) instead of a title."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    draw = getattr(ax, "text2D", ax.text)
    draw(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7.5)


def header_note(ax: Axes, text: str) -> None:
    """A short label just above the top-right corner of the axes, clear of the data."""
    offset = ScaledTranslation(0, 3 / 72, _dpi_trans(ax))
    ax.text(1.0, 1.0, text, transform=ax.transAxes + offset, ha="right", va="bottom",
            fontsize=7.5)


def spread_labels(ys: Sequence[float], min_gap: float) -> list[float]:
    """Nudge label y positions apart so no two sit closer than ``min_gap``.

    Keeps the ordering and the mean position of each crowded cluster.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = [float(ys[i]) for i in order]
    for _ in range(50):
        moved = False
        for k in range(1, len(placed)):
            gap = placed[k] - placed[k - 1]
            if gap < min_gap:
                push = (min_gap - gap) / 2
                placed[k - 1] -= push
                placed[k] += push
                moved = True
        if not moved:
            break
    out = [0.0] * len(ys)
    for rank, i in enumerate(order):
        out[i] = placed[rank]
    return out


def categorical(ax: Axes, axis: str = "x") -> None:
    """Minor ticks on a categorical axis are noise; remove them."""
    if axis in ("x", "both"):
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    if axis in ("y", "both"):
        ax.yaxis.set_minor_locator(ticker.NullLocator())


def time_axis(ax: Axes) -> None:
    """Fixed time axis from ``config.PLOT_*`` with 4 h major ticks."""
    ax.set_xlim(config.PLOT_MIN_TIME_H, config.PLOT_MAX_TIME_H)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Time (h)")


def percent_axis(ax: Axes, label: str = "Drug released (%)") -> None:
    """Fixed release axis from ``config.PLOT_*`` with 20 % major ticks."""
    ax.set_ylim(config.PLOT_MIN_RELEASE_PCT, config.PLOT_MAX_RELEASE_PCT)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.set_ylabel(label)


def log_axis(ax: Axes, which: str = "y") -> None:
    """Log scaling with decade majors and 2..9 minors."""
    axis = ax.yaxis if which == "y" else ax.xaxis
    (ax.set_yscale if which == "y" else ax.set_xscale)("log")
    axis.set_major_locator(ticker.LogLocator(base=10))
    subs = [k / 10 for k in range(2, 10)]
    axis.set_minor_locator(ticker.LogLocator(base=10, subs=subs))


def auto_minor(ax: Axes, n: int = 2) -> None:
    """Minor ticks subdividing each major interval into ``n`` on linear axes."""
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n))
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n))


def _check_titles(fig: Figure, stem: str) -> None:
    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None and suptitle.get_text():
        raise AssertionError(f"{stem}: figure has a suptitle")
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            if ax.get_title(loc=loc):
                raise AssertionError(f"{stem}: axes carries a title {ax.get_title(loc=loc)!r}")


def save(
    fig: Figure,
    out_dir: Path,
    stem: str,
    *,
    formats: Sequence[str] = ("png",),
    dpi: int = 300,
    banner: str | None = None,
) -> str:
    """Write the figure in each format and close it. Returns the PNG (or first) name.

    The banner is drawn just below the figure area; ``bbox_inches="tight"`` then
    extends the saved image to include it, so it can never sit on the axes.
    """
    if STRICT:
        _check_titles(fig, stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    if banner:
        # Below the lowest thing drawn, not at a fixed spot: a legend placed
        # under the panels would otherwise sit on top of the banner.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  # Agg canvas
        lowest = fig.get_tightbbox(renderer).y0 / fig.get_figheight()
        # Freeze the layout just computed, and pin figure legends where they were
        # drawn. bbox_inches="tight" temporarily moves the figure's bottom edge
        # down to take in the banner, and a legend anchored to that edge would
        # follow it down onto the banner. Figure-fraction coordinates do not move.
        fig.set_layout_engine("none")
        to_fig = fig.transFigure.inverted()
        for legend in fig.legends:
            box = legend.get_window_extent(renderer).transformed(to_fig)
            legend.set_bbox_to_anchor(box, transform=fig.transFigure)
            legend.set_loc("center")
        fig.text(
            0.5, min(lowest, 0.0) - 0.01, banner, ha="center", va="top",
            fontsize=6.5, color="#B22222", fontweight="bold",
        )
    names: list[str] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs: dict[str, Any] = {"format": fmt, "bbox_inches": "tight", "pad_inches": 0.03}
        if fmt in ("png", "tiff", "tif", "jpg"):
            kwargs["dpi"] = dpi
        if fmt in ("tiff", "tif"):
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        if fmt == "png":
            kwargs["metadata"] = {"Software": None}
        fig.savefig(path, **kwargs)
        names.append(path.name)
    plt.close(fig)
    png = [n for n in names if n.endswith(".png")]
    return png[0] if png else names[0]


@dataclass(frozen=True)
class FigureRecord:
    """One saved figure and its place in the argument."""

    id: str
    section: str
    rank: int
    caption: str
    file: str


def write_captions(records: Sequence[FigureRecord], out_dir: Path, title: str) -> None:
    """``captions.json`` (machine) and ``captions.md`` (paste under a slide)."""
    ordered = sorted(records, key=lambda r: (r.rank, r.id))
    manifest = [
        {
            "id": r.id,
            "name": Path(r.file).stem,
            "section": r.section,
            "rank": r.rank,
            "caption": r.caption,
            "png": r.file,
        }
        for r in ordered
    ]
    (out_dir / "captions.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    lines = [f"# {title}", ""]
    for r in ordered:
        lines += [f"## {r.id} — `{r.file}`", "", r.caption, ""]
    (out_dir / "captions.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
