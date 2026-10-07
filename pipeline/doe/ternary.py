"""Ternary (simplex) geometry for the mixture plots.

The standard picture of a three-component mixture response is a triangle: each
corner a pure component, every blend a point inside, contours of the fitted
response over the region actually tested. Here each component has a lower
bound (the smallest amount any tested blend used), so the plot shows the
L-pseudocomponent triangle: the sub-triangle whose corners are each
component at its maximum with the others at their minimum. That is the usual
zoom for a constrained mixture region; axes are still labelled in real wt%.

Pure geometry and model evaluation, no plotting, so the PNG figures and the
dashboard draw the same triangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from pipeline.design.matrix import ModelSpec, build_model_matrix

COMPONENTS = ("api", "hpmc", "lactose")
LABELS = {"api": "API", "hpmc": "HPMC", "lactose": "Lactose"}
#: Corner positions (x, y) of API, HPMC and lactose in the drawn triangle.
CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]])


@dataclass(frozen=True)
class Region:
    """The pseudocomponent triangle: lower bounds and span, as fractions."""

    lower: np.ndarray  # (3,)
    span: float

    def to_xy(self, comp: np.ndarray) -> np.ndarray:
        """Blends (n, 3) as fractions -> plot coordinates (n, 2)."""
        pseudo = (np.atleast_2d(comp) - self.lower) / self.span
        return pseudo @ CORNERS

    def from_pseudo(self, pseudo: np.ndarray) -> np.ndarray:
        return self.lower + self.span * np.atleast_2d(pseudo)


def region(design_comp: np.ndarray) -> Region:
    """The smallest L-pseudocomponent triangle holding every tested blend."""
    lower = np.min(design_comp, axis=0)
    span = float(1.0 - lower.sum())
    if span <= 1e-9:  # every blend identical; fall back to the full simplex
        return Region(np.zeros(3), 1.0)
    return Region(lower, span)


@dataclass(frozen=True)
class Tick:
    """A grid line of constant wt% for one component."""

    component: str
    value: float
    #: The two points where it meets the triangle's edges.
    ends: tuple[tuple[float, float], tuple[float, float]]


def ticks(reg: Region, step_pct: float = 10.0) -> list[Tick]:
    """Grid lines at round real wt% values, one family per component.

    A line for component i at value c joins the two points on the triangle's
    edges where component i equals c.
    """
    out: list[Tick] = []
    for i, name in enumerate(COMPONENTS):
        lo = reg.lower[i] * 100.0
        hi = (reg.lower[i] + reg.span) * 100.0
        start = np.ceil((lo + 1e-9) / step_pct) * step_pct
        for c in np.arange(start, hi - 1e-9, step_pct):
            z = (c / 100.0 - reg.lower[i]) / reg.span
            ends: list[tuple[float, float]] = []
            for j in range(3):
                if j == i:
                    continue
                p = np.zeros(3)
                p[i] = z
                p[j] = 1.0 - z
                xy = p @ CORNERS
                ends.append((float(xy[0]), float(xy[1])))
            out.append(Tick(name, float(c), (ends[0], ends[1])))
    return out


@dataclass(frozen=True)
class TernaryGrid:
    """Fitted response on a triangular grid of blends, at one grade."""

    grade: str
    xy: np.ndarray        # (n, 2) plot coordinates
    value: np.ndarray     # (n,) model prediction, NaN outside the tested region
    triangles: np.ndarray  # (m, 3) grid triangles, all three corners inside


def grid(
    reg: Region,
    design_comp: np.ndarray,
    beta: np.ndarray,
    keep: list[int],
    spec: ModelSpec,
    grades: list[tuple[str, float]],
    n: int = 40,
) -> list[TernaryGrid]:
    """Predict on an n-step triangular grid, masked to the tested blends' hull."""
    pts = [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]
    pseudo = np.array(pts, dtype=float)
    comp = reg.from_pseudo(pseudo)
    xy = pseudo @ CORNERS

    index = {(i, j): k for k, (i, j) in enumerate(
        (i, j) for i in range(n + 1) for j in range(n + 1 - i))}
    tris = []
    for i in range(n):
        for j in range(n - i):
            tris.append((index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]))
            if j + 1 <= n - i - 1:
                tris.append((index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]))
    tri = np.array(tris, dtype=int)

    inside = np.ones(len(comp), dtype=bool)
    try:
        hull = Delaunay(design_comp[:, :2])
        inside = hull.find_simplex(comp[:, :2]) >= -1e-12
        inside |= hull.find_simplex(comp[:, :2] + 1e-9) >= 0
    except Exception:  # degenerate design: no hull, keep everything
        pass
    kept_tri = tri[inside[tri].all(axis=1)]

    out: list[TernaryGrid] = []
    for label, coded in grades:
        matrix = build_model_matrix(comp, np.full(len(comp), coded), spec)[:, keep]
        value = np.where(inside, matrix @ beta, np.nan)
        out.append(TernaryGrid(label, xy, value, kept_tri))
    return out
