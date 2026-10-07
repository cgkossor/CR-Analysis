"""The standard mixture-DoE views of one fitted response.

* Ternary contours, one triangle per grade, with the measured blends coloured
  by their measured value on the same scale (pipeline.doe.ternary).
* Predicted vs actual: each formulation's measured mean against the model's
  fitted value, the standard check that the model describes the data.
* Piepel response traces: the effect of each component along its Piepel
  direction (the Cox direction in L-pseudocomponents), from the reference
  blend. A supporting view of which component moves the response, not a
  picture of the data.

Built once here and drawn by both the figures and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from pipeline.design.matrix import build_model_matrix, term_names
from pipeline.doe import ternary

if TYPE_CHECKING:
    from pipeline.doe.analysis import ResponseAnalysis


def model_of(ra: ResponseAnalysis) -> tuple[np.ndarray, list[int]]:
    """Coefficients and the indices of the kept terms in the full term list."""
    labels = term_names(ra.spec)
    kept = set(ra.kept_terms)
    keep = [i for i, name in enumerate(labels) if name in kept]
    beta = np.array([c.estimate for c in ra.fit.coefficients], dtype=float)
    return beta, keep


def grade_levels(design_points: pd.DataFrame) -> list[tuple[str, float]]:
    """(grade, coded v) in viscosity order."""
    seen = design_points.drop_duplicates("grade").sort_values("viscosity_cp")
    return [(str(g), float(v)) for g, v in zip(seen["grade"], seen["v_coded"], strict=True)]


def design_comp(design_points: pd.DataFrame) -> np.ndarray:
    return design_points[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0


@dataclass(frozen=True)
class TernaryView:
    region: ternary.Region
    grids: list[ternary.TernaryGrid]
    #: Measured blends: plot x, y, grade, measured mean.
    points: list[tuple[float, float, str, float]]


def ternary_view(ra: ResponseAnalysis, design_points: pd.DataFrame) -> TernaryView:
    comp = design_comp(design_points)
    reg = ternary.region(comp)
    beta, keep = model_of(ra)
    grids = ternary.grid(reg, comp, beta, keep, ra.spec, grade_levels(design_points))
    xy = reg.to_xy(comp)
    vals = np.asarray(ra.response.values, dtype=float)
    avail = np.asarray(ra.response.available, dtype=bool)
    points = [
        (float(xy[i, 0]), float(xy[i, 1]), str(design_points["grade"].iloc[i]), float(vals[i]))
        for i in range(len(design_points)) if avail[i]
    ]
    return TernaryView(reg, grids, points)


@dataclass(frozen=True)
class PiepelTrace:
    component: str
    x_wt: np.ndarray   # real wt% of the varied component
    y: np.ndarray      # model prediction
    inside: np.ndarray  # within the tested hull


def piepel_traces(
    ra: ResponseAnalysis, design_points: pd.DataFrame, n: int = 41
) -> dict[str, list[PiepelTrace]]:
    """Per grade, one trace per component through the reference (centroid) blend."""
    from scipy.spatial import Delaunay

    comp = design_comp(design_points)
    reg = ternary.region(comp)
    beta, keep = model_of(ra)
    ref = (comp.mean(axis=0) - reg.lower) / reg.span
    try:
        hull: Delaunay | None = Delaunay(comp[:, :2])
    except Exception:
        hull = None
    out: dict[str, list[PiepelTrace]] = {}
    for grade, coded in grade_levels(design_points):
        traces = []
        for i, name in enumerate(ternary.COMPONENTS):
            z = np.linspace(0.0, 1.0, n)
            pseudo = np.zeros((n, 3))
            rest = 1.0 - ref[i]
            for j in range(3):
                pseudo[:, j] = z if j == i else (
                    ref[j] * (1.0 - z) / rest if rest > 1e-9 else (1.0 - z) / 2.0)
            real = reg.from_pseudo(pseudo)
            inside = (hull.find_simplex(real[:, :2]) >= 0) if hull is not None \
                else np.ones(n, dtype=bool)
            y = build_model_matrix(real, np.full(n, coded), ra.spec)[:, keep] @ beta
            traces.append(PiepelTrace(name, real[:, i] * 100.0, y, inside))
        out[grade] = traces
    return out


def fitted_pairs(ra: ResponseAnalysis, design_points: pd.DataFrame
                 ) -> list[tuple[str, int, float, float]]:
    """(grade, case, measured, model-fitted) for every point the model used."""
    avail = np.asarray(ra.response.available, dtype=bool)
    vals = np.asarray(ra.response.values, dtype=float)[avail]
    fitted = np.asarray(ra.fit.fitted, dtype=float)
    if fitted.size != vals.size:
        return []
    rows = design_points[avail]
    return [
        (str(g), int(c), float(m), float(f))
        for g, c, m, f in zip(rows["grade"], rows["case"], vals, fitted, strict=True)
    ]
