"""Contour and 3D surface grids over the tested composition region.

Plotted in API x HPMC with lactose as the balance to 100. That is a faithful
picture of the mixture -- two components determine the third -- and it is far
easier to read off a formulation from than a ternary triangle.

Everything outside the convex hull of the tested compositions is masked rather
than extrapolated (G3). The blank region is the point: it shows where the design
actually has information, which a filled rectangle would hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Delaunay

from pipeline.design.matrix import ModelSpec, build_model_matrix


@dataclass(frozen=True)
class SurfaceGrid:
    """Fitted response over the composition plane at one grade."""

    grade: str
    viscosity_cp: float
    api_axis: tuple[float, ...]
    hpmc_axis: tuple[float, ...]
    z: tuple[tuple[float | None, ...], ...]
    z_min: float
    z_max: float
    design_points: tuple[tuple[float, float, float], ...]
    masked_fraction: float
    notes: tuple[str, ...] = field(default=())


def _hull(points: np.ndarray) -> Delaunay | None:
    try:
        return Delaunay(points)
    except Exception:  # pragma: no cover - degenerate design
        return None


def build_grids(
    beta: np.ndarray,
    keep: list[int],
    spec: ModelSpec,
    design_comp: np.ndarray,
    grades: list[tuple[str, float, float]],
    observed: dict[tuple[str], list[tuple[float, float, float]]] | None = None,
    resolution: int = 60,
) -> list[SurfaceGrid]:
    """Evaluate the fitted surface on a grid, masked to the tested region.

    ``design_comp`` is the tested compositions as fractions, ``grades`` is
    ``(label, coded process value, viscosity)`` per grade.
    """
    api = design_comp[:, 0] * 100.0
    hpmc = design_comp[:, 1] * 100.0
    pad_a = max((api.max() - api.min()) * 0.04, 0.5)
    pad_h = max((hpmc.max() - hpmc.min()) * 0.04, 0.5)

    api_axis = np.linspace(api.min() - pad_a, api.max() + pad_a, resolution)
    hpmc_axis = np.linspace(hpmc.min() - pad_h, hpmc.max() + pad_h, resolution)
    tri = _hull(np.column_stack([api, hpmc]))

    mesh_a, mesh_h = np.meshgrid(api_axis, hpmc_axis, indexing="xy")
    flat_a = mesh_a.ravel()
    flat_h = mesh_h.ravel()
    flat_l = 100.0 - flat_a - flat_h

    inside = np.ones(flat_a.shape, dtype=bool)
    if tri is not None:
        inside &= tri.find_simplex(np.column_stack([flat_a, flat_h])) >= 0
    # Lactose must remain a real, non-negative component.
    inside &= flat_l >= -1e-9

    comps = np.column_stack([flat_a, flat_h, np.clip(flat_l, 0.0, None)]) / 100.0

    out: list[SurfaceGrid] = []
    for label, coded, viscosity in grades:
        matrix = build_model_matrix(comps, np.full(len(comps), coded), spec)[:, keep]
        z = matrix @ beta
        z_masked = np.where(inside, z, np.nan)

        finite = z_masked[np.isfinite(z_masked)]
        grid_rows = z_masked.reshape(mesh_a.shape)

        points = [
            (float(a), float(h), float(100.0 - a - h))
            for a, h in zip(api, hpmc, strict=True)
        ]

        out.append(
            SurfaceGrid(
                grade=label,
                viscosity_cp=viscosity,
                api_axis=tuple(float(v) for v in api_axis),
                hpmc_axis=tuple(float(v) for v in hpmc_axis),
                z=tuple(
                    tuple(None if not np.isfinite(v) else float(v) for v in row)
                    for row in grid_rows
                ),
                z_min=float(finite.min()) if finite.size else float("nan"),
                z_max=float(finite.max()) if finite.size else float("nan"),
                design_points=tuple(points),
                masked_fraction=float(1.0 - inside.mean()),
                notes=(
                    "Blank area lies outside the convex hull of the tested "
                    "compositions. No value is drawn there because predicting it "
                    "would be extrapolation.",
                ),
            )
        )
    return out


def shared_scale(grids: list[SurfaceGrid]) -> tuple[float, float]:
    """One colour scale across grades, so the panels are comparable.

    Scaling each grade to its own range makes every panel look equally variable
    and hides the thing worth seeing: that one grade spans far more of the
    response than another.
    """
    lows = [g.z_min for g in grids if np.isfinite(g.z_min)]
    highs = [g.z_max for g in grids if np.isfinite(g.z_max)]
    if not lows or not highs:
        return (0.0, 1.0)
    return (min(lows), max(highs))
