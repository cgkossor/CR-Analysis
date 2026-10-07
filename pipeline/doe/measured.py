"""The measured formulations behind a DoE response, for plotting beside the model.

Interaction plots and Cox traces draw what the fitted model predicts along a
slice through one reference composition. Without the data beside them, a
reader cannot tell a model line that follows the measurements from one that
does not, which is exactly how a flat predicted line was once read as
"every formulation releases fully". Every model plot therefore also draws the
measured formulation means, with their replicate spread, from here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pipeline.doe.analysis import ResponseAnalysis


@dataclass(frozen=True)
class MeasuredPoint:
    """One formulation's measured mean for a response."""

    case: int
    grade: str
    api_wt: float
    hpmc_wt: float
    lactose_wt: float
    mean: float
    #: Replicate SD; NaN when there is one replicate or none is on record.
    sd: float
    n: int


def measured_points(
    ra: ResponseAnalysis, design_points: pd.DataFrame, replicates: pd.DataFrame | None
) -> list[MeasuredPoint]:
    """Measured means aligned with ``design_points`` (as the response's values are)."""
    values = np.asarray(ra.response.values, dtype=float)
    available = np.asarray(ra.response.available, dtype=bool)
    if len(values) != len(design_points):
        return []
    col = ra.response.spec.source_column
    spread: dict[tuple[int, str], tuple[float, int]] = {}
    if replicates is not None and col in replicates.columns:
        g = replicates.groupby(["case", "grade"])[col]
        for (case, grade), sd, n in zip(g.std(ddof=1).index, g.std(ddof=1), g.count(),
                                        strict=True):
            spread[(int(case), str(grade))] = (float(sd), int(n))
    out: list[MeasuredPoint] = []
    for i, row in enumerate(design_points.itertuples(index=False)):
        if not available[i]:
            continue
        key = (int(row.case), str(row.grade))
        sd, n = spread.get(key, (float("nan"), 0))
        out.append(MeasuredPoint(
            case=key[0], grade=key[1], api_wt=float(row.api_wt), hpmc_wt=float(row.hpmc_wt),
            lactose_wt=float(row.lactose_wt), mean=float(values[i]), sd=sd, n=n,
        ))
    return out
