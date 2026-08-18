"""Assemble the named responses the DoE analysis is fitted on.

These are the quantities a formulator names, not the parameters a curve-fitter
needs. Each carries its units, its direction of goodness, and whether censoring
can make it undefined -- because two of them can, and a response that silently
goes missing on the slowest formulations would bias every conclusion toward the
fast ones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ResponseSpec:
    """One modelled response."""

    key: str
    label: str
    units: str
    source_column: str
    censorable: bool = False
    note: str = ""


#: The responses fitted with the full classical treatment. Chosen with the user:
#: release timing, the fixed-timepoint percentages dissolution specifications are
#: actually written against, overall rate, and completeness.
RESPONSES: tuple[ResponseSpec, ...] = (
    ResponseSpec(
        "t50", "Time to 50% released", "h", "t50", censorable=True,
        note="Undefined for a formulation that never reaches 50%. Those points are "
             "excluded from this response's model and named in the diagnostics; they "
             "are not silently dropped.",
    ),
    ResponseSpec(
        "t80", "Time to 80% released", "h", "t80", censorable=True,
        note="Censored for the slow formulations. % released at 24 h carries the same "
             "completeness information without going undefined, so read that alongside.",
    ),
    ResponseSpec("pct_1h", "% released at 1 h", "% of dose", "pct_1h"),
    ResponseSpec("pct_2h", "% released at 2 h", "% of dose", "pct_2h"),
    ResponseSpec("pct_4h", "% released at 4 h", "% of dose", "pct_4h"),
    ResponseSpec("pct_8h", "% released at 8 h", "% of dose", "pct_8h"),
    ResponseSpec("pct_12h", "% released at 12 h", "% of dose", "pct_12h"),
    ResponseSpec(
        "pct_24h", "% released at 24 h", "% of dose", "pct_24h",
        note="Always defined, so this is the completeness response to trust when t80 "
             "is censored.",
    ),
    ResponseSpec(
        "mdt_h", "Mean dissolution time", "h", "mdt_h",
        note="Integrates the whole curve rather than one crossing. A lower bound for "
             "profiles still rising at the last measurement.",
    ),
)


@dataclass(frozen=True)
class ResponseData:
    """Values of one response across the design, with its coverage recorded."""

    spec: ResponseSpec
    values: np.ndarray
    available: np.ndarray
    n_total: int
    n_missing: int
    missing_points: tuple[str, ...]

    @property
    def usable(self) -> bool:
        """Enough points to fit anything at all."""
        return int(np.sum(self.available)) >= 4

    @property
    def coverage_note(self) -> str:
        if not self.n_missing:
            return ""
        named = ", ".join(self.missing_points[:6])
        more = f" and {self.n_missing - 6} more" if self.n_missing > 6 else ""
        return (
            f"{self.n_missing} of {self.n_total} design points have no value for this "
            f"response ({named}{more}). They are excluded from this model only. Because "
            "the missing ones are the slowest formulations rather than a random subset, "
            "conclusions drawn here describe the faster part of the design space."
        )


def collect(design_points: pd.DataFrame, replicates: pd.DataFrame) -> list[ResponseData]:
    """Build every response from the per-replicate metrics, averaged per design point."""
    keys = ["case", "grade"]
    available_cols = [r.source_column for r in RESPONSES if r.source_column in replicates.columns]
    means = replicates.groupby(keys, sort=True)[available_cols].mean().reset_index()

    merged = design_points[keys].merge(means, on=keys, how="left")
    labels = [f"case {int(c)}/{g}" for c, g in zip(merged["case"], merged["grade"], strict=True)]

    out: list[ResponseData] = []
    for spec in RESPONSES:
        if spec.source_column not in merged.columns:
            continue
        values = merged[spec.source_column].to_numpy(dtype=float)
        available = np.isfinite(values)
        missing = tuple(labels[i] for i in range(len(values)) if not available[i])
        out.append(
            ResponseData(
                spec=spec,
                values=values,
                available=available,
                n_total=len(values),
                n_missing=len(missing),
                missing_points=missing,
            )
        )
    return out
