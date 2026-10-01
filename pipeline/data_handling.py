"""What happened to the raw readings on their way into the analysis and the plots.

Every step that removes, skips or fills in a reading is accounted for here, per
replicate where it matters, so a number in a figure can be traced back to the
readings behind it:

1. **Spike filter** (load): readings removed as momentary spikes, each listed
   with the local median it was judged against.
2. **Analysis window** (load): readings past :data:`config.ANALYSIS_WINDOW_H`.
3. **Comparison grid**: whether profiles were resampled onto the nominal
   schedule, and how many grid points were left blank because the readings
   either side were too far apart.
4. **Plot thinning**: how many readings each measured-profile plot draws.
5. **Response coverage**: which formulations each DoE response leaves out
   (t50/t80 are undefined for runs that never reach 50/80 %), which is the
   usual reason a response "loses" the slow runs.

Written to ``reports/data_handling.md`` and ``.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.analysis import Analysis

#: Rows of the spike table printed in the Markdown; the JSON carries them all.
MAX_SPIKE_ROWS = 60


def _plot_points(analysis: Analysis) -> pd.DataFrame:
    from pipeline.figures.render import measured_profiles

    rows = []
    reps = analysis.db.profiles.groupby(["case", "grade", "replicate"]).size()
    for (case, grade), prof in sorted(measured_profiles(analysis).items()):
        rows.append({
            "case": case,
            "grade": grade,
            "readings_per_replicate": int(reps.loc[(case, grade)].median()),
            "points_plotted": int(np.isfinite(prof.mean).sum()),
            "error_bars": int((prof.bars & np.isfinite(prof.mean)).sum()),
        })
    return pd.DataFrame(rows)


def build(analysis: Analysis) -> dict[str, Any]:
    """Collect every data-handling count into one plain dictionary."""
    db = analysis.db
    readings = db.readings_log if db.readings_log is not None else pd.DataFrame()
    spikes = db.spike_log if db.spike_log is not None else pd.DataFrame()
    grid = analysis.time_grid_info

    blanks = {
        f"case {c} / {g}": int((~np.isfinite(v)).sum())
        for (c, g), v in sorted(analysis.observed_profiles.items())
        if (~np.isfinite(v)).any()
    }
    coverage = [
        {
            "response": ra.response.spec.key,
            "label": ra.response.spec.label,
            "used": int(ra.response.available.sum()),
            "total": int(ra.response.n_total),
            "left_out": list(ra.response.missing_points),
        }
        for ra in analysis.doe.responses
    ]
    loaded = int(readings["loaded"].sum()) if not readings.empty else len(db.profiles)
    return {
        "settings": {
            "spike_filter": bool(config.SPIKE_FILTER),
            "spike_window_readings": config.SPIKE_WINDOW,
            "spike_min_pct": config.SPIKE_MIN_PCT,
            "spike_noise_k": config.SPIKE_NOISE_K,
            "spike_max_run": config.SPIKE_MAX_RUN,
            "spike_min_readings": config.SPIKE_MIN_READINGS,
            "analysis_window_h": config.ANALYSIS_WINDOW_H,
            "max_interp_gap_h": grid.max_gap_h,
            "plot_point_stride": config.PLOT_POINT_STRIDE,
        },
        "totals": {
            "readings_loaded": loaded,
            "spikes_removed": int(db.spikes_removed),
            "beyond_window_dropped": int(db.rows_beyond_window),
            "readings_analysed": len(db.profiles),
            "replicates": len(readings) if not readings.empty else None,
            "replicates_with_spikes": (
                int((readings["spikes_removed"] > 0).sum()) if not readings.empty else 0
            ),
        },
        "grid": {
            "resampled_to_nominal_schedule": bool(grid.resampled),
            "points": len(analysis.time_grid),
            "blank_grid_points_by_profile": blanks,
        },
        "spikes": spikes.to_dict(orient="records"),
        "replicates": readings.to_dict(orient="records"),
        "plotting": _plot_points(analysis).to_dict(orient="records"),
        "response_coverage": coverage,
    }


def _span(values: pd.Series) -> str:
    if values.empty:
        return "0"
    lo, hi = int(values.min()), int(values.max())
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _table(frame: pd.DataFrame, floats: int = 2) -> str:
    if frame.empty:
        return "_none_\n"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame.itertuples(index=False):
        cells = [
            f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(info: dict[str, Any], synthetic: bool) -> str:
    t, s, g = info["totals"], info["settings"], info["grid"]
    out = ["# Data handling", ""]
    if synthetic:
        out += ["> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.**", ""]
    out += [
        "Every reading that was removed, skipped or filled in between the workbook and "
        "the results. Nothing here changes the data; it records what the pipeline did.",
        "",
        "## Summary",
        "",
        "| step | readings |",
        "|---|---|",
        f"| loaded from the workbook | {t['readings_loaded']} |",
        f"| removed as momentary spikes | {t['spikes_removed']} |",
        f"| dropped past the {s['analysis_window_h']:g} h analysis window | "
        f"{t['beyond_window_dropped']} |",
        f"| used in the analysis | {t['readings_analysed']} |",
        "",
        "## 1. Spike filter",
        "",
        (
            f"On. A reading is removed when it is more than max({s['spike_min_pct']:g} % "
            f"released, {s['spike_noise_k']:g} robust SD of that replicate's noise) away "
            f"from the median of the {s['spike_window_readings']} readings around it, "
            "jumps away from both neighbours and comes back, lasts at most "
            f"{s['spike_max_run']} readings, and is not the first or last reading. "
            f"Replicates with fewer than {s['spike_min_readings']} readings (manual pulls) "
            "are not filtered."
            if s["spike_filter"] else "Off (`config.SPIKE_FILTER = False`)."
        ),
        "",
        f"{t['spikes_removed']} reading(s) removed from {t['replicates_with_spikes']} "
        f"replicate(s).",
        "",
    ]
    spikes = pd.DataFrame(info["spikes"])
    if not spikes.empty:
        shown = spikes.head(MAX_SPIKE_ROWS)
        out += [_table(shown)]
        if len(spikes) > MAX_SPIKE_ROWS:
            out += [f"_{len(spikes) - MAX_SPIKE_ROWS} more in data_handling.json._", ""]
        out += ["See `figures/diagnostics/spikes_removed.png` for the affected runs.", ""]

    reps = pd.DataFrame(info["replicates"])
    out += ["## 2. Readings per replicate", ""]
    if reps.empty:
        out += ["_not recorded_", ""]
    else:
        past = reps["beyond_window"]
        spiked = reps[reps["spikes_removed"] > 0]
        out += [
            f"{len(reps)} replicates, {_span(reps['loaded'])} readings each as loaded. "
            f"{int((past > 0).sum())} replicate(s) ran past the "
            f"{s['analysis_window_h']:g} h window and lost {_span(past[past > 0])} "
            "reading(s) each to the trim. Replicates that lost readings "
            "to the spike filter are listed below; the full per-replicate table is in "
            "data_handling.json.",
            "",
            _table(spiked),
        ]

    out += [
        "## 3. Comparison grid",
        "",
        (
            f"Profiles were resampled onto the {g['points']}-point nominal schedule for "
            "mean curves, f2, cross-validation and the stress test (densely logged data). "
            "Per-profile metrics and fits still use every reading."
            if g["resampled_to_nominal_schedule"]
            else f"Profiles were compared on their own {g['points']} shared sampling times."
        ),
        "",
    ]
    if g["blank_grid_points_by_profile"]:
        out += [
            f"Grid points left blank because the readings either side were more than "
            f"{s['max_interp_gap_h']} h apart:",
            "",
            _table(pd.DataFrame(
                [{"profile": k, "blank_points": v}
                 for k, v in g["blank_grid_points_by_profile"].items()]
            )),
        ]
    else:
        out += ["No grid point was left blank.", ""]

    plot = pd.DataFrame(info["plotting"])
    out += [
        "## 4. Plot thinning",
        "",
        f"Measured-profile plots draw every {s['plot_point_stride']}th reading of densely "
        "logged runs (always keeping the last), with error bars at the nominal schedule "
        "times. Manual pulls are drawn in full. This affects the plots only, never the "
        "analysis.",
        "",
        _table(plot),
        "## 5. Formulations each response leaves out",
        "",
        "A response is undefined for a run that never reaches it: t80 needs 80 % "
        "released, so the slowest runs drop out of the t80 model. Percent-released "
        "responses are defined for every run.",
        "",
    ]
    cov = pd.DataFrame(
        [{"response": c["label"], "used": f"{c['used']} of {c['total']}",
          "left out": ", ".join(c["left_out"]) or "none"} for c in info["response_coverage"]]
    )
    out += [_table(cov)]
    return "\n".join(out)


def _plain(value: Any) -> Any:
    """JSON fallback for numpy scalars and timestamps."""
    return value.item() if hasattr(value, "item") else str(value)


def _finite(value: Any) -> Any:
    """NaN and infinity become null: JSON has no spelling for them."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write(analysis: Analysis, reports: Path) -> dict[str, Any]:
    info = build(analysis)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "data_handling.md").write_text(
        render_markdown(info, analysis.quality.is_synthetic), encoding="utf-8", newline="\n"
    )
    (reports / "data_handling.json").write_text(
        json.dumps(_finite(info), indent=1, default=_plain, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    return info
