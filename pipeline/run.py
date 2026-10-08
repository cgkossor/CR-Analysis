"""Pipeline entry point.

    python -m pipeline.run --input "<database.xlsx>"
    python -m pipeline.run --input api_a.xlsx --input api_b.xlsx --apis apis.csv

Regenerates every derived artifact from the raw file in one command (AC14). The
input is opened read-only and never modified (G8); all output goes to
``outputs/`` and ``dashboard/data.js``.

One workbook holds one API. With several ``--input`` files each is analysed on
its own, into ``outputs/<API>/``, and ``data.js`` carries every API for the
dashboard's API selector. With one, the layout is unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.analysis import SURFACE_RESPONSES, Analysis, run_analysis
from pipeline.audit import AuditReport, run_audit, save
from pipeline.data_handling import write as write_data_handling
from pipeline.design.report import render_markdown as design_markdown
from pipeline.diagnostics import Diagnostics
from pipeline.diagnostics import render_console as diag_console
from pipeline.diagnostics import write as write_diagnostics
from pipeline.export.data_js import (
    build_payload,
    figure_gallery,
    paper_gallery,
    write_api_set_js,
    write_data_js,
)
from pipeline.glossary import render_glossary_md, render_parameters_md
from pipeline.io.api_props import load_api_props
from pipeline.io.load import load_database
from pipeline.io.quality import render_markdown as quality_markdown
from pipeline.io.schema import SchemaError
from pipeline.manuscript import build as build_manuscript
from pipeline.manuscript import payload as manuscript_payload
from pipeline.reports import (
    render_doe,
    render_equivalence,
    render_guidelines,
    render_responses,
    render_stress,
    render_surfaces,
)
from pipeline.stress.subsets import StressTest, run_stress_test

if TYPE_CHECKING:
    from pipeline.disintegration.analysis import DisintegrationAnalysis


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipeline.run", description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="path to a dissolution workbook (one API). Repeat for several APIs",
    )
    parser.add_argument("--outputs", default="outputs", help="directory for derived artifacts")
    parser.add_argument(
        "--dashboard", default="dashboard", help="directory holding the static dashboard"
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="run twice and verify byte-identical data.js (G10)",
    )
    parser.add_argument(
        "--skip-figures", action="store_true", help="skip figure rendering"
    )
    parser.add_argument(
        "--disintegration",
        metavar="FILE",
        action="append",
        help=(
            "a separate workbook holding the disintegration data. Without it, a "
            "Disintegration sheet inside --input is used when there is one. With "
            "several --input files give one per input, in the same order, using "
            "'-' for an input that has none"
        ),
    )
    parser.add_argument(
        "--apis",
        metavar="CSV",
        help="per-API properties: columns api, solubility_mg_ml, optional solubility_class",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "print the privacy-safe audit (booleans and integers only) at the end, "
            "or in place of the run if it fails"
        ),
    )
    return parser.parse_args(argv)


def _stress(analysis: Analysis) -> StressTest:
    points = analysis.design_points
    comp = points[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
    proc = points["v_coded"].to_numpy(dtype=float)
    responses = {
        name: points[f"{name}_mean"].to_numpy(dtype=float) for name in SURFACE_RESPONSES
    }
    return run_stress_test(
        comp,
        proc,
        responses,
        analysis.model_spec,
        list(analysis.surfaces["log10_td"].kept_terms),
        time_grid=analysis.time_grid,
        observed_profiles=analysis.observed_profiles,
        case_ids=points["case"].to_numpy(),
        grades=points["grade"].to_numpy(),
        sizes=tuple(range(6, len(points) + 1, 3)),
    )


def _disintegration(
    analysis: Analysis, source: str, dt_file: str | None = None
) -> DisintegrationAnalysis | None:
    """The disintegration section's result, or None when there is no data.

    Read from ``dt_file`` when one is given, otherwise from a Disintegration
    sheet inside the dissolution workbook.
    """
    from pipeline.disintegration.analysis import run_disintegration
    from pipeline.disintegration.load import load_disintegration

    data = load_disintegration(dt_file or source, dedicated=dt_file is not None)
    return run_disintegration(analysis, data) if data is not None else None


def _payload(
    analysis: Analysis,
    stress: StressTest,
    diagnostics: Diagnostics,
    source: str,
    disintegration: DisintegrationAnalysis | None = None,
    dt_file: str | None = None,
    figures: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], AuditReport]:
    """The dashboard payload, carrying its own audit for the Admin tab."""
    payload = build_payload(analysis, stress, diagnostics, disintegration)
    if figures:
        payload["figures"] = figures
    payload["manuscript"] = manuscript_payload(build_manuscript(analysis, disintegration))
    audit = run_audit(
        source, analysis=analysis, stress=stress, payload=payload, disintegration=dt_file
    )
    payload["audit"] = audit.as_payload()
    return payload, audit


def _print_audit(audit: AuditReport, out_root: Path, path: Path | None = None) -> None:
    print(audit.render())
    saved = path or save(audit, out_root / "reports" / "audit.txt")
    print(f"\nAudit saved to {saved}")


def _write_reports(
    analysis: Analysis, stress: StressTest, reports: Path, docs: Path = Path("docs")
) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    synthetic = analysis.quality.is_synthetic

    (reports / "data_quality.md").write_text(
        quality_markdown(analysis.quality), encoding="utf-8", newline="\n"
    )
    (reports / "design_diagnostics.md").write_text(
        design_markdown(
            analysis.design_diagnostics, analysis.lack_of_fit, analysis.design_table, synthetic
        ),
        encoding="utf-8",
        newline="\n",
    )
    (reports / "response_space.md").write_text(
        render_responses(analysis), encoding="utf-8", newline="\n"
    )
    (reports / "doe_analysis.md").write_text(
        render_doe(analysis), encoding="utf-8", newline="\n"
    )
    (reports / "surfaces.md").write_text(
        render_surfaces(analysis), encoding="utf-8", newline="\n"
    )
    (reports / "equivalence.md").write_text(
        render_equivalence(analysis), encoding="utf-8", newline="\n"
    )
    (reports / "stress_test.md").write_text(
        render_stress(analysis, stress), encoding="utf-8", newline="\n"
    )
    (reports / "glossary.md").write_text(render_glossary_md(), encoding="utf-8", newline="\n")

    docs.mkdir(parents=True, exist_ok=True)
    (docs / "parameters.md").write_text(
        render_parameters_md(), encoding="utf-8", newline="\n"
    )
    (docs / "guidelines.md").write_text(
        render_guidelines(analysis, stress), encoding="utf-8", newline="\n"
    )


@dataclass
class ApiRun:
    """One API's finished run, held until data.js is written."""

    api: str
    source: str
    dt_file: str | None
    out_root: Path
    analysis: Analysis
    diagnostics: Diagnostics
    has_dt: bool
    gallery: list[dict[str, Any]] | None
    payload: dict[str, Any]
    audit: AuditReport
    stress: StressTest
    disintegration: Any
    #: The curated manuscript figures, shared by every API in the run.
    paper: list[dict[str, Any]] | None = None


def _dt_files(args: argparse.Namespace) -> list[str | None]:
    """One disintegration file (or None) per --input, in order."""
    given = args.disintegration or []
    if not given:
        return [None] * len(args.input)
    if len(given) != len(args.input):
        raise ValueError(
            f"{len(args.input)} --input file(s) but {len(given)} --disintegration file(s). "
            "Give one per input, in the same order, with '-' where an input has none."
        )
    return [None if f.strip() in ("", "-") else f for f in given]


def _safe_name(api: str) -> str:
    """A folder name for an API label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", api).strip("_") or "API"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _run(args)
    except Exception:
        # The run died, but the audit isolates each stage and so can still say
        # where. Printed before the traceback propagates.
        if args.audit:
            dt = (args.disintegration or [None])[0]
            _print_audit(
                run_audit(args.input[0], disintegration=None if dt == "-" else dt),
                Path(args.outputs),
            )
        raise


def _run(args: argparse.Namespace) -> int:
    try:
        dt_files = _dt_files(args)
        props = load_api_props(args.apis) if args.apis else {}
    except (ValueError, SchemaError) as exc:
        print(f"INPUT FAILED: {exc}", file=sys.stderr)
        return 2

    multi = len(args.input) > 1
    runs: list[ApiRun] = []
    for source, dt_file in zip(args.input, dt_files, strict=True):
        result = _run_one(args, source, dt_file, multi=multi,
                          taken={_safe_name(r.api) for r in runs})
        if isinstance(result, int):
            if runs:
                # Earlier APIs already refreshed their outputs folders, but
                # data.js is only written once every input has run, so it still
                # holds the previous run. Say so rather than let the two mix.
                print(f"Stopped: outputs/ was refreshed for {', '.join(r.api for r in runs)}, "
                      "but dashboard/data.js was NOT updated and still shows the previous "
                      "run.", file=sys.stderr)
            return result
        runs.append(result)

    unknown = sorted(set(props) - {r.api for r in runs})
    if unknown:
        print(f"Note: {args.apis} lists API(s) not run: {', '.join(unknown)}")
    for r in runs:
        if r.api in props:
            r.payload["api_props"] = props[r.api]

    if not args.skip_figures:
        paper = _render_paper(runs, Path(args.outputs), Path(args.dashboard), multi=multi)
        for r in runs:
            r.paper = paper
            _add_paper(r.payload, paper)

    dashboard = Path(args.dashboard) / "data.js"
    data_path = _write_dashboard(runs, props, dashboard)
    print(f"Dashboard data -> {data_path}" + (f" ({len(runs)} APIs)" if multi else ""))

    if args.check_determinism:
        for r in runs:
            code = _check_determinism(r, props)
            if code:
                return code

    for r in runs:
        _finish(args, r, multi=multi)
    return 0


def _write_dashboard(
    runs: list[ApiRun], props: dict[str, dict[str, Any]], path: Path
) -> Path:
    if len(runs) == 1:
        return write_data_js(runs[0].payload, path)
    blank: dict[str, Any] = {"solubility_mg_ml": None, "solubility_class": None}
    return write_api_set_js(
        {r.api: r.payload for r in runs},
        {r.api: props.get(r.api, blank) for r in runs},
        path,
    )


def _run_one(
    args: argparse.Namespace, source: str, dt_file: str | None, *, multi: bool,
    taken: set[str] | None = None,
) -> ApiRun | int:
    """Analyse one API's workbook into its own outputs folder."""
    try:
        db = load_database(source)
    except SchemaError as exc:
        print(f"INGEST FAILED ({source}): {exc}", file=sys.stderr)
        if args.audit:
            _print_audit(run_audit(source, disintegration=dt_file), Path(args.outputs))
        return 2

    api = str(db.profiles["api"].dropna().iloc[0])
    # Checked before anything is written. Folder names, not raw labels: "API 1"
    # and "API_1" would share outputs/API_1 and the second would overwrite.
    if taken and _safe_name(api) in taken:
        print(f"INPUT FAILED: API {api!r} ({source}) appears in more than one --input "
              "workbook, or shares an output folder name with another. Each API is one "
              "workbook with a distinct name.", file=sys.stderr)
        return 2
    out_root = Path(args.outputs) / _safe_name(api) if multi else Path(args.outputs)
    reports = out_root / "reports"
    if multi:
        print(f"\n=== {api} ({source}) -> {out_root} ===")

    analysis = run_analysis(db)
    stress = _stress(analysis)

    print(f"Loaded {analysis.quality.n_ids} formulations x "
          f"{analysis.quality.n_replicates} replicates from {analysis.quality.source_name}")
    print(f"A1: {'mixture' if analysis.quality.mixture.is_mixture else 'not a mixture'} "
          f"(rank {analysis.quality.mixture.rank})")
    print(f"Model: {analysis.model_spec.label} -> "
          f"{len(analysis.surfaces['log10_td'].kept_terms)} terms after reduction")
    print(f"Key responses: {', '.join(analysis.response_space.key_responses)}")
    print(f"CV profile RMSE: {analysis.cross_validation.profile_rmse_pct:.2f}% released, "
          f"median f2 {analysis.cross_validation.median_f2:.1f}")
    print(f"Recommended minimum design: {stress.recommended_size} of "
          f"{len(analysis.design_points)} runs")

    _write_reports(analysis, stress, reports, out_root / "docs" if multi else Path("docs"))
    print(f"Reports -> {reports}")

    # Written last so it can see everything, printed first thing a reader needs.
    diagnostics = write_diagnostics(analysis, stress, reports)
    print(f"Diagnostics -> {reports / 'diagnostics.md'} (+ .json)")
    handling = write_data_handling(analysis, reports)
    print(f"Data handling -> {reports / 'data_handling.md'} (+ .json): "
          f"{handling['totals']['spikes_removed']} spike reading(s) removed, "
          f"{handling['totals']['beyond_window_dropped']} past the window")

    # Figures before the payload, so the dashboard can show them.
    if not args.skip_figures:
        from pipeline.figures.render import render_all

        figures = render_all(analysis, stress, out_root / "figures")
        print(f"Figures -> {len(figures)} rendered in {out_root / 'figures'}")

    # Optional section: runs only when the workbook carries a Disintegration
    # sheet. Before the payload, so the dashboard gets its tab.
    from pipeline.disintegration.__main__ import run_section as run_disintegration_section

    try:
        dt_result = _disintegration(analysis, source, dt_file)
    except SchemaError as exc:
        # A problem in the disintegration data must not cost the dissolution
        # analysis. Say so loudly and carry on without the section.
        print(f"DISINTEGRATION SKIPPED (code {exc.code}): {exc}", file=sys.stderr)
        dt_result = None
    dt = run_disintegration_section(
        analysis, dt_file or source, out_root,
        figures=not args.skip_figures, result=dt_result,
    ) if dt_result is not None else None

    # The manuscript questions read the finished analysis and disintegration
    # sections; their figures must exist before the gallery is listed.
    from pipeline.manuscript import render as render_manuscript
    from pipeline.manuscript import write_report as write_storyline

    manuscript = build_manuscript(analysis, dt)
    write_storyline(manuscript, out_root / "reports" / "storyline.md")
    if not args.skip_figures:
        ms_figures = render_manuscript(
            manuscript, analysis, dt, out_root / "figures" / "manuscript"
        )
        print(f"Manuscript figures -> {len(ms_figures)} rendered")

    gallery = (
        None if args.skip_figures
        else figure_gallery(
            out_root / "figures", Path(args.dashboard), disintegration=dt is not None
        )
    )
    payload, audit = _payload(analysis, stress, diagnostics, source, dt, dt_file, gallery)
    return ApiRun(api, source, dt_file, out_root, analysis, diagnostics, dt is not None,
                  gallery, payload, audit, stress, dt)


def _render_paper(
    runs: list[ApiRun], outputs: Path, dashboard: Path, *, multi: bool
) -> list[dict[str, Any]]:
    """The curated main figures across every API in the run.

    With several APIs they go to outputs/paper/, beside the per-API folders;
    with one, to outputs/figures/paper/.
    """
    from pipeline.manuscript import build as build_ms
    from pipeline.paper import PaperInput, render_paper

    inputs = [PaperInput(r.api, r.analysis, r.stress, r.disintegration,
                         build_ms(r.analysis, r.disintegration)) for r in runs]
    out = outputs / "paper" if multi else outputs / "figures" / "paper"
    records = render_paper(inputs, out, {r.api: r.out_root / "figures" for r in runs})
    print(f"Manuscript main figures -> {len(records)} in {out}")
    return paper_gallery(out, dashboard)


def _add_paper(payload: dict[str, Any], paper: list[dict[str, Any]] | None) -> None:
    """Put the curated main figures first in the Figures tab."""
    if paper:
        payload["figures"] = list(paper) + list(payload.get("figures") or [])


def _check_determinism(run: ApiRun, props: dict[str, dict[str, Any]]) -> int:
    """Re-run the whole chain from the raw file; the payload must match byte for byte."""
    with tempfile.TemporaryDirectory() as tmp:
        first_path = write_data_js(run.payload, Path(tmp) / "first.js")
        first = hashlib.sha256(first_path.read_bytes()).hexdigest()
        repeat = run_analysis(load_database(run.source))
        repeat_stress = _stress(repeat)
        # Recompute the diagnostics as well. Reusing the first run's would
        # exempt them from the check, which is precisely the part most likely
        # to pick up a stray timestamp or dict ordering.
        repeat_diag = write_diagnostics(repeat, repeat_stress, Path(tmp))
        payload = _payload(
            repeat, repeat_stress, repeat_diag, run.source,
            _disintegration(repeat, run.source, run.dt_file) if run.has_dt else None,
            run.dt_file,
            run.gallery,
        )[0]
        if run.api in props:
            payload["api_props"] = props[run.api]
        _add_paper(payload, run.paper)
        again = write_data_js(payload, Path(tmp) / "again.js")
        second = hashlib.sha256(again.read_bytes()).hexdigest()
    if first != second:
        print(f"DETERMINISM FAILED ({run.api}): {first} != {second}", file=sys.stderr)
        return 3
    print(f"Determinism OK ({run.api}): payload sha256 {first[:16]}... reproduced exactly")
    return 0


def _finish(args: argparse.Namespace, run: ApiRun, *, multi: bool) -> None:
    # Printed last so it is the final thing on screen: anything that would make
    # the run untrustworthy should be the reader's last impression, not scrolled
    # off the top behind a list of written files.
    print()
    if multi:
        print(f"=== {run.api} ===")
    print(diag_console(run.diagnostics))

    if run.analysis.quality.is_synthetic:
        print(
            "\nNOTE: the database declares itself synthetic placeholder material. "
            "Every output carries the provenance banner; no result below is experimental."
        )
    # Always saved (outputs/ is gitignored); printed only when asked for.
    audit_path = save(run.audit, run.out_root / "reports" / "audit.txt")
    if args.audit:
        print()
        _print_audit(run.audit, run.out_root, audit_path)


if __name__ == "__main__":
    raise SystemExit(main())
