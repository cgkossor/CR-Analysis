"""Pipeline entry point.

    python -m pipeline.run --input "<database.xlsx>"

Regenerates every derived artifact from the raw file in one command (AC14). The
input is opened read-only and never modified (G8); all output goes to
``outputs/`` and ``dashboard/data.js``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.analysis import SURFACE_RESPONSES, Analysis, run_analysis
from pipeline.audit import AuditReport, run_audit, save
from pipeline.design.report import render_markdown as design_markdown
from pipeline.diagnostics import Diagnostics
from pipeline.diagnostics import render_console as diag_console
from pipeline.diagnostics import write as write_diagnostics
from pipeline.export.data_js import build_payload, write_data_js
from pipeline.glossary import render_parameters_md
from pipeline.io.load import load_database
from pipeline.io.quality import render_markdown as quality_markdown
from pipeline.io.schema import SchemaError
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
    parser.add_argument("--input", required=True, help="path to the dissolution workbook")
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


def _disintegration(analysis: Analysis, source: str) -> DisintegrationAnalysis | None:
    """The disintegration section's result, or None when the workbook has no sheet."""
    from pipeline.disintegration.analysis import run_disintegration
    from pipeline.disintegration.load import load_disintegration

    data = load_disintegration(source)
    return run_disintegration(analysis, data) if data is not None else None


def _payload(
    analysis: Analysis,
    stress: StressTest,
    diagnostics: Diagnostics,
    source: str,
    disintegration: DisintegrationAnalysis | None = None,
) -> tuple[dict[str, Any], AuditReport]:
    """The dashboard payload, carrying its own audit for the Admin tab."""
    payload = build_payload(analysis, stress, diagnostics, disintegration)
    audit = run_audit(source, analysis=analysis, stress=stress, payload=payload)
    payload["audit"] = audit.as_payload()
    return payload, audit


def _print_audit(audit: AuditReport, out_root: Path, path: Path | None = None) -> None:
    print(audit.render())
    saved = path or save(audit, out_root / "reports" / "audit.txt")
    print(f"\nAudit saved to {saved}")


def _write_reports(analysis: Analysis, stress: StressTest, reports: Path) -> None:
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

    docs = Path("docs")
    docs.mkdir(exist_ok=True)
    (docs / "parameters.md").write_text(
        render_parameters_md(), encoding="utf-8", newline="\n"
    )
    (docs / "guidelines.md").write_text(
        render_guidelines(analysis, stress), encoding="utf-8", newline="\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return _run(args)
    except Exception:
        # The run died, but the audit isolates each stage and so can still say
        # where. Printed before the traceback propagates.
        if args.audit:
            _print_audit(run_audit(args.input), Path(args.outputs))
        raise


def _run(args: argparse.Namespace) -> int:
    out_root = Path(args.outputs)
    reports = out_root / "reports"

    try:
        db = load_database(args.input)
    except SchemaError as exc:
        print(f"INGEST FAILED: {exc}", file=sys.stderr)
        if args.audit:
            _print_audit(run_audit(args.input), Path(args.outputs))
        return 2

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

    _write_reports(analysis, stress, reports)
    print(f"Reports -> {reports}")

    # Written last so it can see everything, printed first thing a reader needs.
    diagnostics = write_diagnostics(analysis, stress, reports)
    print(f"Diagnostics -> {reports / 'diagnostics.md'} (+ .json)")

    # Optional section: runs only when the workbook carries a Disintegration
    # sheet. Before the payload, so the dashboard gets its tab.
    from pipeline.disintegration.__main__ import run_section as run_disintegration_section

    dt = run_disintegration_section(
        analysis, args.input, out_root, figures=not args.skip_figures,
        result=_disintegration(analysis, args.input),
    )

    payload, audit = _payload(analysis, stress, diagnostics, args.input, dt)
    data_path = write_data_js(payload, Path(args.dashboard) / "data.js")
    print(f"Dashboard data -> {data_path}")

    if not args.skip_figures:
        from pipeline.figures.render import render_all

        figures = render_all(analysis, stress, out_root / "figures")
        print(f"Figures -> {len(figures)} rendered in {out_root / 'figures'}")

    if args.check_determinism:
        first = hashlib.sha256(data_path.read_bytes()).hexdigest()
        # Re-run the whole chain from the raw file: a second pass must reproduce
        # data.js byte for byte, not merely re-serialise the same objects.
        repeat = run_analysis(load_database(args.input))
        repeat_stress = _stress(repeat)
        with tempfile.TemporaryDirectory() as tmp:
            # Recompute the diagnostics as well. Reusing the first run's would
            # exempt them from the check, which is precisely the part most likely
            # to pick up a stray timestamp or dict ordering.
            repeat_diag = write_diagnostics(repeat, repeat_stress, Path(tmp))
            again = write_data_js(
                _payload(
                    repeat, repeat_stress, repeat_diag, args.input,
                    _disintegration(repeat, args.input),
                )[0],
                Path(tmp) / "data.js",
            )
            second = hashlib.sha256(again.read_bytes()).hexdigest()
        if first != second:
            print(f"DETERMINISM FAILED: {first} != {second}", file=sys.stderr)
            return 3
        print(f"Determinism OK: data.js sha256 {first[:16]}... reproduced exactly")

    # Printed last so it is the final thing on screen: anything that would make
    # the run untrustworthy should be the reader's last impression, not scrolled
    # off the top behind a list of written files.
    print()
    print(diag_console(diagnostics))

    if analysis.quality.is_synthetic:
        print(
            "\nNOTE: the database declares itself synthetic placeholder material. "
            "Every output carries the provenance banner; no result below is experimental."
        )
    # Always saved (outputs/ is gitignored); printed only when asked for.
    audit_path = save(audit, reports / "audit.txt")
    if args.audit:
        print()
        _print_audit(audit, out_root, audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
