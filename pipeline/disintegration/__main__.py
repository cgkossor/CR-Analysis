"""Run the disintegration section on its own.

    python -m pipeline.disintegration --input <workbook.xlsx> [--outputs outputs]
        [--skip-figures] [--formats png,pdf,svg,tiff]

Writes ``<outputs>/reports/disintegration.md``, the diagnostics (``.md`` +
``.json``), and DT-01 … DT-08 in ``<outputs>/figures/disintegration/``. The
workbook is opened read-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.analysis import Analysis
from pipeline.diagnostics import render_console
from pipeline.disintegration.analysis import DisintegrationAnalysis, run_disintegration
from pipeline.disintegration.load import load_disintegration
from pipeline.io.schema import SchemaError

_FORMATS = ("png", "pdf", "svg", "tiff")


def run_section(
    analysis: Analysis,
    source: str | Path,
    out_root: Path,
    *,
    figures: bool = True,
    formats: tuple[str, ...] = ("png",),
    result: DisintegrationAnalysis | None = None,
) -> DisintegrationAnalysis | None:
    """Load, analyse and write everything; ``None`` when there is no DT sheet.

    Pass ``result`` when the caller has already run the analysis (pipeline.run
    does, to put it in the dashboard payload), so it is not computed twice.
    """
    from pipeline.disintegration import diagnostics, glossary, report

    if result is None:
        data = load_disintegration(source)
        if data is None:
            return None
        result = run_disintegration(analysis, data)
    r = result
    reports = out_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    captions: list[tuple[str, str]] = []
    if figures:
        from pipeline.disintegration.figures import render_figures

        fig_dir = out_root / "figures" / "disintegration"
        records = render_figures(r, fig_dir, formats)
        captions = [(rec.id, rec.caption) for rec in records]
        print(f"Disintegration figures -> {len(records)} rendered in {fig_dir}")

    (reports / "disintegration.md").write_text(
        report.render(r, captions) + glossary.render_markdown(), encoding="utf-8", newline="\n"
    )
    d = diagnostics.write(r, reports)
    print(f"Disintegration report -> {reports / 'disintegration.md'}")
    print(f"Disintegration diagnostics -> {reports / 'disintegration_diagnostics.md'} (+ .json)")
    print("Disintegration: " + render_console(d))
    return r


def _formats(text: str) -> tuple[str, ...]:
    chosen = tuple(f.strip().lower() for f in text.split(",") if f.strip())
    bad = [f for f in chosen if f not in _FORMATS]
    if bad or not chosen:
        raise argparse.ArgumentTypeError(f"formats must be from {_FORMATS}, got {text!r}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline.disintegration",
                                     description=__doc__)
    parser.add_argument("--input", required=True, help="workbook with a Disintegration sheet")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--formats", type=_formats, default=("png",),
                        help="comma-separated figure formats: png,pdf,svg,tiff")
    args = parser.parse_args(argv)

    from pipeline.analysis import run_analysis
    from pipeline.io.load import load_database

    try:
        analysis = run_analysis(load_database(args.input))
        r = run_section(analysis, args.input, Path(args.outputs),
                        figures=not args.skip_figures, formats=args.formats)
    except SchemaError as exc:
        print(f"INGEST FAILED (code {exc.code}): {exc}", file=sys.stderr)
        return 2
    if r is None:
        print("No Disintegration sheet in this workbook; nothing to do.")
        return 1
    if r.is_synthetic:
        print("NOTE: synthetic placeholder data. No result is experimental.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
