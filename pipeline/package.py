"""Build email-safe deliverables.

    python -m pipeline.package

There is nothing to deploy: the dashboard is a file, not a service. Whoever
receives it opens it. The only real problem is getting bytes past a mail filter,
which strips ``.py``, ``.js``, ``.zip`` and often ``.html`` too.

So everything is also emitted as plain ``.txt``, which nothing blocks. Two
audiences, two packages:

* **"just let them see the results"** -> one self-contained HTML file, plus a
  ``.txt`` fallback of the same file for when HTML is blocked;
* **"they need to re-run it on new data"** -> the whole project as a ``.txt``.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import textwrap
import zipfile
from pathlib import Path

#: Directories never packaged: regenerable, machine-specific, or huge.
EXCLUDE_DIRS = {
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    "dist",
    ".venv",
    "venv",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAMES = {"package-lock.json"}

#: Line width for base64 payloads. Mail transfer agents may hard-wrap long
#: lines; wrapping first means the wrap is ours and is reversible.
B64_WIDTH = 76


def _should_include(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
        return False
    return path.is_file()


def build_standalone_html(dashboard: Path, out: Path) -> Path:
    """Inline styles, data, model and app into one self-contained HTML file."""
    html = (dashboard / "index.html").read_text(encoding="utf-8")

    css = (dashboard / "styles.css").read_text(encoding="utf-8")
    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        "<style>\n" + css + "\n</style>",
    )

    scripts = ""
    for name in ("data.js", "model.js", "doe.js", "formulator.js", "audit.js", "app.js"):
        source = dashboard / name
        if not source.exists():
            raise FileNotFoundError(
                f"{name} is missing. Run `python -m pipeline.run --input <file>` first "
                "so the dashboard payload exists."
            )
        body = source.read_text(encoding="utf-8")
        # A literal </script> inside a string would end the block early.
        body = body.replace("</script>", "<\\/script>")
        scripts += f"<script>\n{body}\n</script>\n"

    for name in ("data.js", "model.js", "doe.js", "formulator.js", "audit.js", "app.js"):
        html = html.replace(f'<script src="{name}"></script>\n', "")
        html = html.replace(f'<script src="{name}"></script>', "")
    html = html.replace("</body>", scripts + "</body>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    return out


def build_zip(root: Path, out: Path) -> Path:
    """Zip the project, excluding regenerable and machine-specific files."""
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p for p in root.rglob("*") if _should_include(p, root)),
        key=lambda p: str(p.relative_to(root)).replace("\\", "/"),
    )
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            arc = "CR/" + str(path.relative_to(root)).replace("\\", "/")
            # Fixed timestamp so the archive is reproducible (G10).
            info = zipfile.ZipInfo(arc, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return out


def encode_to_text(source: Path, out: Path) -> tuple[Path, str]:
    """Base64-encode a file into wrapped plain text. Returns the path and SHA-256."""
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(raw).decode("ascii")
    wrapped = "\n".join(textwrap.wrap(encoded, B64_WIDTH))
    out.write_text(wrapped + "\n", encoding="utf-8", newline="\n")
    return out, digest


#: Backwards-compatible alias used by the tests and by callers that only ever
#: encode the project archive.
def build_base64(zip_path: Path, out: Path) -> tuple[Path, str]:
    return encode_to_text(zip_path, out)


def _decode_command(text_name: str, output_name: str) -> str:
    """A single copy-pasteable Python command, assembled to keep source lines short."""
    return (
        'python -c "import base64 as b, pathlib as p; '
        f"p.Path('{output_name}').write_bytes("
        f"b.b64decode(p.Path('{text_name}').read_text()))\""
    )


VIEW_ONLY = """\
CR Matrix Tablet Workbench - how to open it
===========================================

There is nothing to install and nothing to run. This is a file, not a program.


IF YOU RECEIVED CR_workbench_standalone.html
--------------------------------------------

Save it anywhere and double-click it. It opens in your browser. Done.

It works with no internet connection. Everything - the data, the charts, the
prediction model - is inside that one file.


IF THE .html WAS BLOCKED AND YOU RECEIVED CR_dashboard_base64.txt
-----------------------------------------------------------------

Your mail system stripped the HTML, so it was sent as plain text instead. Turn
it back into a web page with ONE of these, run in the folder containing the .txt:

Windows (PowerShell):

    $b64 = (Get-Content .\\CR_dashboard_base64.txt -Raw) -replace '\\s',''
    [IO.File]::WriteAllBytes("$PWD\\CR_workbench.html", [Convert]::FromBase64String($b64))

macOS / Linux:

    tr -d '\\n' < CR_dashboard_base64.txt | base64 -d > CR_workbench.html

Anywhere with Python:

    {dashboard_cmd}

Then double-click CR_workbench.html.

To confirm it arrived intact, the SHA-256 of the .html should be:

    {dashboard_digest}


WHAT YOU ARE LOOKING AT
-----------------------

Eight tabs. The database and every measured profile; the release metrics; the
design diagnostics; the fitted response surfaces; the iso-release equivalence
sets; the reduced-design stress test; a formulator tool for forward prediction
and inverse design; and the guidelines with an explicit list of what the data
does NOT support.

Every prediction shows its cross-validated error and links back to the measured
experiments behind it.
"""


REBUILD = """\
CR Matrix Tablet Workbench - unpack and re-run on new data
==========================================================

You do NOT need to open, read, or paste the contents of the .txt file anywhere.
It is just a container. One command turns it back into an ordinary folder.

What you need:
  - Python 3.11 or newer  (check with:  python --version)
  - An internet connection ONCE, to install six standard packages.
    After that, everything runs offline.


STEP 1 - turn the .txt back into a folder
------------------------------------------

Save CR_project_base64.txt somewhere, open a terminal in that folder, and run
ONE of these. You will end up with a folder called CR.

Windows (PowerShell):

    $b64 = (Get-Content .\\CR_project_base64.txt -Raw) -replace '\\s',''
    [IO.File]::WriteAllBytes("$PWD\\CR.zip", [Convert]::FromBase64String($b64))
    Expand-Archive .\\CR.zip -DestinationPath .

macOS / Linux:

    tr -d '\\n' < CR_project_base64.txt | base64 -d > CR.zip
    unzip CR.zip

Anywhere with Python:

    {project_cmd}
    python -c "import zipfile; zipfile.ZipFile('CR.zip').extractall('.')"

Optional integrity check - the SHA-256 of CR.zip should be:

    {project_digest}

    PowerShell:  Get-FileHash .\\CR.zip -Algorithm SHA256
    macOS/Linux: shasum -a 256 CR.zip


USING VS CODE? Open the CR folder and read START_HERE.md
---------------------------------------------------------

If you already work in VS Code, that is the natural way to use this. It is an
ordinary Python project. File -> Open Folder -> CR, then follow START_HERE.md.

The .vscode folder is already set up: pytest is configured, F5 runs the pipeline
under the debugger and prompts for the database path, and Tasks: Run Task gives
you install / run / open-dashboard / test as one-click actions.

The remaining steps below are the plain-terminal equivalents.


STEP 2 - install the packages (once)
-------------------------------------

    cd CR
    python -m pip install -r requirements.txt

Six well-known scientific packages: numpy, pandas, scipy, statsmodels,
matplotlib, openpyxl. Nothing unusual, nothing from outside PyPI.


STEP 3 - run it on your data
-----------------------------

Put your database .xlsx anywhere, then:

    python -m pipeline.run --input "path/to/your_database.xlsx"

That is the whole thing. It regenerates, from the raw file:

    outputs/reports/     six analysis reports
    outputs/figures/     figures (PNG) with captions; headlines/ = slide set
    docs/guidelines.md   the written guidelines and limitations
    dashboard/data.js    the dashboard payload

Then open CR/dashboard/index.html by double-clicking it.

To email the refreshed dashboard onward:

    python -m pipeline.package

...which rebuilds dist/CR_workbench_standalone.html and the .txt versions.


What the pipeline works out on its own
--------------------------------------

You should not need to edit any code to swap the database. Discovered from the
file at load time: the number of timepoints and their spacing, the number of
formulations, grades and replicates, the column names, and the units of
concentration, time and mass.

It adapts the model to what the data can support - for example, with only two
viscosity grades it automatically drops the quadratic viscosity term, because
two levels cannot identify a curve.

Anything it cannot interpret is a loud error naming the problem, never a silent
guess. A concentration column with no unit in its header stops the run, because
a microgram/milligram mix-up would rescale every profile by 1000x.

Two things it cannot read from the file and takes from pipeline/config.py:

    VESSEL_VOLUME_ML             900 mL, needed to convert concentration to
                                 % released
    FALLBACK_GRADE_VISCOSITY_CP  grade -> nominal viscosity, used only if the
                                 workbook has no Design sheet listing them

Check the run was sane: outputs/reports/data_quality.md is written first and
lists what was found, what is missing, and every anomaly detected.

Confirm the code itself still works on your machine:

    python -m pip install pytest
    python -m pytest
"""


EMAIL_TEMPLATE = """\
Paste this into the email body
==============================

Subject: Controlled-release matrix tablet DoE - analysis and formulator tool

---

Attached is the analysis of the controlled-release matrix tablet DoE, as an
interactive workbench.

To open it: save CR_workbench_standalone.html and double-click it. It opens in
your browser. Nothing to install, no internet needed - the data, charts and
prediction model are all inside that one file.

If your mail system strips the .html attachment, use CR_dashboard_base64.txt
instead and follow CR_HOW_TO_OPEN.txt to turn it back into a web page.

Eight tabs: the measured database, release metrics, design diagnostics, fitted
response surfaces, iso-release equivalence sets, a reduced-design stress test, a
formulator tool for forward prediction and inverse design, and the guidelines -
including an explicit statement of what the data does not support.

Every prediction carries its cross-validated error and links back to the
measured experiments behind it.

{synthetic_note}
If you want to re-run the analysis on a different database, also attached is
CR_project_base64.txt with instructions in CR_HOW_TO_REBUILD.txt.

---
"""

SYNTHETIC_NOTE = """\
One important caveat: the database behind this is SYNTHETIC placeholder data
used to build and test the pipeline. Nothing in it is an experimental
measurement, and no result should inform a formulation decision. The dashboard
says so at the top of every page. It will be replaced by the real database.

"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline.package", description=__doc__)
    parser.add_argument("--root", default=".", help="project root to package")
    parser.add_argument("--out", default="dist", help="output directory")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    dist = Path(args.out).resolve()
    dist.mkdir(parents=True, exist_ok=True)

    standalone = build_standalone_html(root / "dashboard", dist / "CR_workbench_standalone.html")
    dash_txt, dash_digest = encode_to_text(standalone, dist / "CR_dashboard_base64.txt")

    zip_path = build_zip(root, dist / "CR.zip")
    proj_txt, proj_digest = encode_to_text(zip_path, dist / "CR_project_base64.txt")

    (dist / "CR_HOW_TO_OPEN.txt").write_text(
        VIEW_ONLY.format(
            dashboard_cmd=_decode_command("CR_dashboard_base64.txt", "CR_workbench.html"),
            dashboard_digest=dash_digest,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (dist / "CR_HOW_TO_REBUILD.txt").write_text(
        REBUILD.format(
            project_cmd=_decode_command("CR_project_base64.txt", "CR.zip"),
            project_digest=proj_digest,
        ),
        encoding="utf-8",
        newline="\n",
    )

    is_synthetic = '"is_synthetic": true' in (
        (root / "dashboard" / "data.js").read_text(encoding="utf-8")
    )
    (dist / "EMAIL_TEMPLATE.txt").write_text(
        EMAIL_TEMPLATE.format(synthetic_note=SYNTHETIC_NOTE if is_synthetic else ""),
        encoding="utf-8",
        newline="\n",
    )

    def size(path: Path) -> str:
        n = path.stat().st_size
        return f"{n / 1_048_576:.1f} MB" if n >= 1_048_576 else f"{n / 1024:.0f} KB"

    print("Built in", dist)
    print()
    print("=" * 68)
    print("TO SEND: they only need to LOOK at the results")
    print("=" * 68)
    print(f"  attach  CR_workbench_standalone.html   {size(standalone):>8}")
    print(f"  attach  CR_HOW_TO_OPEN.txt             {size(dist / 'CR_HOW_TO_OPEN.txt'):>8}")
    print(
        f"  attach  CR_dashboard_base64.txt        {size(dash_txt):>8}"
        "   (only if .html is blocked)"
    )
    print("  They double-click the .html. Nothing to install.")
    print()
    print("=" * 68)
    print("TO SEND: they need to RE-RUN it on a different database")
    print("=" * 68)
    print(f"  attach  CR.zip                         {size(zip_path):>8}   (try this first)")
    print(f"  attach  CR_project_base64.txt          {size(proj_txt):>8}   (if the zip is blocked)")
    print(f"  attach  CR_HOW_TO_REBUILD.txt          {size(dist / 'CR_HOW_TO_REBUILD.txt'):>8}")
    print("  They need Python 3.11+ and internet once, for six standard packages.")
    print()
    print("Email body to paste:  EMAIL_TEMPLATE.txt")
    print()
    print(f"  standalone .html sha256 : {dash_digest}")
    print(f"  project zip     sha256  : {proj_digest}")
    if is_synthetic:
        print()
        print("NOTE: the data is synthetic placeholder material. The email template")
        print("      says so explicitly - keep that paragraph in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
