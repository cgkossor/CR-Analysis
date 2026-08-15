# Start here (VS Code)

This is an ordinary Python project. Nothing unusual, nothing to configure.

## 0. Get the code

**From GitHub** — no account needed. On the repository page, green **Code**
button → **Download ZIP**. Unzip it.

You will get a folder named something like **`CR-main`** (GitHub appends the
branch name). That folder is the project.

**From a zip sent by email** — unzip it; the folder is called `CR`.

## 1. Open the right folder

**File → Open Folder…** and choose the folder that directly contains
`pyproject.toml` — `CR-main`, *not* the folder you unzipped it into.

This matters. Open the parent by mistake and VS Code will not find `.vscode/`,
so none of the tasks or the test setup appear and nothing below will work.

Sanity check: the Explorer should show `pipeline`, `dashboard`, `tests`,
`pyproject.toml` at the top level.

## 2. Pick a Python interpreter

`Ctrl+Shift+P` → **Python: Select Interpreter** → any Python 3.11 or newer.

If VS Code offers to install the recommended extensions (Python, Pylance, Ruff),
accept. Only the Python extension is strictly required.

## 3. Install the dependencies — once

Either `Ctrl+Shift+P` → **Tasks: Run Task** → **1. Install dependencies (once)**,
or in the integrated terminal (`` Ctrl+` ``):

```
python -m pip install -r requirements.txt
```

Six standard scientific packages: numpy, pandas, scipy, statsmodels, matplotlib,
openpyxl. Needs internet this one time. Everything afterwards runs offline.

## 4. Check it works before touching real data

The repository ships with a small synthetic database. Run it:

```
python -m pipeline.run --input "CR_matrix_tablet_dissolution_PLACEHOLDER (1).xlsx"
```

It should finish in well under a minute and print a summary ending with
`Figures -> 9 rendered`. If that works, your setup is correct and any later
problem is about the data, not the install.

Then double-click `dashboard/index.html` to see the result. (Before this first
run the page will say it needs regenerating — the data payload is not committed
to the repository, only produced by the command above.)

## 5. Run it on your own database

Put the `.xlsx` anywhere, then either:

- `Ctrl+Shift+P` → **Tasks: Run Task** → **2. Run analysis on a database**, and
  type the path when prompted; or
- press **F5** and choose **Run pipeline (pick a database)**, which runs it under
  the debugger so you can set breakpoints; or
- just use the terminal:
  ```
  python -m pipeline.run --input "your_database.xlsx"
  ```

All three do the same thing. One command regenerates everything:

| Output | What it is |
|---|---|
| `outputs/reports/` | six analysis reports, Markdown |
| `outputs/figures/` | nine figures, SVG + PNG, with ranked captions |
| `docs/guidelines.md` | written guidelines and limitations |
| `dashboard/data.js` | the dashboard payload |

Then open `dashboard/index.html` — double-click it in the file explorer, or run
**Tasks: Run Task → 3. Open the dashboard**.

**Read `outputs/reports/data_quality.md` first.** It is written before anything
is modelled and tells you what was found, what is missing, and every anomaly
detected. If the database is not what you expected, that report says so.

## Running the tests

`Ctrl+Shift+P` → **Tasks: Run Task → Run tests**, or use VS Code's Testing panel
(the flask icon) — pytest is already configured, so the tests appear with no
setup. Or from the terminal:

```
python -m pytest
```

**Expect around 10 skips on a fresh download. That is correct, not a problem.**
Skipped tests announce their reason:

| Skipped | Why |
|---|---|
| JavaScript parity, dashboard render | need Node.js — `npm install jsdom` to enable |
| Repository safety checks | need a git checkout; a downloaded ZIP has no `.git` |

Nothing silently passes when its dependency is missing.

## What you should not need to change

Swapping the database should require no code edits. Discovered from the file at
load time: timepoint count and spacing, number of formulations, grades and
replicates, column names, and the units of concentration, time and mass. The
model adapts to what the data supports — with only two viscosity grades, for
instance, the quadratic viscosity term is dropped automatically, because two
levels cannot identify a curve.

Anything ambiguous is a loud error naming the problem, never a silent guess. A
concentration column with no unit in its header stops the run, because a
microgram/milligram mix-up would rescale every profile by 1000×.

Two values cannot be read from the file and live in `pipeline/config.py`:

| Setting | Default | Why it is here |
|---|---|---|
| `VESSEL_VOLUME_ML` | 900 | needed to convert concentration → % released; not in the raw file |
| `FALLBACK_GRADE_VISCOSITY_CP` | K100LV/K4M/K100M | used only when the workbook has no Design sheet listing viscosities |

If your method uses a different vessel volume, change it there — it is the one
edit a new study is likely to need.

## Where things live

```
pipeline/
  io/          schema detection, loading, data-quality reporting
  profiles/    kinetic fits, censoring, metric extraction
  design/      model matrices, aliasing, efficiency, prediction variance
  responses/   correlation, clustering, PCA, key-response selection
  surfaces/    least squares, diagnostics, Box–Cox, model reduction
  validation/  leave-one-formulation-out cross-validation
  equivalence/ f2 similarity and equivalence sets
  stress/      reduced-design evaluation
  figures/     figure rendering
  export/      dashboard payload
  run.py       entry point — start reading here
dashboard/     the offline dashboard (open index.html)
tests/
```

`pipeline/run.py` is the place to start reading: it calls each stage in order.

## Sending results onward

```
python -m pipeline.package
```

Builds `dist/CR_workbench_standalone.html` — the whole dashboard as one file
someone can double-click with nothing installed — plus plain-text versions that
survive mail filters. See `README.md`.

## If something goes wrong

**"No module named pipeline"** — you opened the wrong folder. VS Code must be
opened on the folder containing `pyproject.toml`, and the terminal must be in it.
Check with `dir` (Windows) or `ls`; you should see `pipeline/`.

**Tasks and the Testing panel are missing** — same cause. `.vscode/` sits inside
the project folder; open the parent by mistake and VS Code never sees it.

**"python is not recognised"** — Python is not on PATH. Try `py -3` instead of
`python`, or reinstall Python with "Add to PATH" ticked.

**INGEST FAILED: ...** — the pipeline read your file and could not interpret
something specific; the message names it. The common one is a concentration
column with no unit in its header. That is deliberate: guessing between µg/mL
and mg/mL would silently rescale every profile by 1000×.

**Windows says the downloaded files are blocked** — right-click the ZIP →
Properties → Unblock, *before* extracting.

## Before trusting any number

The database that ships with this is **synthetic placeholder data**, and every
report and dashboard page says so at the top. That banner is driven by a marker
detected at load time, so it disappears by itself once you point the pipeline at
a real database. If you have swapped in real data and still see the banner,
check what the workbook's notes sheet says.
