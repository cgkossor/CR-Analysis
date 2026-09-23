# Controlled-Release Matrix Tablet — DoE Analysis & Formulator Workbench

Python pipeline plus a static, offline dashboard for a controlled-release HPMC
matrix tablet DoE: 11 compositions × 3 viscosity grades per API.

> **The database currently in the repo declares itself synthetic placeholder
> material.** Every generated report, figure and dashboard panel carries a
> provenance banner saying so. The banner is driven by a marker detected at
> ingest, so it disappears on its own when a real database replaces it — no code
> change required.

## Run it

```bash
python -m pipeline.run --input "CR_matrix_tablet_dissolution_PLACEHOLDER (1).xlsx"
```

One command regenerates every artifact from the raw file:

| Output | What it is |
|---|---|
| `outputs/reports/data_quality.md` | AC1 ingestion & validation report |
| `outputs/reports/design_diagnostics.md` | AC3 aliasing, efficiency, FDS, lack-of-fit estimability |
| `outputs/reports/response_space.md` | AC5 dimensionality and key-response selection |
| `outputs/reports/surfaces.md` | AC4 surfaces, Box–Cox, sequential SS, residuals |
| `outputs/reports/equivalence.md` | AC7 iso-release equivalence sets |
| `outputs/reports/stress_test.md` | AC8 reduced-design evaluation and recommendation |
| `outputs/figures/` | AC9 journal-style figures (300 dpi PNG) with `captions.json` / `captions.md`; `headlines/` holds the H1–H5 slide set |
| `docs/guidelines.md` | AC13 guidelines and limitations |
| `dashboard/data.js` | generated payload for the offline dashboard |

Then open `dashboard/index.html` by double-clicking it. No server, no build
step, no internet connection.

## Auditing a database you cannot share

```bash
python -m pipeline.audit --input "<database.xlsx>"
```

This prints a report of **booleans and integers only**: counts, flags, enum
codes and the line numbers of any crash. It contains no measured value, name,
header, ID or file name, so it can be copied off the machine that holds the
data. It is also saved to `outputs/reports/audit.txt` (gitignored; choose
another path with `--out`). Every stage runs in isolation: if one fails, the
report says where and the later stages still run. `python -m pipeline.run ...
--audit` appends the same block to a normal run, or prints it in place of a
failed one.

The dashboard's **Admin / Audit** tab shows the same report, plus a browser
half (which tabs rendered, which charts came up empty, whether the model
evaluates). **Copy report** puts both on the clipboard.

The guarantee is enforced in code. `AuditReport.add` rejects anything that is
not a `bool` or `int`, and `tests/test_audit.py` holds every rendered line to a
grammar that has no room for data. An ingest rejection is reported as
`schema_error_code`; the codes are listed in
`pipeline/io/schema.py:SCHEMA_ERROR_CODES`.

## Disintegration time (optional section)

If the workbook has a `Disintegration` sheet, `pipeline.run` also analyses it.
To run this section on its own:

```bash
python -m pipeline.disintegration --input "<database.xlsx>" [--formats png,pdf,svg,tiff]
```

**Sheet layout.** One row per formulation, in the same form as the dissolution sheet:

- `ID`, `Case`, `API`, `HPMC Grade`
- `API [wt%]`, `HPMC [wt%]`, `Lactose [wt%]`
- `DT_1 [min]` … `DT_4 [min]`, one column per replicate. The unit is read from the header and may be `s`, `min` or `h`.
- An optional `Test_end [min]`.

A blank cell means that replicate was not run. A value written `>1440`, or one
at or past the test end, is right-censored: the tablet had not disintegrated
when the test stopped. Censored values are kept and flagged, and excluded from
the models.

**What it produces:**

- `outputs/reports/disintegration.md`
- `disintegration_diagnostics.{md,json}`
- figures DT-01 … DT-08 in `outputs/figures/disintegration/`, with `captions.md`
- a privacy-safe `[T]` block in the audit

The analyses are:

- correlation of DT with every dissolution response
- Deming regression of DT on the Weibull time scale
- ANCOVA by grade
- grade ratios at identical composition, from the randomised-block model `ln DT ~ grade + case` with Tukey intervals
- erosion lag DT/Td
- the classical DoE fitted to DT
- leave-one-out comparison of recipe-only and dissolution-only prediction

**Synthetic data.** There is no real DT data yet. To create a synthetic workbook:

```bash
python -m pipeline.disintegration.synthetic --input "<database.xlsx>"
```

This writes a copy of the workbook to `outputs/synthetic/CR_with_DT_SYNTHETIC.xlsx`,
with DT generated from the workbook's own dissolution. The true parameters are
stored in the `Disintegration_Notes` sheet, and `tests/test_disintegration.py`
checks that the analysis recovers them.

## Verification

```bash
python -m pytest
python -m ruff check .
python -m mypy pipeline
python -m pipeline.run --input "CR_matrix_tablet_dissolution_PLACEHOLDER (1).xlsx" --check-determinism
```

`--check-determinism` re-runs the whole chain from the raw file and fails unless
`data.js` is reproduced byte for byte.

Two test groups need Node (already used to verify the shipped dashboard):

* `tests/test_js_parity.py` runs `dashboard/model.js` under Node and asserts it
  reproduces Python's predictions. The formulator tool re-implements the fitted
  surface in JavaScript so the page works with no Python runtime; without this
  test the two could drift and every number in the tool would be quietly wrong
  while the page rendered perfectly.
* `tests/test_dashboard_smoke.py` loads the page in a real DOM and clicks every
  tab, because a runtime error in `boot()` leaves a blank page that every static
  check still passes. Needs `npm install jsdom`.

Both **skip** rather than pass when their dependency is missing.

## Sending it to someone else

```bash
python -m pipeline.package
```

Writes `dist/`. There is nothing to deploy — the dashboard is a file, not a
service. The recipient double-clicks it.

**If they only need to see the results:** send `CR_workbench_standalone.html`
(282 KB, everything inlined) plus `CR_HOW_TO_OPEN.txt`. Add
`CR_dashboard_base64.txt` as a fallback for filters that strip `.html`.

**If they need to re-run it on new data:** send `CR.zip` (1.1 MB), or
`CR_project_base64.txt` if zips are blocked, plus `CR_HOW_TO_REBUILD.txt`. They
need Python 3.11+ and an internet connection once for
`pip install -r requirements.txt`. After that everything runs offline.

`EMAIL_TEMPLATE.txt` is the email body, ready to paste.

The `.txt` files are base64 containers, not something anyone reads or pastes
into an editor — one command turns them back into the original file. Both
round-trips are covered by tests, and the archive is byte-reproducible.

## Publishing to GitHub

The repository is code-only. Data is ignored by default — `.gitignore` blocks
every `*.xlsx`/`*.csv` and re-allows only the file named `…PLACEHOLDER….xlsx`,
so a real formulation database cannot be committed by accident. Generated
outputs (`outputs/`, `dashboard/data.js`, `docs/guidelines.md`) are excluded too,
since on a real database they would publish the results as surely as the data.
Four tests in `tests/test_repo_safety.py` enforce this rather than trusting it.

```bash
git commit -m "Controlled-release matrix tablet DoE pipeline and workbench"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

**Whoever you send it to does not need a GitHub account.** On a public repo the
green **Code → Download ZIP** button works logged-out. They unzip it, open the
folder in VS Code, and follow `START_HERE.md`. Nobody needs to copy files by
hand.

A fresh clone ships with the synthetic placeholder database, so
`python -m pipeline.run --input "CR_matrix_tablet_dissolution_PLACEHOLDER (1).xlsx"`
works immediately — useful as a smoke test before pointing it at real data.
`dashboard/index.html` will say it needs regenerating until that command has been
run once, because `data.js` is not committed.

Consider whether `TASK.md` should be public: it is the project brief rather than
code, and it is committed by default.

## Layout

```
pipeline/
  io/          schema detection, loading, data-quality reporting
  profiles/    kinetic fits, censoring, metric extraction, tidy tables
  design/      model matrices, aliasing, efficiency, prediction variance
  responses/   correlation, clustering, PCA, key-response selection
  surfaces/    least squares, diagnostics, Box–Cox, model reduction
  validation/  leave-one-formulation-out cross-validation
  equivalence/ f2 similarity and equivalence sets
  stress/      D-optimal reduced-design evaluation
  figures/     matplotlib rendering (publication.py style, headlines.py slide set)
  export/      data.js serialisation
dashboard/     index.html, model.js, app.js, styles.css, data.js (generated)
tests/
```

## Things worth knowing before reading the numbers

* **The analysis window is 0–24 h** (`ANALYSIS_WINDOW_H` in
  `pipeline/config.py`). Readings from runs extended past it for diagnostics
  are dropped at load. The one reading just after 24 h is kept, so that the
  24 h value can be interpolated.
* **Each replicate keeps its own clock.** When the workbook has a time column
  per replicate (`Min_1`, `Min_2`, …), each is paired with its own `conc_N`.
* **Densely logged probe data are resampled for comparison only.** When the
  timestamps are too many to reconcile into a schedule, each replicate is
  interpolated onto `NOMINAL_SCHEDULE_H`, then averaged. Interpolation happens
  only between readings, never across a gap wider than `MAX_INTERP_GAP_H`, and
  never beyond the measured range. Metrics and kinetic fits still use every
  measured point; the resampled curves feed only the mean profiles, f2,
  cross-validation and the stress test.
* **The components sum to 100 wt% exactly**, so this is a mixture: the
  composition matrix has rank 2 and the three components cannot be varied
  independently. A polynomial with an intercept *and* all three components is
  exactly singular, not merely ill-conditioned. The pipeline fits the Scheffé
  canonical form and reports the equivalent slack-variable reading alongside it.
* **The three replicates are vessels from one compression batch.** They are
  subsamples, not independent runs, so surfaces are fitted on design-point means.
  Fitting all 99 profiles as independent would shrink standard errors by roughly
  √3 and manufacture significance the design cannot support.
* **Two different uncertainties.** Replicate SD measures within-batch vessel
  repeatability and is small. The leave-one-formulation-out CV error measures
  predicting an unseen formulation and is what every prediction is reported with.
  They are never substituted for one another. Batch-to-batch variability is
  unmeasured in this design.
* **Censoring and Weibull-asymptote identifiability coincide exactly.** Every
  formulation that fails to reach 80% release also fails to determine its own
  asymptote, so its `Td` is extrapolated rather than estimated. Those are the
  slow formulations a CR study cares most about, and they carry more uncertainty
  than the headline CV figure suggests.
* **The metrics are not independent evidence.** Fourteen computed metrics
  collapse to three real dimensions, which map onto Weibull scale, asymptote and
  shape. Conclusions are drawn only on the reduced key-response set.
