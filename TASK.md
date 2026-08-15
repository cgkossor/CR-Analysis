# TASK — Controlled-Release Matrix Tablet DoE Analysis & Formulator Dashboard

---

## CONFIRM BEFORE START

These are assumptions made in the absence of an answer. Correct any that are
wrong before work begins; several change the modeling approach.

| # | Assumption | Changes what if wrong |
|---|---|---|
| A1 | API + HPMC + lactose are constrained to a fixed total (mixture design), with a fixed balance of lubricant/glidant | Whether mixture-model terms or standard polynomial RSM is correct |
| A2 | One dissolution profile per formulation (no replicates, no SD) | Whether uncertainty bands come from replicates or from model residuals only |
| A3 | Data is long format: one row per (api, formulation_id, hpmc_grade, time_h, pct_released) | Ingestion layer only |
| A4 | Tablet weight is fixed across the design; hardness/porosity controlled but not recorded as a factor | Whether a covariate must enter the model |
| A5 | Three HPMC grades with known nominal viscosities (e.g. K100LV / K4M / K100M) | Whether log-viscosity can be treated as continuous — see G3 |
| A6 | Dissolution method is constant across all formulations and APIs | Whether cross-API comparison is even valid |
| A7 | Timepoints are common across all profiles | Whether interpolation is needed before metric extraction |

**Also required before start:** the truncated placeholder database file, and the
nominal viscosity value for each HPMC grade.

---

## Goal

Build a single, self-contained, offline browser page that lets a formulator (a)
explore an existing controlled-release matrix tablet DoE database and its
analysis, and (b) use that analysis as a starting point when formulating a new
API — receiving a candidate composition set and HPMC grade for a target release
profile and dose, plus a recommended minimum experiment set.

Success means a skeptical formulator opens the page, checks a prediction against
the underlying experiments, finds the backing data and the error estimate, and
decides the tool is worth using. Success is **not** a model that fits well.

---

## Scope

**In scope**

- Python pipeline: ingestion → per-profile model fitting → metric extraction →
  DoE response surfaces → validation → static asset export
- Statistical analysis suite (see Acceptance Criteria 2)
- Two-stage modeling: per-formulation profile fit, then response surface on the
  fitted parameters
- Iso-release equivalence sets via f2 similarity
- Design stress test: minimum experiment set for a new API
- Publication-grade figures, ranked by relevance to the storyline
- Static offline dashboard (HTML + vendored JS + generated `data.js`)
- Interactive formulator tool: forward prediction and inverse design
- Written guidelines document derived from the analysis

**Out of scope**

- Predicting or modeling API physicochemical properties from structure
- Any claim linking solubility class to release behavior until ≥2 APIs per class
  exist in the database (see G1)
- Machine learning beyond what the design supports — no neural nets, no
  gradient boosting on 33 points
- Stability, manufacturability, compaction, or content-uniformity modeling
- IVIVC, PK prediction, or bioequivalence claims
- User accounts, databases, servers, hosting, or deployment
- Any framework requiring a running process: Streamlit, Dash, Panel, Flask,
  Jupyter, Node
- Refactoring, restructuring, or "improving" code outside the current work unit
- Anything not listed under "In scope"

---

## Guardrails

Violating any of these means the work is rejected regardless of quality
elsewhere.

- **G1 — No solubility claims without the data.** With one API in the database,
  no output may state, imply, or visualize a relationship between API solubility
  and release behavior. Every cross-API and solubility-class feature must be
  built and then gated behind a data-sufficiency check that renders an explicit
  "insufficient data — requires ≥2 APIs per solubility class" state. When the
  gate does open at 2×2, outputs must label the contrast as *confounded with
  molecule identity* and directional only.
- **G2 — Never fabricate, simulate, impute, or extend data.** If the placeholder
  database is missing APIs, grades, timepoints, or formulations, the correct
  response is to surface the gap, not to fill it. No synthetic rows, no
  "example" profiles, no filling censored values with 100%.
- **G3 — No silent extrapolation.** Any prediction outside the convex hull of
  tested compositions, outside the tested viscosity range, or for an untested
  API must render a prominent warning *and* display the nearest tested
  formulations with their measured profiles. Interpolation between the three
  tested viscosity grades is permitted; extrapolation beyond them is not.
- **G4 — Peppas power law is fit and reported on the ≤60% released portion
  only.** Any profile with fewer than 3 points in that window gets no Peppas
  parameters. The exponent *n* must never be reported for a fit that violates
  this.
- **G5 — Censored profiles are censored, not missing.** A formulation that never
  reaches 80% release has `t80 = >24 h`, propagated as a censored value through
  every downstream summary. Never drop it, never coerce to NaN and silently
  exclude, never report a mean that ignores it.
- **G6 — Every prediction carries a backtest number.** No predicted profile,
  composition, or recommendation may be displayed without an accompanying
  cross-validated error estimate and a link to the underlying experiments it was
  derived from.
- **G7 — The dashboard runs offline by double-click.** `index.html` opened
  directly from the filesystem must work fully: no server, no internet, no CDN,
  no `fetch()`. Data ships as `data.js` assigning to a global. JS libraries are
  vendored into the repo.
- **G8 — Do not modify the source data file.** All derived outputs go to
  separate directories. The input is read-only.
- **G9 — Do not weaken, skip, delete, or rewrite tests to make a build pass.**
  Do not swallow errors or widen types to silence a failure.
- **G11 — Correlated responses are not independent evidence.** The release
  metrics in AC2 are redundant by construction — `t50`, MDT, `% released at
  12 h`, and Weibull scale largely measure the same latent quantity. No output
  may present agreement among correlated metrics as independent confirmation,
  count them separately toward a conclusion, or run effect tests across the full
  metric set without accounting for multiplicity. All conclusions, figures, and
  formulator outputs are drawn on the key responses selected in AC5.
- **G10 — Deterministic.** All resampling, CV splits, and optimizer starts use
  fixed seeds. Two runs of the pipeline produce byte-identical `data.js`.

---

## Critical attributes

- **Honest uncertainty.** Every surface, prediction, and equivalence set carries
  a cross-validated error estimate. Where the design cannot support an estimate,
  the output says so rather than defaulting to a number.
- **Traceability.** From any prediction, a user can reach the specific measured
  formulations that support it in one interaction.
- **Coherent profiles.** Predicted release profiles are monotonic non-decreasing
  and bounded [0, 100]. A prediction that violates this is a bug, not a warning.
- **Novelty above the obvious.** The headline findings must go beyond "more
  HPMC and higher grade → slower release." That relationship is assumed known;
  confirming it is a validation check, not a result. Findings are ranked by what
  a working formulator does not already know.
- **Graceful degradation.** The full pipeline and dashboard run end-to-end on a
  single-API database without errors, without empty panels, and without
  misleading placeholders.
- **Dimensional honesty.** The analysis reports how many independent things the
  response set actually measures, and draws conclusions on that many — not on
  the number of metrics that happen to be computable.
- **Reproducible.** One command regenerates every figure and every dashboard
  asset from the raw file with no manual steps.

---

## Acceptance criteria

1. **Ingestion & validation.** Pipeline loads the placeholder database, validates
   schema, and emits a data-quality report listing: APIs present, formulations
   per API, grades per formulation, timepoint coverage, monotonicity violations,
   profiles never reaching 80%, and any design points missing versus the intended
   11×3. Malformed input fails loudly with a specific message.

2. **Metric suite.** For every formulation, computes and exports:
   - `t10`, `t25`, `t50`, `t80` (censored-aware per G5)
   - `% released` at 1, 2, 4, 8, 12, 24 h
   - Mean dissolution time
   - Early-phase slope (0–2 h) and late-phase slope (8–24 h), and their ratio
   - Weibull fit: scale, shape β, R², RMSE
   - Peppas fit: `k`, `n`, R², plus the number of points used (per G4)
   - Higuchi and first-order fits with R², for model-comparison context

   Each metric carries a validity flag. Extraction is unit-tested against at
   least three hand-computed reference profiles.

3. **Design diagnostics — run before any response modeling.** Characterizes the
   design itself, independent of the data:
   - Correlation matrix of the coded factors; VIF for every model term;
     condition number of X'X
   - Aliasing/confounding structure of the truncated 11-point design: which
     terms are estimable, which are aliased, and at what order
   - Leverage per design point; D-, A-, and G-efficiency
   - Fraction-of-design-space (FDS) plot and scaled prediction variance across
     the region — this is what tells a formulator where predictions are worth
     trusting, and it feeds G3
   - Whether lack-of-fit is estimable at all. It requires replicates or centre
     points; if the design has none, the report states that plainly rather than
     printing a meaningless F test
   - **If A1 holds**, the three components are perfectly collinear by
     construction and standard polynomial RSM on all three is invalid. A Scheffé
     mixture model or an explicit component-reduction must be used. The report
     states which path was taken and why.

4. **Two-stage response surfaces.** Per API, a response surface is fitted on the
   Weibull parameters (not on each metric independently), with `log10(nominal
   viscosity)` as a continuous fourth factor rather than three categories.
   Composition terms follow AC3. Exports:
   - Sequential model sum of squares (linear → 2FI → quadratic), with hierarchy
     preserved through any model reduction
   - Coefficient estimates with standard errors, 95% CIs, t and p
   - R², adjusted R², and **predicted R² via PRESS**; the adj-vs-pred gap is
     flagged when it exceeds 0.2
   - **Adequate precision** (signal-to-noise), flagged when below 4
   - Lack-of-fit F test where AC3 established it is estimable
   - **Box–Cox analysis with a transformation recommendation.** Release times and
     rate constants are typically log-normal; fitting untransformed is a common
     and avoidable error
   - Residual diagnostics: normal probability plot, externally studentized
     residuals vs predicted and vs run order, Cook's distance, DFFITS, leverage
   - Half-normal / Pareto plot of effects
   - Confidence and prediction intervals across the surface

   A predicted profile reconstructed from predicted parameters must satisfy the
   monotonicity and bounds requirement.

5. **Response space characterization and key response selection.** The metrics in
   AC2 will be heavily correlated; this criterion establishes the true
   dimensionality of the response space and reduces it before anything downstream
   consumes it.
   - Pearson and Spearman correlation matrices across all AC2 metrics, rendered
     as a clustered heatmap
   - Hierarchical clustering into redundancy groups
   - PCA of the standardized response matrix: scree plot, loadings, scores, and
     the number of components reaching ≥90% cumulative variance
   - Explicit separation of **structural** redundancy (`t50` and MDT are
     algebraically related) from **empirical** correlation (two metrics that
     happen to co-vary in this dataset)
   - A minimal set of **key responses** — one representative per redundancy
     group — selected with written justification on interpretability, model
     quality, and robustness to censoring. Not chosen arbitrarily or by
     convenience.
   - **Consistency check on the two-stage model:** if the response space is
     approximately two-dimensional and those dimensions map onto Weibull scale
     and shape, state it — that is the empirical justification for AC4. If it
     does not, AC4's approach must be revisited and the discrepancy reported
     before proceeding.

   Everything downstream — surfaces, figures, guidelines, formulator tool —
   operates on the key responses, per G11.

6. **Validation.** Leave-one-formulation-out cross-validation across all 33
   points per API, reporting predicted-vs-observed error in both parameter space
   and profile space (RMSE in % released, and f2 between predicted and observed).
   Results are exported per formulation so any single prediction's local
   reliability is inspectable. This CV number is what G6 attaches to predictions.

7. **Equivalence sets.** For any target profile, returns the set of
   composition × grade combinations predicted to be similar by f2 ≥ 50, with
   each member's CV-based confidence. Demonstrates with measured data that
   distinct compositions across different grades produce f2-similar profiles —
   this is a headline result, not a footnote.

8. **Design stress test.** Evaluates reduced designs (subsets of the 11×3) by
   resampling: for each candidate subset size and structure, refit the surface,
   and measure degradation in (a) profile prediction error, (b) equivalence-set
   agreement, and (c) directional conclusions versus the full design. Candidate
   subsets are proposed using the design-efficiency criteria from AC3, not chosen
   at random. Output is a specific recommended point set for a new API — the
   actual compositions and grades to run — with the quantified information loss
   at that size, and the size below which conclusions break down.

9. **Figures.** Publication-grade figures generated by Python, exported as both
   vector and raster, each with a caption and an explicit relevance rank tied to
   the storyline. Includes at minimum: raw profile overlays by grade, response
   correlation heatmap and PCA biplot (AC5), response surface contours,
   half-normal effects plot, FDS plot, observed-vs-predicted CV plot, an
   iso-release equivalence demonstration, and the stress-test degradation curve.

10. **Dashboard.** Static `index.html` opens by double-click and works fully
    offline (G7), with tabs for: Database Explorer, Analysis & Metrics, Design
    Diagnostics, Response Surfaces, Equivalence Sets, Design Stress Test,
    Formulator Tool, and Guidelines & Limitations. Every panel handles the
    single-API case correctly.

11. **Formulator tool — minimum design.** Given a new API and a target release
    window, returns the recommended experiment set from AC8, with the rationale
    and the expected information loss versus running the full 33.

12. **Formulator tool — inverse design.** Given a target profile and API dose,
    returns ranked candidate compositions and HPMC grades, each with predicted
    profile, CV error, in/out-of-design-space status per G3, and the nearest
    measured formulations. Ranking uses a **Derringer–Suich desirability
    function** across the key responses, with the weights exposed to the user
    rather than hidden. Includes an **overlay / sweet-spot plot** showing the
    composition region satisfying all constraints simultaneously. Where multiple
    candidates are equivalent, presents them as a set with the trade-offs, not a
    single answer.

13. **Guidelines document.** A written summary of the levers, their interactions,
    what the data supports, and — explicitly — what it does not. Includes a
    limitations section covering the single-API status, the confounding in G1,
    the response redundancy found in AC5, and the design-space boundaries.

14. **Reproducibility.** A clean checkout plus the raw file regenerates every
    artifact via one command, deterministically (G10).

---

## Context

- Controlled-release matrix tablet formulation is conventionally trial-and-error.
  The coarse levers are known — more HPMC and higher viscosity grade slow
  release — but the interactions among API content, HPMC content, soluble filler
  (lactose), and grade are not well characterized.
- Design: 11 formulations per API varying API / HPMC / lactose content, repeated
  across 3 HPMC viscosity grades = 33 per API. A truncated/hybrid DoE, not a full
  factorial.
- Target database: 4 APIs — 2 high solubility, 2 low solubility. **Currently
  available: one high-solubility API.** Build for 4, run correctly on 1.
- Profiles run 0–24 h; many never reach 100% release.
- The database in hand is truncated and serves as a development placeholder. It
  will be replaced. Nothing may hardcode its specific contents.
- The intended audience is practicing formulators who will be skeptical by
  default and who can check any claim against their own experience.

**Expected results that are NOT findings.** These are assumed known; use them as
sanity checks on the pipeline, and do not present them as conclusions:

- More HPMC (or less API / less lactose) → slower release
- Higher HPMC viscosity grade → slower release
- Overlapping profiles within an API and grade where design points are close

**Expected results that ARE findings.** Rank the storyline around these:

- Where the composition × grade interaction is non-additive — where the second
  lever stops paying
- The size and shape of the equivalence sets: how much formulation freedom
  actually exists for a given target, and where it collapses
- Which regions of the design space are information-rich, and the minimum
  experiment set that preserves the conclusions

---

## Verification

```
# fill in once the project skeleton exists
# python -m pytest
# python -m ruff check .
# python -m mypy pipeline
# python -m pipeline.run --input data/raw/<file> --check-determinism
```

---

## Definition of done

- Every acceptance criterion met, with evidence
- All verification commands pass
- No guardrail violated
- `index.html` opens by double-click on a machine with no internet connection and
  every tab renders correctly against the single-API database
- A review subagent has read this file and the diff and has no blocking findings

---

<!-- ============================================================
     REVIEW PROMPT — paste when work is reported done
     ============================================================

Spawn a subagent to review these changes. Do not summarize anything for it:
tell it to read TASK.md and `git diff HEAD` itself.

It should report:
  - each acceptance criterion: met / not met / partially met, with evidence
  - any guardrail violation (this alone means not done)
  - anything touched outside the declared scope
  - each critical attribute, checked by name
  - findings marked [BLOCKING], [MAJOR], or [MINOR], each naming file and line,
    the problem, and what would resolve it

Look specifically for, in this project:
  - synthetic, simulated, imputed, or hardcoded data anywhere (G2)
  - censored profiles silently dropped or coerced (G5)
  - Peppas n reported beyond 60% release (G4)
  - any solubility or cross-API claim surviving with one API loaded (G1)
  - fetch(), CDN links, or a localhost server requirement in the dashboard (G7)
  - predictions rendered without a CV error or a link to source experiments (G6)
  - correlated metrics presented as independent confirmation, or conclusions
    drawn on metrics outside the AC5 key-response set (G11)
  - standard polynomial RSM applied to three components that sum to a constant
  - lack-of-fit reported for a design that cannot estimate it
  - tests weakened, skipped, or made tautological; errors swallowed; stubs left
    on a real code path

It must not fix anything. Judging and fixing in one pass is how findings get
quietly rationalized away.
     ============================================================ -->
