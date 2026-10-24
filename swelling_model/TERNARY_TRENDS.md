# Trending D0 and γ in ternary API–HPMC–lactose matrices

These are working notes for an agent that extends the parameter trends of `README.md` §6 from
binary drug/HPMC tablets (MS5) to the ternary DoE (API, HPMC, lactose). They record what was
seen, how to interpret it physically, what to try next, and the pitfalls. Treat everything
here as hypotheses to test, not results. The outcome goes into the MS5 seminar deck as an
extension of the binary results, so §4 says what the deck needs.

**Where the data stand.** The design has 11 compositions per API, each run in 3 HPMC grades
(K100LV, K15M, K100M). Each API load comes with a low and a high HPMC level, and therefore a
high and a low lactose level. First observations:

- **D0 varies with HPMC MW**, unlike in MS5, where it was a drug property.
- **γ tracks lactose content strongly.**
- **Trending either parameter against MW or MW_eq alone gives no clear trend**, because each
  MW appears in several compositions.
- **A brute-force search did not settle it.** Products of composition variables with
  exponents of −2 to 2 found only a weak trend, roughly γ or D0 ∝ √MW · w_API / w_lactose.
  See pitfall 1 before building on this.

---

## 1. Physical picture: binary vs ternary

Read the two parameters as two different stages of the gel:

- **D0 is the initial gel barrier:** how easily drug crosses the first hydrated layer.
- **γ is the rate at which the matrix opens:** how fast the gel becomes more permeable over
  time, through hydration, porosity growth and erosion.

**Binary (MS5).**
- **Coupled composition:** drug load and polymer content are tied (HPMC = 100 − load), so
  one variable, MW_eq = (1 − x_drug)·MW, captures both. γ falls with MW_eq (S1, APAP ρ = −0.90).
- **D0 identifies the drug:** APAP/IBU ≈ 29 at the same grade and load. Part of this is
  solubility, not diffusivity. At 20–60 % loading the drug exceeds its solubility in the
  gel, so the fitted D0 behaves like D·c_s/c_0 (Higuchi-type). It is an effective value that
  is fixed by the drug and its load.
- **The drug was already a pore former:** within K15M, γ for APAP rose with drug load
  (0.24 → 0.96 × 10⁻⁹ mm² s⁻², 20 → 60 %).

**Ternary (this DoE).**
- **Lactose decouples polymer content from drug load.** The same drug load now comes with
  more or less HPMC, so MW_eq no longer describes the formulation on its own.
- **Lactose is a highly soluble pore former.** It dissolves and leaches out of the gel,
  leaving water-filled pores and channels that grow with time. That time-dependent rise in
  permeability is exactly what D(t) = D0 + γt absorbs, so γ tracking lactose is physically
  expected. Literature support:
  - Li et al. (2005, J Pharm Pharmacol): soluble fillers increase release, and soluble
    components act as pore formers (Yang & Fassihi).
  - Siepmann & Peppas (2012, Adv Drug Deliv Rev): matrices become more porous as soluble
    material depletes, which requires a time-dependent diffusivity.
- **D0 can now show the polymer.** At a fixed drug load, the drug-solubility part of D0 is
  held constant, so the polymer's contribution to the initial barrier becomes visible.
- **Lactose dilution matters most for low-MW grades** (next section).

**The disentanglement concentration (Ju et al., 1995, J Pharm Sci).** Below ρ_dis, HPMC
chains disentangle from the gel and leave:

    ρ_dis = 0.05 · (M / 96 kDa)^−0.8   g/mL
    K100LV (29 kDa): 0.13    K15M (134 kDa): 0.038    K100M (267 kDa): 0.022 g/mL

Diluting the polymer with lactose pushes low-MW grades (high ρ_dis) toward the threshold, so
their first gel layer is weak. High-MW grades stay well entangled. This predicts that D0
depends on MW most strongly when the polymer is diluted, which is the ternary case.

**Entanglement ratio, defined in the MS5 seminar (not from the literature).**

    ψ = φ_p / ρ_dis       φ_p = HPMC mass fraction of the tablet

Because ρ_dis ∝ M^−0.8, ψ ∝ w_HPMC · M^0.8. The √MW that the exponent search turned up
belongs to this family, so ψ may be the physical form of what the search was reaching for.
In the MS5 data:

- **ψ vs measured gel tan δ at 8 h:** ρ = −0.79 (p = 0.04). This is 7 IBU tablets, with tan δ
  read off the T3.9 closeout report, Fig. 10. More entangled gels are more elastic.
- **γ (S1) vs ψ:** APAP ρ = −0.89, IBU ρ = −0.52.
- **Do not use γ (D1) vs tan δ.** It correlates (r = −0.79) but runs opposite to the γ–ψ
  trend, likely an artifact of the 200 mg swelling data transferred to K100M tablets. γ (S1)
  vs tan δ shows no clear trend.

---

## 2. Things to try, in order

1. **Run D0 and γ through the existing Scheffé pipeline first.** `pipeline/doe/analysis.run`
   already fits the dissolution responses on this design: Scheffé canonical form in
   API/HPMC/lactose, grade as a process term in coded log-viscosity (`v_coded`), degree chosen
   by sequential sums of squares, backward reduction, ANOVA, and the leave-one-formulation-out
   CV error. Its `responses` argument takes any study measured on the same design points, and
   `pipeline/disintegration/analysis.py` shows the pattern:
   - define `ResponseSpec("log10_D0", ...)` and `ResponseSpec("log10_gamma", ...)` (fit the
     logs; both parameters span decades and are positive);
   - build `ResponseData` aligned row for row with `design_points` (merge on case and grade,
     as `_response_data` does), marking D0 at the lower bound and failed fits as unavailable;
   - call `run(design_points, replicates, max_process_power, responses=[...])`.

   This gives D0 and γ the same treatment, tables and CV figures as t50 and MDT, so the two
   can be compared directly. Note that grade enters as viscosity, not MW; the two rank the
   grades identically, but quote MW (and ρ_dis) when the physical reading needs it.

2. **Test 2–3 physical descriptors, chosen before fitting:**
   - **Entanglement:** ψ = w_HPMC / ρ_dis(MW). Expected to drive D0, and possibly γ.
   - **Pore formers per polymer:** s = (w_API + w_lactose) / w_HPMC. Also try
     w_lactose / w_HPMC on its own. Expected to drive γ.
   - Fit log θ = a + b·log ψ + c·log s.
   - Compare against the Scheffé fit from step 1 with AICc and the same cross-validation. If two
     physical descriptors match the mixture model's predictive error, you have both the
     composition link and its interpretation.

3. **Try better-identified combinations of D0 and γ.** D0 and γ are strongly correlated
   within each fit (r ≈ −0.7 to −0.9). Composition may map cleanly onto a combination even
   when D0 and γ separately look noisy:
   - D(t50) = D0 + γ·t50: the diffusivity at mid-release.
   - t* = D0/γ: the crossover time.
   - The mean D over the fitted window.

4. **Weight each regression by the parameter's uncertainty.** Use the profile-likelihood
   intervals (`D0_prof_lo/hi`, `gamma_prof_lo/hi`). Exclude D0 values at the lower bound, as
   `parameter_trends` already does.

5. **Close the loop; this is the strongest evidence.** For each held-out composition:
   - predict D0 and γ from composition alone (Scheffé or descriptor model);
   - generate the full release profile with `fraction_released`;
   - compare it to the measured profile (RMSE, f₂).

   If this works, the composition link holds well enough to predict release of unseen
   formulations.

6. **Try one cross-dataset check.** Plot γ against the total soluble fraction
   (w_API + w_lactose, or s) for both the MS5 binary and the ternary data. If they collapse
   onto one trend, γ is "how fast soluble components open the gel" in both systems.

---

## 3. Pitfalls

1. **Exponent grid searches overfit.** Five exponents on three variables is about 125
   candidate expressions, fitted to 11 compositions per grade and API. The best of 125 will
   show *some* trend by chance. If you search, report how many expressions were tried,
   select on cross-validated error (not R² or ρ), and confirm on held-out compositions.
   Prefer prespecified physical descriptors.
2. **The mixture constraint ties variables together** (`README.md`, "Things worth knowing before reading the numbers"). At fixed API load, lactose and HPMC
   move in opposite directions, so "γ tracks lactose" may be the same finding as "γ falls
   with polymer fraction", which is the binary trend again. To separate them, compare
   formulations at equal HPMC but different API/lactose ratios, or read the lactose
   coefficient of the Scheffé model. Check collinearity (VIF) before reading individual
   coefficients.
   **Fit design-point means, not vessels.** The three replicates come from one compression
   batch, so fit the swelling model to each design point's mean profile (or average the
   per-vessel parameters) before trending. Treating vessels as independent shrinks the
   standard errors by about √3.
3. **D0 and γ trade off within a fit.** A trend can migrate from one parameter to the other
   when the profile is short or nearly linear. Check the intervals, and see item 3 above.
4. **Keep each model separate.** Trend S1 with S1 and D1 with D1 (`README.md` §6). D1's γ
   inherits errors in the swelling input, and lactose changes swelling: lactose-rich tablets
   swell less and may erode sooner. The preloaded placebo HPMC swelling is not
   representative of filled tablets. Prefer S1 for trends unless you have measured swelling
   for the ternary tablets.
5. **The apparatus can differ between datasets.** If the ternary data are USP II and MS5 was
   USP I, the hydrodynamics differ. D0 is the parameter most sensitive to the surface
   boundary layer. Do not pool D0 across apparatus without checking.
6. **D0 carries drug solubility at high loading.** When the load exceeds the drug's
   solubility in the gel, D0 behaves like D·c_s/c_0. Compare D0 across grades only at a fixed
   API and load.
7. **K100LV erodes.** In MS5, K100LV behaved like a threshold (erosion) case for IBU rather
   than part of a continuous MW trend. A ternary trend may break at K100LV for the same
   reason, so test it with and without K100LV.
8. **ψ is an index, not a measured concentration.** It uses the tablet's dry polymer mass
   fraction, while the gel's polymer concentration falls as the tablet swells. Use ψ to rank
   formulations, not as a physical concentration.
9. **The sample is small.** Always report ρ with n and p. A few points per API and grade
   make a trend suggestive at most.
10. **Molecular weights are nominal.** K100LV 29, K15M 134, K100M 267 kDa
    (`trends.HPMC_MW_KDA`). Batch-to-batch variability of HPMC is known (Dahl et al., 1990),
    so confirm the values against supplier data where possible.

---

## 4. What goes into the deck

The results will be added to the MS5 seminar deck (built elsewhere by script), after the
binary parameter-trend slides. Present the ternary results as an extension of the MS5
parameter map, not a contradiction. Deliver figures and a short brief, in the spirit of
`pipeline/deck_brief.py`, rather than slides:

1. **Recap (already in the deck):** in binary tablets, the drug sets D0 and MW_eq sets γ.
2. **Ternary trends:** D0 against grade at fixed drug load, and γ against lactose (or s),
   coloured by grade.
3. **A 2 × 2 "what moves each parameter" grid:**

   | | Binary | Ternary |
   |---|---|---|
   | **D0** (initial barrier) | drug identity | grade MW (entanglement, ψ) |
   | **γ** (opening rate) | polymer content and MW | lactose, the soluble pore former |

4. **Composition maps:** ternary contour plots of predicted log γ and log D0 per grade, with
   the measured formulations overlaid (`pipeline/doe/ternary.py` already draws the region
   and grid).
5. **Parity plot:** Scheffé or descriptor model against the fitted parameters, with the
   cross-validated error.
6. **Closed loop:** predicted vs measured release profiles for held-out compositions.

Figure and text conventions for the deck:
- PNG at ≥ 220 dpi, sized for a 13.33 × 7.5 in slide (one panel ≈ 5.5–6 in wide, a pair
  ≈ 11.6 in); transparent or white background.
- Ticks inward on all four sides; Calibri for text, Cambria italic for math (D₀, γ, ψ,
  ρ_dis, M_t/M_∞); the MS5 grade colours and APAP/IBU markers.
- No error bars on log-scale parameter plots unless they stay readable; give the interval
  method in the caption instead.
- For each figure, a one-sentence takeaway and the numbers behind it (ρ or slope with n and
  p, CV error), written in past tense and hedged where the evidence is only directional.

If the descriptors do not beat the Scheffé fit, present the Scheffé fit as the composition
link and the descriptor result as a hypothesis (e.g. "γ tracks soluble content per polymer").

---

## 5. Code hooks

- **Composition columns:**
  - `trends.composition_from_columns` and `mw_eq` read `wt_pct_<GRADE>`, `hpmc_wt_pct` or
    `drug_load_pct` (`README.md` §0).
  - Add `api_wt_pct` and `lactose_wt_pct` columns for the ternary descriptors. Lactose
    counts as MW 0 in `mw_eq`.
- **`trends.parameter_trends`:** gives Spearman ρ and the log–log slope against `mw_eq`. Add
  ψ and s as further x-variables, or write a `ternary_trends` function alongside it. Keep
  the at-bound exclusion and the profile-CI columns.
- **Scheffé pipeline:** `pipeline/doe/analysis.run` (with `responses=`),
  `pipeline/doe/responses.ResponseSpec`/`ResponseData`, and the disintegration study in
  `pipeline/disintegration/analysis.py` as the worked example of a non-dissolution response.
- **`fit_model`, `profile_ci`, `fraction_released`:** use these for the fits, the intervals
  and the closed-loop predictions (pitfall 3 and item 5 of §2).
