# The early delay: what it is and how the ternary DoE can tell

These are working notes for the agent that runs the swelling model on the ternary API–HPMC–lactose
DoE. They describe a short delay at the start of release that MS5 found, the models in `delay.py`
that represent it, and a composition probe that can identify its cause. As in
`TERNARY_TRENDS.md`, treat everything here as hypotheses to test.

## 1. What MS5 found

The MS5 data were 24 binary formulations (APAP or IBU, each in K15M, K100M and K100LV at
20–60 % load) and one 30 h APAP/K15M profile. Adding a single early-delay parameter to D1
gave the following.

| | D1 | D1+lag | D1+k |
|---|---|---|---|
| 30 h APAP, test RMSE (train ≤ 8.21 h, test 8.21–30 h) | 2.30 pp | **1.28 pp** | 1.36 pp |
| 30 h APAP, training AICc | −117.9 | −120.4 | **−130.4** |
| 24 formulations, median RMSE, APAP / IBU | 1.51 / 0.35 pp | 0.99 / 0.19 | **0.59 / 0.18** |
| D0 bounded by the data (profile CI) | 15 / 24 | 21 / 24 | 21 / 24 (IBU 4 → 10) |
| Delay, APAP / IBU (medians) | – | 22 / 21 min | k = 3.0 / 3.0 h⁻¹ (τ = 1/k ≈ 20 min) |

- **Prediction gain.** Some early delay was enough. The plain lag predicted the 30 h data as
  well as the first-order step.
- **Fit quality.** The smooth first-order shape fitted the early points better. Its AICc was
  more than 2 below the lag's in 16/24 formulations; the lag's was lower in 3/24.
- **Identifiability.** The delay absorbed the first ~20 min, which otherwise pushed D0 toward
  0, so D0 became identifiable in 6 more IBU formulations.
- **Trade-offs.** k correlated with D0 (median r = −0.89) and with γ (+0.78). Read D0 and γ
  of a delay fit as a set, not separately from a D1 fit.
- **Not drug dissolution.** k was the same for APAP and IBU, whose solubilities differ about
  40–400×. The delay was, however, grade-dependent: K15M gave 2–21 min (median ≈ 11), K100M and K100LV
  gave 12–28 min (median ≈ 22). That leaves two candidates, matrix wetting/hydration and the experimental set-up.
  The binary design cannot separate them, because grade and run session may be confounded.

## 2. Three families of cause, with different signatures

| Cause | What should control the delay | What should not |
|---|---|---|
| Wetting, hydration, first gel layer forming | Matrix: HPMC grade and fraction, soluble or hydrophilic fillers (lactose) | The API |
| Drug dissolution | The API: particle size, solubility, wettability | Polymer composition, to first order |
| Experimental (dead time, how t = 0 is defined, tablet settling or sticking) | Run set-up and session | Formulation |

**Low solubility alone does not make dissolution rate-limiting.** For a particle dissolving
in a stagnant gel, compare the particle's dissolution time with the tablet's diffusion time:

    t_dissolve / t_diffuse ≈ ρ r0² / (2 c_s R²)

- ρ is the drug density, r0 the particle radius, c_s the solubility in the gel and R the
  tablet radius. The gel diffusivity cancels out of the ratio.
- **IBU example:** ρ ≈ 1.1 g/mL, r0 ≈ 25 µm (substitute measured values) and R ≈ 6 mm.
  - At a buffer solubility of a few mg/mL the ratio is ≈ 0.003, so dissolution is effectively
    instantaneous.
  - At the intrinsic solubility (≈ 0.03 mg/mL) the ratio is ≈ 0.3, still faster than diffusion.
- Only coarse particles combined with very low solubility make the step rate-limiting.
  Elsewhere, solubility acts through capacity (how much drug is mobile, i.e. D0 ≈ D·c_s/c0),
  not through rate.

## 3. Why lactose is the probe

Lactose changes the matrix without changing the drug.

- **Wetting.** Lactose is soluble and hydrophilic, so it speeds wetting and water uptake. A
  wetting/hydration delay should shorten as lactose rises. A drug-dissolution delay should not
  change.
- **Leaching.** Lactose dissolves out and leaves pores. The matrix keeps opening after swelling
  has stopped, which makes leaching a candidate for γ. Prediction: γ rises with lactose at a
  fixed HPMC grade. This is the same γ–lactose link noted in `TERNARY_TRENDS.md`.
- **Dilution.** Lactose dilutes the HPMC, giving a weaker gel and probably faster erosion.
  Prediction: higher D0 and γ. This is the same direction as the binary MW_eq trend for APAP.

## 4. How to run it

```bash
python -m swelling_model --input ternary_release.csv --cutoff-h 8 --delay k,lag
```

**Input.** In addition to the usual columns (README §0), the input needs one row per tablet and
time point with `api`, `lactose_wt_pct`, and ideally `api_wt_pct` and `hpmc_wt_pct`. These
columns are copied into `fits.csv`.

**Base model.** `--delay-base S1` uses S1 instead of D1. Prefer S1 for trends unless the swelling
of the ternary tablets has been measured (`TERNARY_TRENDS.md` pitfall 4).

**Outputs.** `fits.csv` gains `D1+k` and `D1+lag` rows. Each row has:
- `delay_min` (the lag, or τ = 1/k);
- the profile CIs `D0_prof_*` and `delay_prof_*`, plus `D0_bounded` and `delay_bounded`;
- the correlations `corr_D0_delay` and `corr_gamma_delay`.

With lactose given, the CLI also writes:
- `delay_probe_spearman.csv`: ρ of the delay, D0 and γ against lactose wt % within each
  model × API × grade.
- `delay_probe_ols.csv`: log10(parameter) ~ lactose wt % + C(api) + C(grade). Read the lactose
  coefficient as lactose replacing HPMC at a fixed API, and the C(api) terms as the API effect.
- `delay_probe_api_effect.csv`: per grade, the delay medians by API and a Kruskal–Wallis p.

**From Python:** `fit_delay(model, "k" | "lag", t_h, frac, geom, sw)` and `delay_probe(fits)`.

**Mixture model.** For the full mixture treatment, also put log10(delay) through the Scheffé
pipeline as a response, exactly as `TERNARY_TRENDS.md` §2 item 1 does for D0 and γ.

## 5. Reading the result (decide these before looking)

| Result | Interpretation |
|---|---|
| Delay falls with lactose (negative OLS coefficient, p < 0.05) and the C(api) terms are not significant | Wetting / hydration |
| Delay differs by API (C(api) significant, Kruskal p < 0.05) and does not track lactose; ideally it also follows particle size | Drug dissolution |
| Delay is the same everywhere, or follows run date rather than composition | Experimental |
| γ rises with lactose at a fixed grade | Pore formation by leaching is part of the "continuing process" behind γ |

## 6. Pitfalls

1. **Early sampling decides everything.** The delay is set by the first 1–2 h. With fewer than
   3 points in the first hour, expect `delay_bounded = False`, and do not trend it.
2. **Check the experimental family first; it is the cheapest to rule out.**
   - Compare the fitted delay with the set-up's dead time: tubing volume / flow rate for a
     flow-through UV system, and the time-zero convention for manual sampling.
   - Check whether formulations that share a delay also share a run session. In MS5 the
     grade-dependence (K15M shorter) may reflect session, not grade.
3. **Collinearity.** At a fixed API load, lactose and HPMC move in opposite directions (the
   mixture constraint), so a lactose effect is also an HPMC effect. Formulations with equal
   HPMC but different API/lactose ratios separate them; check VIF before reading coefficients.
4. **Lag vs k.** If the lag model wins on AICc for most formulations, the delay is sharp (a dead
   time or an induction period). A smooth first-order onset fits wetting or dissolution better.
5. **Trade-offs.** The delay trades off against D0 (r ≈ −0.9). When trending D0 or γ, take them
   from the same model family throughout: all from D1 or all from D1+k, never mixed.
6. **Use design-point means**, as in `TERNARY_TRENDS.md` pitfall 2. Vessels from one compression
   batch are not independent.
