# Swelling-cylinder release model (D1, D2, S1, S2)

This folder holds the 2D analytical advection-diffusion model of the MS5 paper (Kossor et al.,
*Pharm Res*, in revision). The model describes drug release from a cylindrical HPMC matrix
tablet that swells. It can fit the four model cases to new release data, predict the full release profile
from a short early window, and trend the fitted parameters across formulations. In that
trend, D0 is summarised per API and γ is trended against the weighted-average HPMC
molecular weight of the tablet (MW_eq, §6).

It is self-contained. It does not use `pipeline/` and needs only numpy, scipy, pandas and
matplotlib, which are already in `requirements.txt`. No raw data is stored here. The only
numbers shipped are the fitted swelling parameters of the drug-free HPMC tablets (see §3).

The code reproduces the MS5 batch fit (`nondim_batch_fit.py`) exactly. All 24 formulations ×
4 models match to a relative RMSE of 6e-5, and the 30 h extrapolation also matches.

---

## 0. Quick start

```bash
python -m swelling_model --demo                       # synthetic check of the install
python -m swelling_model --input release.csv          # fit all data
python -m swelling_model --input release.csv --cutoff-h 8 --predict-to-h 24
python -m pytest tests/test_swelling_model.py
```

The input is a tidy CSV with one row per tablet per time point:

| column | meaning |
|---|---|
| `formulation` | any ID; profiles are grouped by it |
| `grade` | `K15M`, `K100M`, `K100LV` (preloaded swelling); other grades run S1/S2 only unless `--swelling-data` is given |
| `tablet` | replicate ID (used for the tablet bootstrap) |
| `time_h` (or `time_min`) | sampling time |
| `released_pct` | cumulative % of **nominal dose**, corrected for sample withdrawal |
| `diameter_mm`, `height_mm` | **initial dry** release-tablet dimensions, **full** height |
| `api` *(optional)* | API name; D0 is summarised per API (§6) |
| `wt_pct_<GRADE>` *(optional)* | wt % of each HPMC grade in the tablet, for blends (e.g. `wt_pct_K15M`, `wt_pct_K100LV`) |
| `hpmc_wt_pct` *(optional)* | wt % HPMC when there is a single grade |
| `drug_load_pct` *(optional)* | binary drug/HPMC tablet (HPMC = 100 − load), as in MS5 |

Give one of the three composition options so that MW_eq can be computed (§6).

Outputs go to `outputs/swelling_model/`, which is gitignored:

- `fits.csv`: parameters, Jacobian and profile-likelihood CIs, RMSE, AICc, t*, α, β, t50, t80.
- `predictions.csv`: predicted curves with a 95 % band.
- `test_scores.csv`: held-out scores when `--cutoff-h` leaves later points.
- One PNG per formulation.
- `trends.csv` and `trends.png`: D0 per API and γ against MW_eq (§6), written when the
  composition columns are present.

The Python API:

```python
from swelling_model import Geometry, swelling_for_tablet, fit_model, extrapolate, profile_ci
geom = Geometry.from_tablet(diameter_mm=12.7, thickness_mm=2.8)      # Z0 = thickness/2
sw   = swelling_for_tablet("K15M", geom.R0, geom.Z0)                 # preloaded defaults
res  = fit_model("D1", t_h, frac, geom, sw)                          # frac in 0-1
ex   = extrapolate("D1", t_h, frac, geom, sw, cutoff_h=8, t_pred_h=np.linspace(0, 24, 97),
                   per_tablet=df[["tablet", "time_h", "frac"]])
```

---

## 1. The model

The model treats a finite cylinder with radius R(t) and **half-height** Z(t). The midplane is
symmetric. The outer surfaces are perfect sinks. The drug starts uniformly distributed. The
effective diffusivity is spatially uniform and changes with time as **D(t) = D0 + γt**. Swelling
is a uniform (affine) dilation, independent in r and z:

    X(t) = Xf + (X0 − Xf) / (1 + k t)        (X = R or Z; first-order saturation)

The solution is an **analytical infinite series**. It is not closed form, but its time
dependence enters only through two **closed-form** diffusion-time integrals:

    I_r(t) = ∫ D(s)/R(s)² ds,    I_z(t) = ∫ D(s)/Z(s)² ds
    M(t)   = 1 − [Σ_n 4/λ_n² e^(−λ_n² I_r)] · [Σ_m 2/μ_m² e^(−μ_m² I_z)]

Here λ_n are the zeros of J0 and μ_m = (2m+1)π/2. The four cases differ only in the two
integrals:

| Case | Boundaries | D(t) | Fitted | Needs swelling data | Role |
|---|---|---|---|---|---|
| **D1** | dynamic R(t), Z(t) | D0 + γt | D0, γ | yes (or defaults) | most physical; best short-window extrapolation |
| **D2** | dynamic | D0 | D0 | yes | constant-D reference with swelling |
| **S1** | static R0, Z0 | D0 + γt | D0, γ | **no** | fits as well as D1 in-sample; no swelling needed |
| **S2** | static | D0 | D0 | no | classic Fickian cylinder (Fu et al. 1976 solution) |

Units: lengths in mm, D0 in mm²/s, γ in mm²/s², and swelling rates in 1/h. The API takes
time in hours. Results are reported as D0 × 10⁶ and γ × 10⁹, matching the paper's tables.

Derived quantities:
- **t\* = D0/γ** is the time at which D has doubled. A small t\* means release is driven by γ.
  It is defined for D1 and S1.
- **α = γR0²/D0²** is the static-model dimensionless group. It applies **only to S1/S2**. Do
  not compute α for D1, because the moving boundaries break the scaling.
- **β = Z0/R0** is the aspect ratio. At β = 0.5 the radial and axial surface areas are equal.
  Release is most sensitive to β when β is small (disk-like tablets).

---

## 2. Nuances in interpreting the four cases

1. **The γ values of D1 and S1 are not comparable.** D1 must overcome a diffusion path that
   lengthens as the tablet swells, so γ_D1 > γ_S1 in every case. In MS5, γ_D1/γ_S1 ranged
   from 4.9 to 22, with a median of 8.1. Compare γ only within one model.
2. **D1 and S1 fit equally well in-sample.** In MS5, S1 had the lowest AICc in 12 of 24
   formulations and D1 in 11. Their median RMSEs were about 1.5 % (APAP) and 0.4 % (IBU). An
   S1 with a suitable γ mimics the swelling system. The difference shows up only when
   **extrapolating** (§5).
3. **D2 usually fits worst, often worse than S2.** Swelling dilutes the drug and makes the
   axial integral saturate, so a constant D first over-predicts and then under-predicts. Both
   D2 and S2 show this S-shaped residual pattern, and it is the evidence that D must increase
   with time. Keep D2 and S2 as diagnostic baselines, not as candidate models. If D2 or S2
   fits as well as D1 or S1, the system is close to Fickian with constant D.
4. **D0 can collapse to its lower bound (1e-4 × 10⁻⁶).** In MS5 this happened in D1 for
   IBU/K100LV at 20, 40 and 60 % and for APAP/K15M at 60 %, and in S1 for IBU/K100LV at 60 %.
   The profile is then nearly zero-order and driven entirely by γ, and D0 is **not
   identifiable**. Report it as "D0 ≈ 0 (bound)", cite γ and t\*, and do not interpret the D0
   value. The CI from `profile_ci` will have a lower bound of 0.
5. **Release up to about 20 % does not depend on γ.** The sensitivity analysis (MS5 Fig. 2a)
   showed that the first ~20 % of release is set by D0 and geometry alone; γ controls the
   slope after that. **A dataset that stops below about 20-30 % released cannot separate D0
   from γ.** The D0–γ correlation in `fits.csv` is then close to −1.
6. **Low solubility (zero-order) and high solubility (first-order) both fit.** In MS5, the
   APAP (BCS I) profiles were first-order or Fickian, and the IBU (BCS II) profiles were close
   to zero-order. The same four models fitted both. The regime shows up in t\*: IBU has a
   small t\* (γ dominates), and APAP has a larger one.

### Assumptions to keep in mind (state them when reporting)

- **The drug is treated as dissolved and mobile at t = 0.** In practice c0 ≫ cs, even for
  APAP, so the real system is a Higuchi-type dispersion. Dissolution-limited behavior is
  absorbed into D0 and γ, which makes them *effective* parameters, not molecular
  diffusivities.
- **There is no erosion term.** For low-viscosity grades (K100LV) and long times, γ also
  absorbs erosion and growing porosity. A large γ on K100LV does not mean faster diffusion
  alone.
- **Swelling is affine (uniform), not front-based.** The model does not resolve the
  swelling, diffusion and erosion fronts. It captures how the path length grows on average.
- **D(t) is linear and unbounded.** This is accepted for the release window. It means D keeps
  growing during extrapolation, so very long predictions (several times the training window)
  tend to run fast toward 100 %. A saturating D(t) is future work.
- **Perfect sink.** Check sink conditions for the solubility and volume used.
- **The swelling inputs come from placebo HPMC (no drug).** Drug loading changes swelling.
  The preloaded defaults are for the drug-free grade.

---

## 3. Preloaded swelling (for D1/D2 without swelling data)

`swelling.py:DEFAULT_SWELLING` holds the primary swelling fits of MS5. They are stored as
ratios plus rates, so they can be transferred to any initial tablet size.

| Grade | Rf/R0 | Zf/Z0 | kr (1/h) | kz (1/h) | Source tablet | Usable window | R² (r / z) |
|---|---|---|---|---|---|---|---|
| K15M | 1.356 | 4.646 | 0.251 | 0.310 | 400 mg, 12.7 × 2.79 mm | 0–12 h | 0.968 / 0.970 |
| K100M | 1.434 | 4.786 | 0.698 | 2.206 | 200 mg, 12.75 × 1.52 mm | 0–14 h | 0.945 / 0.956 |
| K100LV | 1.264 | 3.748 | 1.442 | 2.341 | 200 mg, 12.75 × 1.52 mm | 0–3 h only | 0.979 / 0.950 |

`EXTRA_SWELLING` adds K4M (200 mg) and `K15M@200mg`, which is used for the rate-transfer
check below.

**Transfer ("scaled" mode, as in MS5).** R0 and Z0 come from the release tablet. Then
Rf = R0·(Rf/R0) and Zf = Z0·(Zf/Z0), and kr and kz are used as-is.

**Precautions:**
- **The K100M and K100LV defaults come from 200 mg tablets, half as thick as a 400 mg release
  tablet.** Axial hydration is diffusive, so its time scale grows with thickness². K15M shows
  this directly: the 200 mg kz is 1.065/h and the 400 mg kz is 0.310/h, a 3.4× difference.
  `rate_scaling="axial_L2"` rescales kz by (Z0_source/Z0)². This predicts the 400 mg K15M kz
  from the 200 mg value to within 7 % (0.288 vs 0.310). kr is left unchanged and still runs
  about 1.7× fast for a twice-thick tablet. MS5 used `"none"`. **Run both and report the
  difference as a swelling-sensitivity check** whenever the release tablets differ in
  thickness from the source.
- **K100LV: only 0–3 h of swelling was usable** (the gel erodes after that), so Rf and Zf are
  extrapolated asymptotes. Treat D1/D2 on K100LV as the least certain fits, and lean on S1.
- **For other grades, or with your own swelling data:** pass `--swelling-data swell.csv`
  (`grade, time_h, diameter_mm, height_mm`, with t = 0 dry rows), or call `fit_swelling()`.
  Drop late points where dimensions shrink (erosion) with `drop_after_h`, because the
  saturation curve cannot follow shrinkage.
- **If no swelling is available at all**, S1 and S2 need only R0 and Z0. S1 is a legitimate
  primary model in-sample.

---

## 4. Fitting procedure and precautions

The defaults below follow the MS5 batch fit. Change them only on purpose.

1. **Fit the mean profile, as a fraction from 0 to 1**, with unweighted least squares.
   Replicates enter through the mean. Use the per-tablet data for the bootstrap.
2. **Use the nominal dose.** Do not renormalize to the final plateau, because M∞ = dose in
   this model. Values above 100 % stay as measured; flag them for assay rather than clipping.
3. **Geometry: R0 = diameter/2 and Z0 = thickness/2 (half-height).** Using the full height
   for Z is the most common error. It makes axial release 4× too slow.
4. **Use one fit window for all formulations being compared.** MS5 used 0–8 h for every
   compiled study. Points beyond the window are held out, not dropped silently.
5. **Scale the parameters.** The fit works in D0 × 10⁶ and γ × 10⁹ with `x_scale="jac"` and
   tolerances of 1e-12. Unscaled fits stop early; the original script was 14 % off in D0.
6. **Start from a grid search, then multistart.** A log grid ranks starting points and the
   best of 6 local fits is kept. D0 and γ trade off along a curved valley, so a single start
   can stop on the valley floor away from the minimum.
7. **Keep γ ≥ 0.** D should not decrease as the matrix hydrates. Check `bound_hit` in every
   result (see §2 item 4).
8. **Use the profile-likelihood CI (`profile_ci`), not only the Jacobian CI.** The Jacobian
   interval is symmetric and often too narrow on one side of the valley. MS5 Fig. 8 uses the
   profile CIs. A lower bound of 0 means the data do not bound that parameter from below.
9. **Compare models with AICc,** which penalizes the extra parameter of D1 and S1. R² is
   uninformative here (all four models give R² > 0.95). MS5 also reports a bootstrap-median
   f2, where f2 ≥ 50 means similar. `f2()` in this package is the plain, untruncated f2 of
   model against mean.
10. **Empirical models fit better in-sample.** Korsmeyer–Peppas beat S1 in RMSE in 24 of 24
    formulations, and Weibull in 22 of 24, over 0–8 h. That is expected for flexible
    2-parameter curves. Do not argue mechanism from in-sample fit. The case for this model is
    physical interpretation plus extrapolation.
11. **Series truncation.** 200 terms per direction leave M(0) = 0.3 %. The error is below
    0.01 percentage points after about 15 min, so a t = 0 point is harmless. Do not lower
    `N_TERMS` below about 100.

---

## 5. Extrapolation: predicting the full profile from a short window

The parameters have a physical form: D grows linearly and the boundaries saturate on measured
time scales. A fit to the early profile therefore fixes the whole curve. The empirical models
have no such structure and drift once they leave the training data.

**Evidence from MS5** (APAP 40 %/K15M, 30 h profile, trained on t ≤ 8.21 h ≈ 50 % released;
the test set is 8.2–30 h):

| Model | test RMSE | test f2 (plain) | Note |
|---|---|---|---|
| D1 | 0.023 | 80 | best |
| S1 | 0.038 | 71 | |
| semi-empirical | 0.043 | 68 | 1−exp(−√(1/τi+(t/τ0)²)) |
| Korsmeyer–Peppas | 0.071 | 57 | best in-sample, worse out of sample |
| Weibull | 0.075 | 56 | |
| S2 | 0.132 | 44 | constant D under-predicts late release |
| D2 | 0.274 | 28 | |

**How short can the window be?** A training-window scan on the same profile (observed t80
about 18 h) gave:

| Cutoff (h) | Released at cutoff | D1 test f2 | D1 t80 (h) | S1 test f2 | S1 t80 (h) |
|---|---|---|---|---|---|
| 2 | 20 % | 59 | 14.8 | 38 | 9.7 |
| 4 | 33 % | 60 | 15.4 | 44 | 12.0 |
| 6 | 40 % | 66 | 16.2 | 51 | 14.0 |
| 8.2 | 50 % | 80 | 17.6 | 71 | 16.7 |
| 12 | 59 % | 93 | 18.5 | 91 | 18.2 |

Points to take from this:
- **D1 degrades gracefully, while S1 does not.** The swelling boundaries carry information
  that S1 has to infer from the release data. With swelling data or defaults available, D1 is
  the extrapolation model.
- **Both models under-predict t80 when the window is too short.** They read too much of the
  early curvature as γ. The bias shrinks as the window extends.
- **Rule of thumb:** train until **≥ 40-50 % released**, with at least about 8 time points,
  for a usable prediction (f2 ≥ 60). Below 20-30 % released, D0 and γ are not separable
  (§2 item 5).

**Recommended workflow on a new dataset:**
1. Pick one or two formulations that **have** the full profile. Run `train_window_scan()` over
   several cutoffs to measure how short a window your system tolerates. This calibrates the
   rule of thumb for your API, grades and apparatus.
2. Fit the short-window formulations with `extrapolate(..., cutoff_h, t_pred_h,
   per_tablet=...)`. The **tablet bootstrap** band is the honest one. Without per-tablet data,
   the band falls back to Jacobian draws, which understate the D0–γ valley.
3. Report the predicted Q(t), t50 and t80 with their bands. Also report the **D1 vs S1
   agreement**: if the two disagree by more than the band, the window is too short.
4. Flag any prediction beyond about 3× the training window. The unbounded linear D(t) makes
   late predictions optimistic.
5. When held-out points exist, report test RMSE and f2 (`test_scores.csv`), not training fit.

---

## 6. Parameter trends: D0 by API, γ by formulation composition

**This is the standard way the fitted parameters are read in this project. Always do it
once every formulation is fitted.** A single D0 or γ means little on its own. The physical
interpretation comes from how the parameters line up across formulations:

- **D0 is an API property.** Summarise it per API (median, range). Compare APIs through it.
  It should not trend strongly with the polymer.
- **γ is a formulation property.** Trend it against the composition through the
  **weighted-average HPMC molecular weight of the tablet**:

      MW_eq = Σ_i w_i · MW_i        w_i = mass fraction of HPMC grade i in the whole tablet

  Drug, lactose and other non-polymer excipients count as MW 0. For a binary drug/HPMC
  tablet this reduces to the MS5 form, MW_eq = (1 − x_drug)·MW_HPMC. A higher MW_eq means a
  stronger, more entangled gel, and γ is expected to fall.
- The nominal grade MWs used in MS5 are K100LV 29, K15M 134 and K100M 267 kDa
  (`trends.HPMC_MW_KDA`). K4M = 86 kDa is included as a literature value that was **not**
  used in MS5. Confirm it, and add any other grade, from supplier data
  (`mw_eq(..., mw_kda={...})`).

**What MS5 found** (24 formulations: APAP and IBU × K100LV/K15M/K100M × 20–60 % load):

| Quantity | APAP (BCS I) | IBU (BCS II) | Reading |
|---|---|---|---|
| D0, S1 median (× 10⁻⁶ mm²/s) | 5.2 | 0.18 | about 29× apart: D0 identifies the API |
| D0 vs MW_eq, S1 (Spearman ρ) | −0.19 | 0.2–0.4 | no polymer trend |
| γ vs MW_eq, S1 (ρ) | **−0.90** | −0.52 | γ falls as the gel strengthens |
| γ vs MW_eq, D1 (ρ) | −0.73 | −0.44 | same direction |
| γ_D1/γ_S1 vs MW_eq (ρ) | 0.72 | 0.80 | more swelling, so S1's γ absorbs more geometry |

For IBU, γ is not a continuous trend: K100LV sits about 5× above K15M/K100M. This was read
as a threshold (erosion) effect. Drug load enters through MW_eq. In K15M, γ rose with load,
which made release more linear.

**How to run it:** add the composition columns (§0) and the CLI writes `trends.csv`
(Spearman ρ, p, log–log slope, median and range per API × model × parameter) and
`trends.png`. From Python:

```python
from swelling_model.trends import mw_eq, parameter_trends, plot_trends
fits["mw_eq"] = [mw_eq({"K15M": 60}), ...]        # or let the CLI fill it
parameter_trends(fits)                            # needs api, model, mw_eq, D0_e6, gamma_e9
```

**Precautions for the trends:**
- **Trend each model on its own (S1 with S1, D1 with D1).** γ_D1 and γ_S1 differ by 5–22×
  (§2 item 1), so pooling them across models is meaningless.
- **S1 gives the cleaner trends.** S1 does not depend on transferred swelling. D1's γ
  inherits any error in the swelling defaults, especially for K100M/K100LV from 200 mg
  tablets (§3). If a D1 trend disagrees with the S1 trend, suspect the swelling input first.
- **Exclude D0 values at the lower bound** from the D0 trend. They are not identifiable
  (§2 item 4). `parameter_trends` drops them and counts them in `n_D0_at_bound`.
- **Use the profile CIs (`D0_prof_lo/hi`, `gamma_prof_lo/hi`) as error bars**, on log axes.
  Where the lower bound is 0, draw an arrow; do not clip the bar.
- **Swelling for HPMC blends.** The defaults are per grade. For a blend, use the swelling of
  the majority grade, or measure it. Note which you did, because D1's γ, and so the D1
  trend, depends on it.
- **MW_eq mixes linearly in MW.** That is a simplification (solution viscosity mixes closer
  to log-linearly). If γ correlates poorly with MW_eq across blends, also try log-weighted MW
  or an entanglement descriptor (polymer volume fraction over the entanglement density)
  before concluding there is no polymer effect.
- **Report ρ with n and p.** With 3–4 formulations per API a trend is suggestive at best.
- **Use the trend to predict, not only to describe.** A fitted γ(MW_eq) per API and the API's
  median D0 give D0 and γ for an *untested* composition, which `fraction_released` turns into
  a predicted profile. Check such a prediction against at least one held-out formulation
  before relying on it.

---

## 7. Files

| File | Contents |
|---|---|
| `model.py` | series solution, static and dynamic integrals, `fraction_released`, t*, α, t_x |
| `trends.py` | `mw_eq`, composition parsing, `parameter_trends`, `plot_trends` |
| `swelling.py` | preloaded defaults, tablet transfer, `rate_scaling`, `fit_swelling` |
| `fitting.py` | `fit_model`, `fit_all`, `profile_ci`, `score`, `extrapolate`, `train_window_scan` |
| `__main__.py` | CLI and synthetic demo |
| `../tests/test_swelling_model.py` | quadrature check of the integrals, limits, parameter recovery and extrapolation, all on synthetic data |
