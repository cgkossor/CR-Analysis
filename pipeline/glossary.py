"""Definitions for every statistic and tunable constant in the pipeline.

One source, three consumers: hover tooltips in the dashboard, ``docs/parameters.md``,
and the diagnostics report. A number a reader cannot interpret is not evidence,
and a constant whose meaning lives only in the head of whoever set it is a
landmine for whoever inherits the study.

Each entry carries what the thing *is*, its units, how to read it, and -- where it
matters -- when it misleads. That last field earns its keep: most of these
quantities have a regime where the obvious reading is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

from pipeline import config


@dataclass(frozen=True)
class Term:
    """One defined quantity."""

    key: str
    label: str
    definition: str
    units: str = ""
    how_to_read: str = ""
    caveat: str = ""

    def tooltip(self) -> str:
        parts = [self.definition]
        if self.units:
            parts.append(f"Units: {self.units}.")
        if self.how_to_read:
            parts.append(self.how_to_read)
        if self.caveat:
            parts.append(f"Careful: {self.caveat}")
        return " ".join(parts)


# --- Release metrics --------------------------------------------------------
_METRICS: tuple[Term, ...] = (
    Term(
        "t50", "t50",
        "Time at which 50% of the dose has been released.",
        "hours",
        "Lower means faster release.",
        "undefined when a formulation never reaches 50%, in which case it is "
        "reported as censored rather than as a number.",
    ),
    Term(
        "t80", "t80",
        "Time at which 80% of the dose has been released.",
        "hours",
        "The conventional marker for practically complete release.",
        "many controlled-release formulations never reach 80% within 24 h. Those "
        "are censored (>24 h), not missing, and dropping them would bias every "
        "summary toward the fast formulations.",
    ),
    Term(
        "mdt", "MDT",
        "Mean dissolution time: the average time a drug molecule spends in the "
        "tablet before release, obtained by integrating the whole curve.",
        "hours",
        "A single summary of overall release rate that uses every timepoint "
        "rather than one crossing.",
        "for a profile still rising at the last measurement it is a lower bound "
        "over the observed window, not the true MDT.",
    ),
    Term(
        "pct_released", "% released",
        "Percentage of the labelled dose released at a given time.",
        "% of dose",
        "Computed as concentration x vessel volume / dose.",
        "values above 100% are ordinary near plateau. A one-sided offset across "
        "many formulations suggests the dose denominator is understated rather "
        "than that the assay is noisy.",
    ),
    Term(
        "weibull_td", "Td (Weibull scale)",
        "Weibull scale parameter: the time constant of the fitted release curve.",
        "hours",
        "Larger means slower release. Modelled on a log scale because release "
        "times are log-normal.",
        "for a profile that never approaches its own plateau, Td is extrapolated "
        "rather than estimated and carries far more uncertainty than the headline "
        "error suggests.",
    ),
    Term(
        "weibull_beta", "beta (Weibull shape)",
        "Weibull shape parameter: controls the form of the curve independently of "
        "its timing.",
        "dimensionless",
        "Below 1 gives steep early release tailing off; near 1 is "
        "first-order-like; above 1 gives a sigmoidal curve with a lag.",
    ),
    Term(
        "weibull_f_inf", "F_inf (asymptote)",
        "The fraction of dose the fitted curve approaches at infinite time.",
        "% of dose",
        "How much of the dose the formulation ultimately releases.",
        "only identified when the profile actually climbs near it. For slow "
        "formulations it is an extrapolation.",
    ),
    Term(
        "peppas_n", "Peppas n",
        "Exponent of the Korsmeyer-Peppas power law, a release-mechanism indicator.",
        "dimensionless",
        "Around 0.45 suggests Fickian diffusion; around 0.89 suggests "
        "erosion-controlled release; between the two suggests both operating.",
        "only valid on the portion of the curve at or below 60% released, and not "
        "reported at all when fewer than three points fall in that window.",
    ),
    Term(
        "f2", "f2 similarity",
        "Similarity factor between two release profiles. 100 means identical.",
        "dimensionless, 0-100",
        "At or above 50 the profiles are conventionally considered similar.",
        "the value depends heavily on which timepoints are used. Here it is "
        "computed with at most one point past 85% release, so the plateau cannot "
        "dominate a front-loaded sampling schedule.",
    ),
)

# --- Model and design statistics -------------------------------------------
_STATISTICS: tuple[Term, ...] = (
    Term(
        "r2", "R2",
        "Fraction of the variation in the response explained by the model.",
        "0-1",
        "Higher is better, but it never decreases when terms are added.",
        "on its own it says nothing about prediction. A model can have R2 near 1 "
        "and still predict a new formulation badly.",
    ),
    Term(
        "adj_r2", "adjusted R2",
        "R2 penalised for the number of terms in the model.",
        "0-1",
        "Comparable between models of different size, unlike raw R2.",
    ),
    Term(
        "pred_r2", "predicted R2 (PRESS)",
        "R2 computed from leave-one-out predictions rather than from the fit.",
        "0-1",
        "This is the honest one: it estimates how well the model predicts a point "
        "it has not seen.",
        "a large gap below adjusted R2 means the model describes this data better "
        "than it predicts new data, i.e. it is over-fitted.",
    ),
    Term(
        "adequate_precision", "adequate precision",
        "Signal-to-noise ratio: the range of predicted values divided by their "
        "average prediction error.",
        "dimensionless",
        "Above 4 the surface can be used to navigate the design space.",
        "below 4 the modelled signal is not large enough against its own noise, "
        "and the surface must not be used for optimisation.",
    ),
    Term(
        "p_value", "p",
        "Probability of seeing an effect this large if the term truly had none.",
        "0-1",
        "Below 0.05 is conventionally called significant.",
        "significance is not importance. With enough replicates a trivially small "
        "effect becomes significant; read the effect size alongside it.",
    ),
    Term(
        "lack_of_fit", "lack of fit",
        "Test of whether the model's error exceeds the experiment's own "
        "repeatability, i.e. whether the model shape is wrong.",
        "F statistic and p",
        "A significant result suggests the model form is inadequate.",
        "here the replicates are vessels from one compression batch, so pure error "
        "excludes batch-to-batch variation. The denominator is too small and the "
        "test will over-declare lack of fit.",
    ),
    Term(
        "cv_rmse", "cross-validated error",
        "Root-mean-square prediction error from leave-one-formulation-out "
        "cross-validation, in percentage points of dose released.",
        "% released",
        "The uncertainty on a prediction for a formulation the model has never "
        "seen. This is the number attached to every prediction here.",
        "do not confuse it with the replicate SD, which is within-batch analytical "
        "repeatability -- a much smaller number measuring something else entirely.",
    ),
    Term(
        "replicate_sd", "replicate SD",
        "Standard deviation across the replicate vessels of one formulation.",
        "% released",
        "How repeatable the measurement is within a single compression batch.",
        "this is NOT prediction error and must never be shown in its place. "
        "Batch-to-batch variation is unmeasured in this design.",
    ),
    Term(
        "leverage", "leverage",
        "How much influence one design point has over its own fitted value.",
        "0-1",
        "Points near 1 are barely checked by the rest of the design.",
    ),
    Term(
        "cooks_d", "Cook's distance",
        "How much every fitted value would shift if this point were removed.",
        "dimensionless",
        "Values above 1 mark a point worth inspecting.",
    ),
    Term(
        "vif", "VIF",
        "Variance inflation factor: how much a coefficient's variance is inflated "
        "by correlation with the other terms.",
        "dimensionless, >= 1",
        "Above about 10 the term is hard to separate from the others.",
        "infinite VIF means an exact dependency, not merely a strong one -- which "
        "is what a constant-sum mixture produces by construction.",
    ),
    Term(
        "spv", "scaled prediction variance",
        "Prediction variance across the design region, scaled by the number of "
        "runs so designs of different size can be compared.",
        "dimensionless",
        "Lower and flatter is better: predictions are then uniformly trustworthy "
        "across the region rather than only near the design points.",
    ),
    Term(
        "d_criterion", "D-criterion",
        "Determinant-based measure of how much information the design carries "
        "about the model coefficients.",
        "dimensionless",
        "Higher is better. Comparable between rows of the same table only.",
        "not a percentage of an optimal design; a true D-efficiency needs a "
        "D-optimal reference for the same model and region.",
    ),
    Term(
        "g_efficiency", "G-efficiency",
        "Worst-case prediction variance relative to the theoretical best.",
        "%",
        "Higher is better; this one is a genuine percentage.",
    ),
    Term(
        "box_cox", "Box-Cox lambda",
        "The power transformation best matching the response to a normal "
        "distribution.",
        "dimensionless",
        "Lambda near 0 recommends a log transform; near 1 recommends none.",
        "read the confidence interval, not the point estimate. Lambda is rarely "
        "exactly 0 or 1, and rounding it without the interval applies a "
        "transformation the data may not support.",
    ),
    Term(
        "desirability", "desirability",
        "Derringer-Suich score combining several responses into one number between "
        "0 and 1.",
        "0-1",
        "1 means every goal met perfectly; 0 means at least one goal missed "
        "entirely.",
        "the weights are a judgement, not a measurement. Two candidates with "
        "similar scores are not distinguished by the data.",
    ),
    Term(
        "censored", "censored value",
        "A value known only to lie beyond the observation window, e.g. t80 > 24 h.",
        "",
        "Censored is not missing: the experiment did happen, and the answer is "
        "known to be large.",
        "dropping censored values biases every summary toward the fast "
        "formulations, which in a controlled-release study is precisely backwards.",
    ),
)


# --- More release metrics -----------------------------------------------------
_MORE_METRICS: tuple[Term, ...] = (
    Term(
        "pct_at", "% released at 1, 2, 4, 8, 12, 24 h",
        "Percent of the dose released at that time, read off the measured curve by "
        "linear interpolation between readings.",
        "% of dose",
        "These are the quantities dissolution specifications are written against. Later "
        "times describe completeness; early times describe burst.",
        "values above 100% are measurement offsets (assay, dose denominator), not more "
        "drug than the tablet held. A run that stopped before the time has no value; it "
        "is never extrapolated.",
    ),
    Term(
        "log10_td", "log Td",
        "Base-10 logarithm of the Weibull time scale Td.",
        "log10(hours)",
        "One unit is a factor of 10 in release time; 0.3 is a factor of 2. Modelled on "
        "the log scale because release times multiply rather than add.",
    ),
    Term(
        "early_slope", "early slope",
        "Least-squares release rate over the early window of the curve.",
        "% per hour",
        "The initial (burst plus early diffusion) release rate.",
    ),
    Term(
        "late_slope", "late slope",
        "Least-squares release rate over the late window of the curve.",
        "% per hour",
        "How fast release continues once the gel layer has formed.",
    ),
    Term(
        "slope_ratio", "slope ratio",
        "Late slope divided by early slope.",
        "ratio",
        "Near 1 is close to constant-rate (zero-order) release; well below 1 is release "
        "that slows down over time.",
    ),
)

# --- Plots --------------------------------------------------------------------
_PLOTS: tuple[Term, ...] = (
    Term(
        "model_prediction", "model prediction vs measured",
        "Lines on the DoE plots are what the fitted model predicts; points and shaded "
        "bands are the measured formulations.",
        "",
        "Trust a model line where it runs through its grade's measured points. Where it "
        "leaves them, the model does not describe that region.",
        "a model line can look precise where no formulation was run. Read it only inside "
        "the tested region.",
    ),
    Term(
        "ternary_plot", "ternary (triangle) plot",
        "The standard picture of a three-component mixture. Each corner is one component "
        "at its highest possible level with the other two at their lowest tested levels "
        "(L-pseudocomponents); every blend is a point inside. One triangle per grade.",
        "",
        "Background colour is the fitted model; dots are the measured blends filled with "
        "their measured value on the same scale. A dot that stands out from its "
        "surroundings is a blend the model does not fit. Edge numbers give each "
        "component's wt%: lactose on the left, API along the bottom, HPMC on the right.",
        "blank area is outside the tested region and is not predicted.",
    ),
    Term(
        "predicted_actual", "predicted vs actual",
        "Each formulation's measured value against the model's prediction for it, with "
        "the 1:1 line.",
        "",
        "Points on the line are predicted exactly; the scatter about it is the model's "
        "error. A grade whose points sit off the line is one the model describes badly.",
        "predicted R2 below about 0.5 means the model should not be read as a prediction "
        "for that response.",
    ),
    Term(
        "piepel_trace", "Piepel response trace",
        "The model's prediction as one component rises from the average tested blend, "
        "the other two keeping their ratio above their lowest tested levels; one panel "
        "per grade, drawn inside the tested region only.",
        "",
        "A steep trace is a component that moves the response; a flat one barely does. "
        "It is the standard trace for a mixture whose components have lower limits.",
        "it is a direction through one blend. Only measured blends lying on a trace line "
        "(within TRACE_POINT_TOL_WT) are drawn on it; in this design that is the average "
        "blend on every trace, plus two more blends on the lactose trace.",
    ),
    Term(
        "interaction_plot", "interaction plot (not used)",
        "A response against one component with one model line per grade.",
        "",
        "Not drawn here: in a three-component mixture no component can change alone, so "
        "a one-component line passes through none of the measured blends. The ternary "
        "plot shows the same information honestly.",
    ),
    Term(
        "cox_trace", "Cox response trace (not used)",
        "A trace like the Piepel trace, but along the Cox direction in real proportions.",
        "",
        "Replaced by the Piepel trace, which suits components with lower limits.",
    ),
    Term(
        "reference_composition", "reference composition",
        "The single API / HPMC / lactose blend that traces and interaction plots start "
        "from, normally the centroid (average) of the tested compositions.",
        "wt%",
        "Every trace passes through it (open circle on the trace plots).",
    ),
    Term(
        "contour_plot", "contour plot",
        "The fitted response across the tested composition region, one panel per grade, "
        "API wt% against HPMC wt% with lactose as the balance.",
        "",
        "Bands of colour are levels of the response. Open circles are the formulations "
        "actually run; blank area is outside the tested region and is not predicted.",
    ),
    Term(
        "pareto_plot", "Pareto plot of effects",
        "Each model term's |standardised effect|, largest first, with the 5% "
        "significance line and the stricter Bonferroni line.",
        "|t|",
        "Bars past the line are terms the data support. Bars past the Bonferroni line "
        "survive having tested every term at once.",
    ),
    Term(
        "half_normal", "half-normal plot",
        "Each term's |standardised effect| against where it would fall if every term "
        "were pure noise.",
        "",
        "Inert terms lie on a straight line through the origin; real effects leave it, "
        "upward and to the right.",
    ),
    Term(
        "standardised_effect", "standardised effect |t|",
        "A coefficient divided by its standard error.",
        "dimensionless",
        "Above about 2 is distinguishable from noise at the 5% level.",
    ),
    Term(
        "error_bars", "error bars / SD band",
        "Plus or minus one standard deviation across the replicate vessels of one "
        "formulation.",
        "% of dose",
        "Within-batch repeatability. On probe-logged curves the bars are drawn at the "
        "nominal sampling times only, so they stay readable.",
        "replicates come from one compression batch, so this is smaller than the "
        "batch-to-batch or model prediction error.",
    ),
    Term(
        "headline_example", "example curves in headline figures",
        "Curves chosen to illustrate a result in the headline (H) figures.",
        "",
        "Chosen automatically from the analysis results.",
        f"a formulation whose mean peaks above {config.SHOWCASE_MAX_PEAK_PCT:g}% is never "
        "used as an example, but it stays in every analysis.",
    ),
)

# --- Models and methods -------------------------------------------------------
_METHODS: tuple[Term, ...] = (
    Term(
        "scheffe", "Scheffe mixture model",
        "A regression model for blends whose components sum to 100%, with no intercept. "
        "Terms are the components (api, hpmc, lactose), their blends (api*hpmc) and "
        "their crosses with grade (hpmc:v).",
        "",
        "A linear term's coefficient is the predicted response of the pure component; "
        "a blend term (api*hpmc) is how far the mixture departs from straight-line "
        "blending.",
        "pure components lie outside the tested region, so individual coefficients are "
        "not predictions anyone should use on their own.",
    ),
    Term(
        "v_coded", "v (coded grade viscosity)",
        "log10 of the HPMC grade's nominal viscosity, rescaled so the lowest grade is "
        "-1 and the highest +1.",
        "coded",
        "A term ending in :v is how a composition effect changes with grade; :v^2 lets "
        "that change be curved across the three grades.",
    ),
    Term(
        "model_reduction", "model reduction",
        "Removing model terms the data do not support, one at a time: the least "
        "significant term that nothing else depends on goes, the model is refitted, "
        "and this repeats until every remaining term is significant.",
        "",
        "Hierarchy is kept: a blend or grade term never stays without the terms it "
        "builds on. The full model is kept if reduction would predict worse.",
        "a dropped term means its effect is within the noise at this sample size, not "
        "that it is zero.",
    ),
    Term(
        "anova", "ANOVA",
        "Analysis of variance: how much of the response's spread each group of model "
        "terms accounts for, with an F-test for each.",
        "",
        "A small p next to a group means those terms move the response more than "
        "noise would.",
    ),
    Term(
        "lofo_cv", "leave-one-formulation-out (LOFO) cross-validation",
        "Each formulation in turn is left out, the model is refitted without it, and "
        "its profile is predicted. The error over all of them is the CV error.",
        "",
        "The honest estimate of how well the model predicts a formulation it has not "
        "seen. Shown on every prediction.",
    ),
    Term(
        "eta_squared", "share of spread (eta-squared)",
        "The fraction of the case x grade spread in a response explained by "
        "composition, by grade, and by their interaction. The three add to 100%.",
        "%",
        "Used in Q1 to say which lever matters more.",
        "computed on formulation means; it says nothing about prediction error.",
    ),
    Term(
        "spearman_rho", "Spearman rho",
        "Rank correlation between two quantities.",
        "-1 to 1",
        "+1: always rise together; -1: one rises as the other falls; 0: no consistent "
        "ordering.",
        "a correlation across formulations is not a cause; composition variables move "
        "together in a mixture design.",
    ),
    Term(
        "q2", "Q2 (leave-one-out)",
        "Predictive R2: how much of the spread a model explains for points it was not "
        "fitted on.",
        "up to 1",
        "Near 1 predicts well; at or below 0 predicts no better than the average.",
    ),
    Term(
        "equivalence_set", "equivalence set",
        f"All formulations whose measured profile matches a target's with f2 >= "
        f"{config.F2_SIMILAR_THRESHOLD:g}.",
        "",
        "A set spanning several grades means a grade can be substituted at that "
        "composition.",
    ),
    Term(
        "design_space", "tested region (design space)",
        "The convex hull of the compositions actually run, in each grade.",
        "",
        "Predictions inside it are interpolation; outside, extrapolation, which the "
        "tools refuse or flag.",
    ),
    Term(
        "stress_test", "design stress test",
        "Refitting on smaller and smaller subsets of the design to see how few runs "
        "reach the same conclusions.",
        "",
        "The recommended size is the smallest subset whose prediction error and "
        "conclusions match the full design; its runs are the plan for a new API.",
    ),
)

# --- Manuscript questions -----------------------------------------------------
_MANUSCRIPT: tuple[Term, ...] = (
    Term(
        "claim_status", "claim status",
        "How strongly the data back a claim: supported, directional, not supported, "
        "gated, or unavailable.",
        "",
        "Supported: clear of its uncertainty. Directional: consistent sign but size not "
        "established, or a proxy measure. Gated: needs data this run does not have "
        "(G1). Unavailable: an optional input, such as disintegration, was not given.",
    ),
    Term(
        "shear_index", "shear index (proxy)",
        "log10 of the disintegration time predicted from the tablet's dissolution time "
        "scale (pooled Deming fit of ln DT on ln Td), over the measured disintegration "
        "time.",
        "log10 ratio",
        "0: breaks up when its release speed says it should. Positive: breaks up earlier "
        "under agitation than its release speed predicts, so more shear-sensitive.",
        "a proxy from two different apparatus, not a measured response to paddle speed.",
    ),
    Term(
        "solubility_gate", "G1 solubility gate",
        "No claim about solubility is made until at least two APIs in each solubility "
        "class have been run on the same design.",
        "",
        "With fewer, solubility cannot be told apart from everything else that differs "
        "between the molecules.",
    ),
)

# --- Formulator tools ---------------------------------------------------------
_TOOLS: tuple[Term, ...] = (
    Term(
        "target_band", "target band",
        "The +/- tolerance around each target time point in the target-profile tool.",
        "% of dose",
        "Defaults are +/-5% up to 2 h and +/-10% after.",
        "a band narrower than the cross-validated error asks for more precision than "
        "the model has.",
    ),
    Term(
        "worst_miss", "worst miss (x band)",
        "The largest deviation of a candidate's predicted profile from the target, in "
        "units of that time point's band.",
        "multiples of the band",
        "Below 1: inside every band. Candidates within about 0.2 of each other are not "
        "separated by the data.",
    ),
    Term(
        "feasible_plausible", "inside every band vs within model error",
        "Inside every band: the predicted profile meets every target. Within model "
        "error: it only does once each miss is forgiven by the cross-validated error.",
        "",
        "Make and measure the second kind before relying on it.",
    ),
    Term(
        "drug_load_third", "drug-load third",
        "Low, mid or high third of the tested API wt% range, used to pick distinct "
        "candidates in the shortlist.",
    ),
)


def _constants() -> tuple[Term, ...]:
    """Every tunable constant, with its current value read from config."""
    return (
        Term(
            "VESSEL_VOLUME_ML", "vessel volume",
            "Dissolution vessel volume, used to convert assay concentration into "
            "% released.",
            "mL",
            f"Currently {config.VESSEL_VOLUME_ML:g} mL. Not present in the raw "
            "file, so it must be set here.",
            "it scales every % released value. A wrong volume rescales the whole "
            "study without any error appearing.",
        ),
        Term(
            "PLOT_MAX_TIME_H", "plot x-axis maximum",
            "Upper limit of the time axis on every plot.",
            "hours",
            f"Currently {config.PLOT_MAX_TIME_H:g} h. Edit in pipeline/config.py.",
        ),
        Term(
            "PLOT_MAX_RELEASE_PCT", "plot y-axis maximum",
            "Upper limit of the release axis on every plot.",
            "% released",
            f"Currently {config.PLOT_MAX_RELEASE_PCT:g}%. Above 100 on purpose, so "
            "real overshoot near plateau is visible rather than clipped.",
        ),
        Term(
            "CENSORING_PCT", "censoring threshold",
            "Release level defining t80 and the censoring flag.",
            "% released",
            f"Currently {config.CENSORING_PCT:g}%.",
        ),
        Term(
            "PEPPAS_MAX_PCT", "Peppas window",
            "Upper release level of the portion the Peppas power law is fitted on.",
            "% released",
            f"Currently {config.PEPPAS_MAX_PCT:g}%. The power law only describes "
            "the early, pre-plateau part of a release curve.",
        ),
        Term(
            "PEPPAS_MIN_POINTS", "Peppas minimum points",
            "Fewest points required inside the Peppas window before k and n are "
            "reported at all.",
            "count",
            f"Currently {config.PEPPAS_MIN_POINTS}. Below this, no exponent is "
            "reported rather than an unreliable one.",
        ),
        Term(
            "F2_SIMILAR_THRESHOLD", "f2 similarity threshold",
            "f2 value at or above which two profiles are called similar.",
            "dimensionless",
            f"Currently {config.F2_SIMILAR_THRESHOLD:g}, the regulatory convention.",
        ),
        Term(
            "F2_PLATEAU_PCT", "f2 plateau rule",
            "At most one timepoint beyond this release level enters an f2 "
            "calculation.",
            "% released",
            f"Currently {config.F2_PLATEAU_PCT:g}%. Stops the plateau, where all "
            "profiles agree, from dominating the statistic.",
        ),
        Term(
            "MAX_PHYSICAL_RELEASE_PCT", "asymptote ceiling floor",
            "Lower bound for the ceiling placed on the fitted Weibull asymptote.",
            "% released",
            f"Currently {config.MAX_PHYSICAL_RELEASE_PCT:g}%. The ceiling actually "
            "applied is the larger of this and the profile's own peak times a "
            "margin.",
            "it exists to stop a censored profile's asymptote running off to "
            "nothing physical, not to assert that release cannot exceed it.",
        ),
        Term(
            "ASYMPTOTE_IDENTIFIED_FRACTION", "asymptote identifiability",
            "How close to its fitted plateau a profile must climb before that "
            "plateau counts as estimated rather than extrapolated.",
            "fraction",
            f"Currently {config.ASYMPTOTE_IDENTIFIED_FRACTION:g}. Profiles below it "
            "are flagged, because their Td is an extrapolation.",
        ),
        Term(
            "TIME_CLUSTER_REL", "timepoint tolerance",
            "How far apart two timestamps may be and still count as the same "
            "nominal sample, as a fraction of the elapsed time.",
            "fraction",
            f"Currently {config.TIME_CLUSTER_REL:g}. Real sampling lands seconds to "
            "a minute off nominal; this reconciles it.",
            "too large and distinct pulls merge, silently lowering the time "
            "resolution of every profile comparison. The diagnostics warn when "
            "that happens.",
        ),
        Term(
            "R2_GAP_FLAG", "adj-vs-pred R2 gap limit",
            "Gap between adjusted and predicted R2 beyond which over-fitting is "
            "flagged.",
            "dimensionless",
            f"Currently {config.R2_GAP_FLAG:g}.",
        ),
        Term(
            "ADEQUATE_PRECISION_FLAG", "adequate precision threshold",
            "Signal-to-noise ratio below which a surface is flagged as unfit for "
            "optimisation.",
            "dimensionless",
            f"Currently {config.ADEQUATE_PRECISION_FLAG:g}, the conventional value.",
        ),
        Term(
            "PIPELINE_SEED", "random seed",
            "Seed for every resample, cross-validation split and optimiser start.",
            "integer",
            f"Currently {config.PIPELINE_SEED}. Fixed so two runs produce "
            "byte-identical output.",
        ),
    )


def groups() -> tuple[tuple[str, tuple[Term, ...]], ...]:
    """Every term, by topic, in the order the dashboard glossary lists them."""
    # Imported here: the disintegration glossary imports Term from this module.
    from pipeline.disintegration.glossary import TERMS as DISINTEGRATION

    return (
        ("Release metrics", _METRICS + _MORE_METRICS),
        ("Plots", _PLOTS),
        ("Models and methods", _METHODS),
        ("Statistics", _STATISTICS),
        ("Research questions", _MANUSCRIPT),
        ("Formulator tools", _TOOLS),
        ("Disintegration", DISINTEGRATION),
        ("Settings (pipeline/config.py)", _constants()),
    )


def all_terms() -> tuple[Term, ...]:
    return tuple(t for _, terms in groups() for t in terms)


def as_dict() -> dict[str, dict[str, str]]:
    """Glossary keyed by term, for export to the dashboard."""
    return {
        t.key: {
            "label": t.label,
            "definition": t.definition,
            "units": t.units,
            "how_to_read": t.how_to_read,
            "caveat": t.caveat,
            "tooltip": t.tooltip(),
            "group": group,
        }
        for group, terms in groups()
        for t in terms
    }


def render_glossary_md() -> str:
    """``reports/glossary.md``: every term by topic, for reading beside the figures."""
    out = ["# Glossary", "",
           "Every term used in the figures, captions, reports and dashboard. "
           "Generated by `pipeline.glossary`.", ""]
    for group, terms in groups():
        out += [f"## {group}", ""]
        for t in terms:
            line = f"**{t.label}**. {t.definition}"
            if t.units:
                line += f" *Units: {t.units}.*"
            if t.how_to_read:
                line += f" {t.how_to_read}"
            if t.caveat:
                line += f" **Careful:** {t.caveat}"
            out += [line, ""]
    return "\n".join(out)


def render_parameters_md() -> str:
    """Generate ``docs/parameters.md`` from the definitions above."""
    out: list[str] = [
        "# Parameters and definitions\n",
        "Generated by `pipeline.glossary` — do not edit by hand.\n",
        "## Tunable constants\n",
        "Every value below lives in `pipeline/config.py`. Nothing here encodes the "
        "contents of a particular database; these are method and design constants "
        "a formulator would genuinely re-specify for a new study.\n",
        "| constant | what it is | units | current value and guidance |",
        "|---|---|---|---|",
    ]
    for t in _constants():
        note = t.how_to_read + (f" **Careful:** {t.caveat}" if t.caveat else "")
        out.append(f"| `{t.key}` | {t.definition} | {t.units or '—'} | {note} |")

    out.append("\n## Release metrics\n")
    out.append("| metric | definition | units | how to read it |")
    out.append("|---|---|---|---|")
    for t in _METRICS:
        note = t.how_to_read + (f" **Careful:** {t.caveat}" if t.caveat else "")
        out.append(f"| {t.label} | {t.definition} | {t.units or '—'} | {note} |")

    out.append("\n## Model and design statistics\n")
    out.append("| statistic | definition | how to read it |")
    out.append("|---|---|---|")
    for t in _STATISTICS:
        note = t.how_to_read + (f" **Careful:** {t.caveat}" if t.caveat else "")
        out.append(f"| {t.label} | {t.definition} | {note} |")

    return "\n".join(out) + "\n"
