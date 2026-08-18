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


def all_terms() -> tuple[Term, ...]:
    return _METRICS + _STATISTICS + _constants()


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
        }
        for t in all_terms()
    }


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
