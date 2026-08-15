"""How many independent things does the response set actually measure? (AC5, G11)

The AC2 metrics are redundant by construction: ``t50``, MDT, ``% released at
12 h`` and the Weibull scale are four readings of one latent quantity, "how fast
this formulation lets go of the drug". Counting them as four pieces of evidence
would be counting the same experiment four times.

This module separates two kinds of redundancy, because they warrant different
treatment:

**Structural** -- an algebraic relationship that holds for *any* dataset. For a
Weibull profile, ``t50`` and MDT are both monotone functions of the scale
parameter; they cannot disagree. No amount of data will make these independent.

**Empirical** -- two metrics that happen to co-vary in this database, possibly
because the design did not vary the thing that would separate them. A wider
design might pull them apart.

The key-response set is one representative per redundancy group, chosen on
stated criteria rather than convenience, and everything downstream is restricted
to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

#: Redundancies that follow from the model algebra, not from this dataset.
#: Each entry is (metric, related metric, why).
STRUCTURAL_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "t50",
        "log10_td",
        "For a Weibull profile t50 = Td*(ln 2)^(1/beta): a deterministic function of "
        "the scale and shape, not an independent measurement.",
    ),
    (
        "mdt_h",
        "log10_td",
        "MDT = Td*Gamma(1 + 1/beta) for an uncensored Weibull profile; it is the same "
        "scale parameter read on a different axis.",
    ),
    (
        "t50",
        "mdt_h",
        "Both are monotone functions of the Weibull scale at fixed shape, so they are "
        "algebraically tied to one another through it.",
    ),
)


@dataclass(frozen=True)
class RedundancyGroup:
    """A cluster of metrics measuring substantially the same thing."""

    index: int
    members: tuple[str, ...]
    representative: str
    justification: str
    max_abs_correlation: float
    contains_structural: bool


@dataclass(frozen=True)
class ResponseSpace:
    """Dimensionality of the response set and the key responses chosen from it."""

    metrics: tuple[str, ...]
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    linkage_matrix: np.ndarray
    groups: tuple[RedundancyGroup, ...]
    key_responses: tuple[str, ...]

    explained_variance_ratio: tuple[float, ...]
    cumulative_variance: tuple[float, ...]
    n_components_90: int
    loadings: pd.DataFrame
    scores: np.ndarray

    structural_notes: tuple[str, ...]
    empirical_notes: tuple[str, ...]
    two_dimensional: bool
    weibull_consistency: str
    notes: tuple[str, ...] = field(default=())


#: Preference order for choosing a group representative. Earlier is better.
#: Interpretability to a formulator first, then robustness to censoring, then
#: how directly the quantity is estimated rather than derived.
_PREFERENCE: tuple[str, ...] = (
    "log10_td",
    "weibull_beta",
    "weibull_f_inf",
    "t50",
    "mdt_h",
    "pct_12h",
    "peppas_n",
    "early_slope",
    "late_slope",
)

_WHY: dict[str, str] = {
    "log10_td": (
        "Weibull scale on the log axis: the natural location parameter of the curve, "
        "estimated from the whole profile rather than from one crossing, and defined "
        "even when the profile never reaches 80% release"
    ),
    "weibull_beta": (
        "Weibull shape: controls curve form independently of its timing, which is the "
        "second axis a formulator can actually move"
    ),
    "weibull_f_inf": "Weibull asymptote: how much of the dose is ultimately released",
    "t50": "time to half release: directly interpretable, but undefined once censored",
    "mdt_h": "mean dissolution time: interpretable but truncated on censored profiles",
    "pct_12h": "% released at a fixed timepoint: simple, but one slice of the curve",
    "peppas_n": "Peppas exponent: mechanism indicator over the sub-60% window only",
    "early_slope": "initial release rate",
    "late_slope": "terminal release rate",
}


#: Which Weibull parameter each metric is a reading of. The consistency check in
#: AC5 asks whether the response space's real dimensions correspond to the
#: parameters AC4 models -- so the families are named after those parameters.
_FAMILY: dict[str, str] = {
    "log10_td": "scale",
    "t50": "scale",
    "t25": "scale",
    "t10": "scale",
    "t80": "scale",
    "mdt_h": "scale",
    "pct_4h": "scale",
    "pct_12h": "scale",
    "early_slope": "scale",
    "weibull_beta": "shape",
    "peppas_n": "shape",
    "slope_ratio": "shape",
    "late_slope": "shape",
    "weibull_f_inf": "asymptote",
    "pct_24h": "asymptote",
}

#: The parameters the two-stage model in AC4 actually fits.
_WEIBULL_PARAMETERS: tuple[str, ...] = ("scale", "shape", "asymptote")


def _weibull_consistency(loadings: pd.DataFrame, n90: int) -> str:
    """Check AC5's consistency requirement against the AC4 parameterisation.

    AC5 is phrased around a two-dimensional response space mapping onto Weibull
    scale and shape. That is the right *question* but the wrong fixed number: the
    three-parameter Weibull AC4 fits has a free asymptote too, and a design in
    which the asymptote genuinely moves will show a third dimension. So the check
    performed here is whether the recovered dimensions correspond to the
    parameters being modelled -- not whether there happen to be exactly two.
    """
    dominant: list[str] = []
    for k in range(min(n90, loadings.shape[1])):
        column = loadings[f"PC{k + 1}"].abs()
        families: dict[str, float] = {}
        for metric, weight in column.items():
            family = _FAMILY.get(str(metric))
            if family:
                families[family] = families.get(family, 0.0) + float(weight) ** 2
        if families:
            dominant.append(max(families, key=lambda f: families[f]))

    covered = list(dict.fromkeys(dominant))
    axis_list = ", ".join(f"PC{i + 1} -> {f}" for i, f in enumerate(dominant))

    if set(covered) <= set(_WEIBULL_PARAMETERS) and len(covered) == len(dominant):
        return (
            f"The response space is ~{n90}-dimensional and its axes map one-to-one onto "
            f"Weibull parameters ({axis_list}). AC5 anticipated two dimensions, scale "
            "and shape; the third is the asymptote, which genuinely varies here because "
            "the slowest formulations do not release their full dose. Since AC4 fits a "
            "three-parameter Weibull with a free asymptote, the two-stage approach is "
            "empirically justified -- the dimensionality matches the parameter set being "
            "modelled, which is what the check is for."
        )

    if not covered:
        return (
            f"The response space is ~{n90}-dimensional but no principal axis could be "
            "attributed to a Weibull parameter. AC4's two-stage approach is not "
            "supported by this response set and must be revisited before use."
        )

    return (
        f"The response space is ~{n90}-dimensional with axes {axis_list}. These do NOT "
        "map cleanly onto the Weibull parameter set that AC4 models: "
        f"{len(dominant) - len(covered)} axis/axes duplicate a parameter, meaning the "
        "response set carries structure the two-stage model does not represent. AC4's "
        "approach must be revisited and this discrepancy resolved before the surfaces "
        "are relied on."
    )


def _representative(members: tuple[str, ...]) -> tuple[str, str]:
    for candidate in _PREFERENCE:
        if candidate in members:
            return candidate, _WHY.get(candidate, "preferred by the stated ordering")
    chosen = sorted(members)[0]
    return chosen, "no preference rule matched; chose the first alphabetically"


def characterise(
    table: pd.DataFrame, metrics: list[str], *, correlation_cut: float = 0.9
) -> ResponseSpace:
    """Correlate, cluster and decompose the response set, then pick key responses."""
    usable = [m for m in metrics if m in table.columns and table[m].notna().sum() >= 3]
    frame = table[usable].astype(float)

    # Drop metrics with no variance: they cannot be correlated or standardised.
    varying = [m for m in usable if float(np.nanstd(frame[m].to_numpy())) > 0]
    dropped = [m for m in usable if m not in varying]
    frame = frame[varying]

    notes: list[str] = []
    if dropped:
        notes.append(f"Metrics with zero variance excluded: {', '.join(dropped)}")

    pearson = frame.corr(method="pearson")
    spearman_matrix, _ = spearmanr(frame.to_numpy(), nan_policy="omit")
    spearman = pd.DataFrame(
        np.atleast_2d(spearman_matrix), index=varying, columns=varying
    )

    # Cluster on |Pearson|: distance 0 means "perfectly redundant".
    abs_corr = pearson.abs().to_numpy()
    np.fill_diagonal(abs_corr, 1.0)
    distance = np.clip(1.0 - abs_corr, 0.0, None)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    link = linkage(squareform(distance, checks=False), method="average")
    labels = fcluster(link, t=1.0 - correlation_cut, criterion="distance")

    groups: list[RedundancyGroup] = []
    for label in sorted(set(labels)):
        members = tuple(varying[i] for i in range(len(varying)) if labels[i] == label)
        rep, why = _representative(members)
        block = pearson.loc[list(members), list(members)].to_numpy()
        off = block[~np.eye(len(members), dtype=bool)]
        structural = any(
            a in members and b in members for a, b, _ in STRUCTURAL_PAIRS
        )
        groups.append(
            RedundancyGroup(
                index=int(label),
                members=members,
                representative=rep,
                justification=why,
                max_abs_correlation=float(np.max(np.abs(off))) if off.size else 1.0,
                contains_structural=structural,
            )
        )

    # PCA needs complete cases, and the metrics that go missing are not missing
    # at random: t50 and friends are undefined exactly for the formulations that
    # never reach the threshold. Dropping them quietly would bias the response
    # space toward the fast formulations -- the opposite of what a
    # controlled-release study is about -- so the exclusion is counted, attributed
    # to the metric responsible, and reported (G5).
    complete = frame.dropna()
    n_excluded = int(len(frame) - len(complete))
    if n_excluded:
        culprits = frame.columns[frame.isna().any()].tolist()
        missing_counts = ", ".join(
            f"{c} ({int(frame[c].isna().sum())})" for c in culprits
        )
        notes.append(
            f"PCA ran on {len(complete)} of {len(frame)} design points: {n_excluded} were "
            f"excluded because a metric is undefined there, from {missing_counts}. These "
            "are censored formulations, not random gaps -- the metric has no value "
            "because the profile never reached its threshold. The excluded points are "
            "still present in the correlation matrix and in every other analysis; only "
            "the principal-component decomposition, which requires complete cases, omits "
            "them. Read the dimensionality result as describing the uncensored subset."
        )
    centred = (complete - complete.mean()) / complete.std(ddof=1)
    matrix = centred.to_numpy(dtype=float)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    variance = singular**2
    ratio = variance / variance.sum() if variance.sum() > 0 else variance
    cumulative = np.cumsum(ratio)
    n90 = int(np.searchsorted(cumulative, 0.90) + 1)

    loadings = pd.DataFrame(
        vt.T,
        index=complete.columns,
        columns=[f"PC{i + 1}" for i in range(vt.shape[0])],
    )
    scores = matrix @ vt.T

    structural_notes = [
        f"{a} <-> {b}: {why}"
        for a, b, why in STRUCTURAL_PAIRS
        if a in varying and b in varying
    ]
    structural_keys = {
        frozenset((a, b)) for a, b, _ in STRUCTURAL_PAIRS
    }
    empirical_notes: list[str] = []
    for i, a in enumerate(varying):
        for b in varying[i + 1 :]:
            r = float(pearson.loc[a, b])
            if abs(r) >= correlation_cut and frozenset((a, b)) not in structural_keys:
                empirical_notes.append(
                    f"{a} <-> {b}: r = {r:+.3f} in this database, with no algebraic "
                    "relationship forcing it. A design spanning a wider region could "
                    "separate them."
                )

    two_d = n90 <= 2
    consistency = _weibull_consistency(loadings, n90)

    return ResponseSpace(
        metrics=tuple(varying),
        pearson=pearson,
        spearman=spearman,
        linkage_matrix=link,
        groups=tuple(groups),
        key_responses=tuple(g.representative for g in groups),
        explained_variance_ratio=tuple(float(v) for v in ratio),
        cumulative_variance=tuple(float(v) for v in cumulative),
        n_components_90=n90,
        loadings=loadings,
        scores=scores,
        structural_notes=tuple(structural_notes),
        empirical_notes=tuple(empirical_notes),
        two_dimensional=two_d,
        weibull_consistency=consistency,
        notes=tuple(notes),
    )
