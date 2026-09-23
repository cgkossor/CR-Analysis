"""Definitions for the quantities this section introduces.

Same ``Term`` type as ``pipeline.glossary``. These merge into the dashboard
tooltips once the section gets a dashboard tab. Until then they are rendered
at the foot of the report.
"""

from __future__ import annotations

from pipeline.disintegration import settings
from pipeline.glossary import Term

TERMS: tuple[Term, ...] = (
    Term(
        "dt", "Disintegration time (DT)",
        "Time for the tablet to disintegrate completely in the disintegration apparatus. "
        "For an HPMC matrix this means the swollen gel has eroded away.",
        "hours",
        "Summarised per formulation as the geometric mean of the replicates.",
        "a tablet still intact at the test end is censored (>test end), not missing, and "
        "is excluded from the models rather than read as a value.",
    ),
    Term(
        "erosion_lag", "Erosion lag R",
        "Disintegration time divided by the Weibull dissolution time scale Td.",
        "ratio",
        "Above 1, the matrix outlasts its release time scale; below 1, it is gone first.",
        "the two tests use different hydrodynamics (basket and discs vs paddle), so R "
        "mixes tablet behaviour with test severity.",
    ),
    Term(
        "deming", "Deming regression",
        "A straight-line fit that allows for measurement error in x as well as y.",
        "",
        "Used for ln DT on ln Td because both are means of noisy replicates.",
        "the ratio of the two error variances is estimated from the replicates; if it is "
        "badly wrong the slope is biased.",
    ),
    Term(
        "icc", "ICC(1)",
        "Intraclass correlation: the share of total variance that lies between "
        "formulations rather than between tablets of one formulation.",
        "0–1",
        "Near 1 means formulation means are well determined relative to their differences.",
    ),
    Term(
        "dixon_q", "Dixon's Q",
        "Gap between the most extreme replicate and its neighbour, divided by the range.",
        "",
        f"Compared with the critical value at alpha = {settings.DIXON_ALPHA} for n = 3 to 10.",
        "with three replicates, Q flags about 1 in 20 formulations by chance. A flag is "
        "reported, never used to delete data.",
    ),
    Term(
        "compression", "Grade-span ratio",
        "The spread of grade effects on ln DT at matched composition, divided by the same "
        "spread for ln Td.",
        "ratio",
        "Below 1, grade separates disintegration less than it separates release.",
    ),
)


def render_markdown() -> str:
    lines = ["\n## Terms\n"]
    for t in TERMS:
        lines.append(f"- **{t.label}.** {t.tooltip()}")
    return "\n".join(lines) + "\n"
