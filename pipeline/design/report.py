"""Render the AC3 design-diagnostics report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.design.diagnostics import (
    DesignDiagnostics,
    LackOfFitEstimability,
    evaluate_design,
    lack_of_fit_estimability,
)
from pipeline.design.matrix import (
    CompositionDegree,
    ModelSpec,
    code_process,
    max_process_power,
)
from pipeline.io.load import Database


def design_points(db: Database) -> pd.DataFrame:
    """One row per (case, grade) design point, with coded process variable."""
    pts = (
        db.profiles[["id", "case", "grade", "api_wt", "hpmc_wt", "lactose_wt"]]
        .drop_duplicates(["case", "grade"])
        .sort_values(["case", "grade"])
        .reset_index(drop=True)
    )
    pts["viscosity_cp"] = pts["grade"].map(db.grade_viscosity_cp).astype(float)
    pts["log10_visc"] = np.log10(pts["viscosity_cp"])
    pts["v_coded"] = code_process(pts["log10_visc"].to_numpy())
    return pts


def analyse(
    db: Database,
) -> tuple[dict[str, DesignDiagnostics], LackOfFitEstimability, pd.DataFrame]:
    """Evaluate the design under the candidate model hierarchy."""
    pts = design_points(db)
    comp = pts[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
    proc = pts["v_coded"].to_numpy(dtype=float)

    n_levels = int(pts["v_coded"].nunique())
    power = min(max_process_power(n_levels), 2)

    results: dict[str, DesignDiagnostics] = {}

    # The model AC3 warns about: an intercept alongside all three components.
    # Evaluated, not just asserted, so the report can show the rank deficiency.
    results["singular_intercept_plus_three_components"] = evaluate_design(
        comp, proc, ModelSpec("singular", "quadratic", power)
    )

    degrees: tuple[CompositionDegree, ...] = ("linear", "quadratic", "special_cubic")
    for degree in degrees:
        spec = ModelSpec("scheffe", degree, power)
        richer = ModelSpec("scheffe", "special_cubic", power)
        results[f"scheffe_{degree}"] = evaluate_design(comp, proc, spec, candidate_spec=richer)
        slack = ModelSpec("slack", degree, power, dropped_component="lactose")
        results[f"slack_{degree}"] = evaluate_design(comp, proc, slack)

    keys = [
        (row.case, row.grade, rep)
        for row in db.profiles[["id", "case", "grade", "replicate"]]
        .drop_duplicates(["case", "grade", "replicate"])
        .itertuples()
        for rep in [row.replicate]
    ]
    design_keys: list[tuple[object, ...]] = [(c, g) for c, g, _ in keys]
    lof = lack_of_fit_estimability(design_keys, results["scheffe_quadratic"].n_terms)

    return results, lof, pts


def render_markdown(
    results: dict[str, DesignDiagnostics],
    lof: LackOfFitEstimability,
    points: pd.DataFrame,
    is_synthetic: bool,
) -> str:
    out: list[str] = []
    add = out.append

    add("# Design diagnostics (AC3)\n")
    if is_synthetic:
        add(
            "> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.** These are properties "
            "of the design geometry, which is real, but nothing here is an experimental "
            "finding.\n"
        )
    add(
        "Computed before any response is modelled. These are properties of where the "
        "experiments sit, not of what they measured.\n"
    )

    add("## Design points\n")
    add(f"- Distinct (case, grade) points: {len(points)}")
    add(f"- Cases: {points['case'].nunique()} | grades: {points['grade'].nunique()}")
    levels = points[["grade", "viscosity_cp", "log10_visc", "v_coded"]].drop_duplicates()
    levels = levels.sort_values("viscosity_cp")
    add("\n| grade | viscosity (cP) | log10 | coded |")
    add("|---|---|---|---|")
    for row in levels.itertuples():
        add(
            f"| {row.grade} | {row.viscosity_cp:,.0f} | {row.log10_visc:.3f} | "
            f"{row.v_coded:+.3f} |"
        )
    add(
        "\nThe log transform is what makes the three grades usable as a continuous "
        "factor: untransformed they sit at 100 / 4,000 / 100,000, which places the "
        "middle grade far from centre. On the log scale they are near-equispaced.\n"
    )

    add("\n## The singular model AC3 warns about\n")
    bad = results["singular_intercept_plus_three_components"]
    add(
        f"A full quadratic carrying an intercept alongside all three components lists "
        f"{bad.n_terms} terms but has rank **{bad.rank}** — "
        f"**{bad.n_terms - bad.rank} exact linear dependencies**. This is not "
        "ill-conditioning to be managed; the normal equations have no unique solution."
    )
    infinite = [t for t, v in bad.vif.items() if np.isinf(v)]
    if infinite:
        add(
            f"\nInfinite VIF (exactly reproduced by the other columns): "
            f"{', '.join(f'`{t}`' for t in infinite)}"
        )
    for note in bad.notes:
        add(f"\n- {note}")

    add("\n## Estimable models\n")
    add(
        "| model | terms | rank | estimable | cond(X) | D-criterion | A-criterion "
        "| G-eff % | max leverage |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for key, d in results.items():
        if key.startswith("singular"):
            continue
        add(
            f"| `{key}` | {d.n_terms} | {d.rank} | {'yes' if d.estimable else '**no**'} | "
            f"{d.condition_number:,.1f} | {d.d_efficiency:.3g} | {d.a_efficiency:.3g} | "
            f"{d.g_efficiency:.1f} | {d.max_leverage:.3f} |"
        )

    add(
        "\nThe Scheffe and slack rows at matching degree carry the same term count and "
        "the same rank because they span the same function space on the simplex. They "
        "are two readings of one model, not two models — which is why the A1 result, "
        "on its own, does not invalidate a surface. What invalidates it is the singular "
        "form above."
    )
    add(
        "\n**On the criterion columns.** D and A are reported as raw criterion values "
        "(`|X'X|^(1/p)/n` and `p/(n·tr (X'X)⁻¹)`), not as percentages of an optimal "
        "design: a true D-efficiency needs a D-optimal reference design for the same "
        "model and region, which is not computed here. They are comparable *between "
        "the rows of this table* and nowhere else. G-efficiency is a genuine "
        "percentage, since `p/max(SPV)` has a known theoretical optimum.\n"
    )

    add("\n## Aliasing\n")
    quad = results["scheffe_quadratic"]
    if quad.aliases:
        add(
            "Terms present in the special-cubic model but absent from the fitted "
            "quadratic, and how they would bias the fitted coefficients if real:\n"
        )
        for entry in quad.aliases:
            top = sorted(
                entry.aliased_with.items(), key=lambda kv: -abs(kv[1])
            )[:4]
            parts = ", ".join(f"`{n}` {c:+.4g}" for n, c in top if abs(c) > 1e-8)
            marker = "**exact**" if entry.exact else "partial"
            add(f"- `{entry.term}` — {marker} — loads on {parts or '(negligible)'}")
    else:
        add("No candidate terms outside the fitted model.")

    add(
        "\n### Structural aliases that hold for any 11 points\n"
        "Because the components sum to a constant, two aliases are guaranteed by the "
        "design's algebra rather than by which points were chosen:\n\n"
        "- the intercept is exactly the scaled sum of the three linear terms, "
        "`1 = (x_api + x_hpmc + x_lactose)/S`;\n"
        "- every pure quadratic is exactly a combination of linear and cross terms, "
        "`x_i^2 = S*x_i - x_i*x_j - x_i*x_k`.\n\n"
        "The second is why the Scheffe form carries no squared terms: they would add "
        "nothing. Any three-component polynomial that includes both an intercept and "
        "squared terms is over-parameterised before a single observation is taken.\n"
    )

    add("\n## Lack of fit\n")
    add(lof.describe())
    add(
        f"\n- Observations: {lof.n_observations} | distinct design points: "
        f"{lof.n_distinct_points} | replicate pure-error df: "
        f"{lof.pure_error_df_replicate}"
    )
    add(
        "\n**At the design-point level there is no replication at all** — one batch per "
        "(case, grade). Since the response surface is fitted on design-point means to "
        "avoid pseudo-replication, its own residual error carries lack of fit and "
        "batch variation together, and cannot be decomposed."
    )

    add("\n## Prediction variance\n")
    for key in ("scheffe_quadratic", "scheffe_special_cubic"):
        variance = results.get(key)
        if variance is None or not variance.fds_spv:
            continue
        spv = np.asarray(variance.fds_spv)
        add(f"\n**{key}** — scaled prediction variance over the tested composition hull:")
        add(f"- median {np.median(spv):.2f}, max {spv.max():.2f}, min {spv.min():.2f}")
        for frac in (0.5, 0.8, 0.95):
            add(f"- {frac:.0%} of the region has SPV ≤ {np.quantile(spv, frac):.2f}")
    add(
        "\nThe region sampled is the convex hull of the tested compositions, crossed "
        "with the tested viscosity range. Outside it, prediction variance is not "
        "reported because prediction itself is extrapolation (G3)."
    )

    return "\n".join(out) + "\n"
