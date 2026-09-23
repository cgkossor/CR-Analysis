"""Health checks for the disintegration section, with the numbers behind them.

The same PASS/WARN/FAIL/INFO contract as ``pipeline.diagnostics``. FAIL means a
result below it cannot be trusted. WARN means it constrains how results may be
read. This report stays on the local machine: it names formulations and prints
values. The privacy-safe counterpart is ``pipeline.disintegration.audit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline.diagnostics import Diagnostics, render_json
from pipeline.disintegration import settings
from pipeline.disintegration.analysis import DisintegrationAnalysis
from pipeline.disintegration.replicates import cv_offenders


def _names(items: tuple[str, ...] | list[str], limit: int = 6) -> str:
    items = list(items)
    more = f" and {len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(items[:limit]) + more


def _finite(x: float) -> bool:
    return bool(np.isfinite(x))


def collect(r: DisintegrationAnalysis) -> Diagnostics:
    d = Diagnostics()
    data, m = r.data, r.matching

    # --- Ingest -----------------------------------------------------------
    s = "Ingest"
    d.add(s, "replicate columns found", "PASS", data.n_replicate_columns)
    d.add(s, "time unit", "PASS", {1: "s", 2: "min", 3: "h"}.get(data.unit_code, "?"),
          "read from the column header, converted to hours")
    if data.per_tablet_rows:
        d.add(s, "layout", "INFO", "one row per tablet",
              "time columns added up as parts of one time; a tablet counts as still "
              "intact only where a cell is marked '>'")
    else:
        d.add(s, "test end (h)", "PASS" if data.test_end_from_sheet else "WARN",
              data.test_end_h,
              "" if data.test_end_from_sheet else
              f"no Test_end column; assumed {settings.DT_DEFAULT_TEST_END_H:g} h for censoring")
    d.add(s, "non-numeric cells", "WARN" if data.n_nonnumeric else "PASS", data.n_nonnumeric,
          "read as missing" if data.n_nonnumeric else "")
    d.add(s, "non-positive times", "FAIL" if data.n_nonpositive else "PASS", data.n_nonpositive,
          "a disintegration time must be positive; these were excluded"
          if data.n_nonpositive else "")
    d.add(s, "unrecognised columns", "INFO", data.n_unrecognised_columns)
    d.add(s, "DT points without dissolution", "WARN" if m.dt_without_dissolution else "PASS",
          len(m.dt_without_dissolution), _names(m.dt_without_dissolution))
    d.add(s, "dissolution points without DT", "WARN" if m.dissolution_without_dt else "PASS",
          len(m.dissolution_without_dt), _names(m.dissolution_without_dt))
    d.add(s, "composition disagrees with dissolution sheet",
          "FAIL" if m.composition_mismatch else "PASS", len(m.composition_mismatch),
          _names(m.composition_mismatch) + (" — the two sheets describe different tablets"
                                            if m.composition_mismatch else ""))
    d.add(s, "composition sums to 100", "WARN" if m.composition_sum_off else "PASS",
          len(m.composition_sum_off), _names(m.composition_sum_off))
    d.add(s, "grades without viscosity", "FAIL" if m.grades_without_viscosity else "PASS",
          len(m.grades_without_viscosity), _names(m.grades_without_viscosity))

    # --- Replicates -------------------------------------------------------
    s = "Replicates"
    pts = r.precision.points
    few = pts[pts["n"] < settings.DT_MIN_REPS]
    d.add(s, f"formulations with n < {settings.DT_MIN_REPS}", "WARN" if len(few) else "PASS",
          len(few), _names([f"case {c}/{g}" for c, g in zip(few["case"], few["grade"],
                                                                   strict=True)]))
    d.add(s, "replicates per formulation (min)", "INFO", int(pts["n"].min()))
    d.add(s, "replicates per formulation (max)", "INFO", int(pts["n"].max()))
    high = cv_offenders(pts[~pts["censored"]])
    d.add(s, f"formulations with CV > {settings.DT_CV_WARN:.0%}", "WARN" if len(high) else "PASS",
          len(high),
          _names([f"case {c}/{g} ({v:.0%})" for c, g, v in zip(
              high["case"], high["grade"], high["cv"], strict=True)]))
    flags = r.precision.flags
    d.add(s, "Dixon Q outliers (alpha 0.05)", "WARN" if flags else "PASS", len(flags),
          _names([f"case {f.case}/{f.grade} rep {f.replicate} ({f.dt_h:.2f} h, Q={f.q:.2f})"
                  for f in flags]) + ("; reported, not removed" if flags else ""))
    icc = r.precision.icc
    d.add(s, "ICC(1) between-formulation share", "PASS" if _finite(icc) and icc >= 0.9 else
          "WARN", icc, "formulations differ far more than tablets within one"
          if _finite(icc) and icc >= 0.9 else "replicate scatter is a large share of the total")
    for gp in r.precision.by_grade:
        d.add(s, f"pooled replicate CV, {gp.grade}", "INFO", gp.pooled_cv,
              f"median {gp.median_cv:.1%} over {gp.n_formulations} formulations")
    bf = r.precision.brown_forsythe_p
    d.add(s, "scatter differs by grade (Brown-Forsythe p)", "WARN" if _finite(bf) and bf < 0.05
          else "INFO", bf, "unequal precision by grade; weight or interpret accordingly"
          if _finite(bf) and bf < 0.05 else "")

    # --- Censoring --------------------------------------------------------
    s = "Censoring"
    cens = r.censored_points
    n_pts = len(r.matched)
    d.add(s, "formulations censored at test end", "WARN" if len(cens) else "PASS",
          len(cens),
          _names([f"case {c}/{g}" for c, g in zip(cens["case"], cens["grade"], strict=True)])
          + (" — lower bounds, excluded from models" if len(cens) else ""))
    d.add(s, "fraction censored", "WARN" if n_pts and len(cens) / n_pts > 0.25 else "INFO",
          len(cens) / n_pts if n_pts else float("nan"))
    lost = [g for g in r.grade_order
            if not (r.matched[(r.matched["grade"] == g) & ~r.matched["dt_censored"]]).shape[0]]
    d.add(s, "grades removed entirely by censoring", "FAIL" if lost else "PASS", len(lost),
          _names(lost))

    # --- Plausibility -----------------------------------------------------
    s = "Plausibility"
    p = r.plausibility
    d.add(s, "DT < t50", "WARN" if p.below_t50 else "PASS", len(p.below_t50),
          (_names(p.below_t50) + " — the matrix is gone before half the dose is out. "
           "Possible when agitation erodes gel fragments that still hold drug; confirm the "
           "end-point definition.") if p.below_t50 else f"of {p.n_compared_t50} compared")
    d.add(s, "DT < t80", "INFO", len(p.below_t80), f"of {p.n_compared_t80} with a defined t80")

    # --- Correlation ------------------------------------------------------
    s = "Correlation"
    c = r.correlation
    n_corr = c.by_key("td_h").n if c.by_key("td_h") else 0  # type: ignore[union-attr]
    d.add(s, "matched uncensored design points", "PASS" if n_corr >=
          settings.MIN_POINTS_CORRELATION else "WARN", n_corr)
    for key in ("td_h", "t80"):
        cc = c.by_key(key)
        if cc is None:
            continue
        ok = _finite(cc.spearman) and cc.spearman >= settings.RHO_EXPECTED
        d.add(s, f"Spearman rho(DT, {cc.metric.label})",
              "PASS" if ok else ("WARN" if _finite(cc.spearman) and cc.spearman > 0 else "FAIL"),
              cc.spearman,
              f"95% CI {cc.spearman_ci[0]:.2f} to {cc.spearman_ci[1]:.2f}, n={cc.n}"
              + ("" if ok else f"; below the expected {settings.RHO_EXPECTED}"))
    for w in c.within_grade:
        d.add(s, f"within-grade Pearson r(ln DT, ln Td), {w.grade}", "INFO", w.pearson,
              f"n={w.n}")
    disagree = [x.metric.label for x in c.table if _finite(x.pearson) and not x.sign_agrees]
    d.add(s, "Pearson and Spearman disagree in sign", "WARN" if disagree else "PASS",
          len(disagree), _names(disagree))
    wrong = [x.metric.label for x in c.table if x.sign_as_expected is False]
    d.add(s, "correlations against the expected sign", "WARN" if wrong else "PASS", len(wrong),
          _names(wrong))
    pc = c.partial
    d.add(s, "partial r(ln DT, ln Td | grade, composition)", "INFO", pc.r,
          f"p = {pc.p:.2g}" if _finite(pc.p) else "")

    # --- Models -----------------------------------------------------------
    s = "Models"
    a = r.grades.ancova
    if a is None:
        d.add(s, "ANCOVA", "WARN", False, "too few uncensored points per grade")
    else:
        d.add(s, "ANCOVA residual df", "PASS" if a.residual_df > 0 else "FAIL", a.residual_df)
        d.add(s, "ANCOVA model selected", "INFO", a.selected,
              f"slopes-equal p = {a.p_common_slope:.3g}")
        d.add(s, "Shapiro-Wilk on residuals (p)", "WARN" if a.shapiro_p < 0.05 else "PASS",
              a.shapiro_p)
        d.add(s, "Breusch-Pagan (p)", "WARN" if _finite(a.breusch_pagan_p) and
              a.breusch_pagan_p < 0.05 else "PASS", a.breusch_pagan_p)
        limit = 4.0 / a.n
        infl = [a.labels[i] for i in np.flatnonzero(a.cooks > limit)]
        d.add(s, f"points with Cook's D > 4/n ({limit:.2f})", "WARN" if infl else "PASS",
              len(infl), _names(infl) + (" — refit without them before quoting slopes"
                                         if infl else ""))
        p_lev = 2 * (a.n - a.residual_df) / a.n
        lev = [a.labels[i] for i in np.flatnonzero(a.leverage > p_lev)]
        d.add(s, f"high-leverage points (h > {p_lev:.2f})", "INFO", len(lev), _names(lev))
        d.add(s, "VIF of ln Td given the recipe", "WARN" if a.vif_ln_td > 10 else "PASS",
              a.vif_ln_td, "dissolution is largely determined by the recipe; its "
              "coefficient cannot be separated from composition effects"
              if a.vif_ln_td > 10 else "")
    b = r.grades
    d.add(s, "complete case x grade blocks", "PASS" if b.n_complete_blocks >= 3 else "WARN",
          b.n_complete_blocks, "blocks with every grade uncensored")
    for ra in r.doe.responses:
        t = ra.anova
        gap = t.adj_r_squared - t.pred_r_squared
        d.add(s, f"DoE on {ra.response.spec.key}: R2 / adj / pred", "WARN" if (
            not ra.usable or gap > 0.2) else "PASS",
            f"{t.r_squared:.3f} / {t.adj_r_squared:.3f} / {t.pred_r_squared:.3f}",
            f"{len(ra.kept_terms)} terms, residual df {t.residual_df}"
            + ("; R2pred far below R2adj — overfitted" if gap > 0.2 else ""))
    for mdl in r.loo:
        d.add(s, f"LOO Q2, {mdl.label}", "INFO", mdl.q2,
              f"typical error ×{mdl.fold_error:.2f}, {mdl.n_params} parameters")

    # --- Synthetic recovery ----------------------------------------------
    if r.recovery:
        s = "Synthetic recovery"
        for rec in r.recovery:
            d.add(s, rec.name, "PASS" if rec.covered else "WARN", rec.estimate,
                  f"truth {rec.truth:g}, 95% CI {rec.ci[0]:.3f} to {rec.ci[1]:.3f}")
    return d


def render_markdown(d: Diagnostics, r: DisintegrationAnalysis) -> str:
    out = ["# Disintegration diagnostics\n", f"**{d.verdict}**\n"]
    if r.is_synthetic:
        out.append("> **PLACEHOLDER DATA — SYNTHETIC, NOT EXPERIMENTAL.** "
                   "No value here is a measurement.\n")
    for title, items in (("Failures", d.failures), ("Warnings", d.warnings)):
        if items:
            out.append(f"## {title}\n")
            out += [f"- **{c.section} / {c.name}** = `{c.display}`"
                    + (f" — {c.note}" if c.note else "") for c in items]
            out.append("")
    out.append("## All checks\n")
    current = ""
    for c in d.checks:
        if c.section != current:
            current = c.section
            out += [f"\n### {current}\n", "| check | value | status | note |", "|---|---|---|---|"]
        mark = "" if c.status == "INFO" else f"**{c.status}**"
        out.append(f"| {c.name} | `{c.display}` | {mark} | {c.note} |")
    return "\n".join(out) + "\n"


def write(r: DisintegrationAnalysis, reports: Path) -> Diagnostics:
    d = collect(r)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "disintegration_diagnostics.md").write_text(
        render_markdown(d, r), encoding="utf-8", newline="\n"
    )
    (reports / "disintegration_diagnostics.json").write_text(
        render_json(d), encoding="utf-8", newline="\n"
    )
    return d
