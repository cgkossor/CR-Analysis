"""Classical DoE analysis: the arithmetic has to add up.

An ANOVA that looks plausible but decomposes wrongly is worse than none, because
it carries the authority of a familiar table. These tests check the decomposition
against an independently computed least-squares fit rather than against a
previously recorded output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pipeline.analysis import run_analysis
from pipeline.design.matrix import build_model_matrix, term_names
from pipeline.doe.responses import RESPONSES, collect
from pipeline.io.load import load_database
from pipeline.surfaces.model import fit_surface

ROOT = Path(__file__).resolve().parents[1]


def _database() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture(scope="module")
def analysis() -> Any:
    return run_analysis(load_database(_database()))


class TestAnovaDecomposition:
    def test_sums_of_squares_add_up(self, analysis: Any) -> None:
        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            t = ra.anova
            assert t.model_row.adj_ss + t.residual_ss == pytest.approx(
                t.total_ss, rel=1e-9
            ), f"{ra.response.spec.key}: model + residual != total"

    def test_no_adjusted_ss_exceeds_the_total(self, analysis: Any) -> None:
        """A reduced model must still be able to fit the mean.

        Dropping every linear term from a Scheffe model leaves something that
        cannot represent the grand mean, and the resulting "reduction in error"
        came out larger than the total sum of squares.
        """
        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            for row in ra.anova.rows:
                assert row.adj_ss <= ra.anova.total_ss * (1 + 1e-6), (
                    f"{ra.response.spec.key}: Adj SS for {row.source} exceeds the total"
                )

    def test_degrees_of_freedom_balance(self, analysis: Any) -> None:
        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            t = ra.anova
            assert t.model_row.df + t.residual_df == t.total_df

    def test_r_squared_matches_an_independent_fit(self, analysis: Any) -> None:
        """Cross-check against the least-squares path used elsewhere."""
        dp = analysis.design_points
        comp = dp[["api_wt", "hpmc_wt", "lactose_wt"]].to_numpy(dtype=float) / 100.0
        proc = dp["v_coded"].to_numpy(dtype=float)

        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            mask = ra.response.available
            labels = term_names(ra.spec)
            keep = [i for i, n in enumerate(labels) if n in set(ra.kept_terms)]
            matrix = build_model_matrix(comp[mask], proc[mask], ra.spec)[:, keep]
            fit = fit_surface(
                matrix, ra.response.values[mask], list(ra.kept_terms),
                response_name=ra.response.spec.key,
            )
            assert ra.anova.r_squared == pytest.approx(fit.r_squared, abs=1e-9)
            assert ra.anova.pred_r_squared == pytest.approx(fit.pred_r_squared, abs=1e-9)


class TestCensoredResponses:
    def test_censorable_responses_declare_their_missing_points(
        self, analysis: Any
    ) -> None:
        """G5: a response that goes undefined on the slow formulations says so."""
        for ra in analysis.doe.responses:
            if ra.response.n_missing:
                assert ra.response.coverage_note, (
                    f"{ra.response.spec.key} drops {ra.response.n_missing} "
                    "points silently"
                )

    def test_percent_responses_are_never_censored(self, analysis: Any) -> None:
        """% released at a fixed time is always defined; that is why it is here."""
        for ra in analysis.doe.responses:
            if ra.response.spec.key.startswith("pct_"):
                assert ra.response.n_missing == 0

    def test_model_adapts_when_censoring_removes_information(
        self, analysis: Any
    ) -> None:
        """Censoring is not random, so it can strip a factor level of support.

        t80 loses the slowest formulations, leaving one viscosity grade with only
        a couple of points -- too few to estimate a quadratic in log-viscosity
        even though all three grades are nominally still present. The model must
        step down rather than come back inestimable and vanish from the analysis.
        """
        t80 = analysis.doe.by_key("t80")
        if t80 is None or not t80.response.n_missing:
            pytest.skip("t80 is not censored in this database")
        assert t80.usable, "t80 became inestimable instead of reducing its model"
        assert any(
            "log-viscosity" in n or "censoring" in n or "censored" in n
            for n in t80.notes
        ), "the model was reduced without saying why"


class TestEffects:
    def test_significant_terms_rank_above_the_critical_line(
        self, analysis: Any
    ) -> None:
        for ra in analysis.doe.responses:
            if not ra.usable or not np.isfinite(ra.ranking.t_critical):
                continue
            for e in ra.ranking.effects:
                if e.significant:
                    assert e.abs_t >= ra.ranking.t_critical - 1e-9

    def test_bonferroni_line_is_stricter(self, analysis: Any) -> None:
        for ra in analysis.doe.responses:
            r = ra.ranking
            if np.isfinite(r.t_critical) and np.isfinite(r.bonferroni_t):
                assert r.bonferroni_t >= r.t_critical

    def test_interaction_is_detected_and_described(self, analysis: Any) -> None:
        """The headline finding: the levers are not additive."""
        ra = analysis.doe.by_key("pct_12h")
        if ra is None:
            pytest.skip("pct_12h not modelled")
        prof = ra.interactions[0]
        assert not prof.parallel, "expected a non-additive HPMC x grade interaction"
        assert prof.divergence > 0.15
        assert "not parallel" in prof.interpretation


class TestTakeaways:
    def test_takeaway_reflects_the_actual_numbers(self, analysis: Any) -> None:
        """A sentence that would read the same on any dataset is not a takeaway."""
        ra = analysis.doe.by_key("pct_12h")
        if ra is None:
            pytest.skip("pct_12h not modelled")
        text = ra.takeaway_anova
        assert f"{ra.anova.r_squared:.1%}" in text
        assert f"{ra.anova.pred_r_squared:.1%}" in text

    def test_does_not_call_a_small_contribution_dominant(self, analysis: Any) -> None:
        """Type III SS on a mixture can be small for every term at once.

        Calling the largest of several small contributions "dominant" would be
        confidently wrong, which is worse than saying nothing.
        """
        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            groups = [r for r in ra.anova.rows if r.is_group and r.significant]
            if not groups:
                continue
            top = max(groups, key=lambda r: r.adj_ss)
            share = top.adj_ss / ra.anova.total_ss if ra.anova.total_ss else 0.0
            if share < 0.10:
                assert "largest single source" not in ra.takeaway_anova, (
                    f"{ra.response.spec.key}: called a {share:.0%} contribution dominant"
                )

    def test_every_method_has_a_note(self, analysis: Any) -> None:
        for key in (
            "anova", "pareto", "half_normal", "cox_trace",
            "interaction", "contour", "model_summary",
        ):
            assert analysis.doe.method_notes.get(key), f"no method note for {key}"


class TestSurfaceGrids:
    def test_region_outside_the_hull_is_masked(self, analysis: Any) -> None:
        """G3: no prediction is offered where the design has no information."""
        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            for g in ra.grids:
                assert g.masked_fraction > 0.0, (
                    f"{ra.response.spec.key}/{g.grade}: nothing masked, so the grid "
                    "extrapolates beyond the tested compositions"
                )
                assert any(v is None for row in g.z for v in row)

    def test_grades_share_one_colour_scale(self, analysis: Any) -> None:
        """Per-panel scaling would hide that one grade spans far more response."""
        for ra in analysis.doe.responses:
            if not ra.usable:
                continue
            lo, hi = ra.scale
            for g in ra.grids:
                if np.isfinite(g.z_min):
                    assert g.z_min >= lo - 1e-6
                    assert g.z_max <= hi + 1e-6


def test_named_responses_cover_what_was_asked_for() -> None:
    keys = {r.key for r in RESPONSES}
    for expected in ("t50", "t80", "mdt_h", "pct_12h", "pct_24h"):
        assert expected in keys


def test_responses_are_built_from_design_points(analysis: Any) -> None:
    data = collect(analysis.design_points, analysis.replicates)
    assert data
    for d in data:
        assert d.n_total == len(analysis.design_points)
