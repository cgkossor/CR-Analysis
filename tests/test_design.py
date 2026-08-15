"""Design-diagnostic tests (AC3).

The properties asserted here are consequences of the constant-sum constraint, so
they must hold for *any* set of mixture design points, not just the one in the
current database. The fixtures are therefore small hand-built designs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.design.diagnostics import (
    evaluate_design,
    lack_of_fit_estimability,
    variance_inflation,
)
from pipeline.design.matrix import (
    ModelSpec,
    build_model_matrix,
    code_process,
    max_process_power,
    term_names,
)

# A small constant-sum design: 7 compositions on the unit simplex, 3 process levels.
_COMPS = np.array(
    [
        [0.10, 0.20, 0.70],
        [0.10, 0.60, 0.30],
        [0.225, 0.295, 0.480],
        [0.35, 0.39, 0.26],
        [0.35, 0.60, 0.05],
        [0.475, 0.295, 0.230],
        [0.60, 0.20, 0.20],
    ]
)
_PROC_LEVELS = np.array([2.0, 3.60206, 5.0])


def _crossed() -> tuple[np.ndarray, np.ndarray]:
    comp = np.repeat(_COMPS, len(_PROC_LEVELS), axis=0)
    proc = np.tile(code_process(_PROC_LEVELS, _PROC_LEVELS), len(_COMPS))
    return comp, proc


class TestProcessCoding:
    def test_three_levels_identify_at_most_quadratic(self) -> None:
        assert max_process_power(3) == 2

    def test_log_levels_are_near_equispaced_when_coded(self) -> None:
        coded = code_process(_PROC_LEVELS, _PROC_LEVELS)
        assert coded[0] == pytest.approx(-1.0)
        assert coded[-1] == pytest.approx(1.0)
        # The middle grade lands essentially at centre on the log scale.
        assert abs(coded[1]) < 0.1


class TestConstantSumIsExactlySingular:
    def test_intercept_plus_three_components_is_rank_deficient(self) -> None:
        comp, proc = _crossed()
        result = evaluate_design(comp, proc, ModelSpec("singular", "quadratic", 2))
        assert not result.estimable
        assert result.rank < result.n_terms
        # 10 composition terms spanning only 6 dimensions, across 3 process powers.
        assert result.n_terms - result.rank == 12

    def test_intercept_is_exactly_reproduced_by_the_linear_terms(self) -> None:
        """1 = (x1 + x2 + x3)/S holds for every constant-sum design."""
        comp, proc = _crossed()
        spec = ModelSpec("singular", "linear", 0)
        matrix = build_model_matrix(comp, proc, spec)
        names = term_names(spec)
        vif = variance_inflation(matrix, names)
        # Every linear component column is perfectly explained by the others
        # together with the intercept.
        assert np.isinf(vif["api"])
        assert np.isinf(vif["hpmc"])
        assert np.isinf(vif["lactose"])

    def test_pure_quadratic_is_a_combination_of_linear_and_cross_terms(self) -> None:
        """x1^2 = S*x1 - x1*x2 - x1*x3, exactly, on the simplex."""
        x = _COMPS
        total = x.sum(axis=1)
        lhs = x[:, 0] ** 2
        rhs = total * x[:, 0] - x[:, 0] * x[:, 1] - x[:, 0] * x[:, 2]
        assert np.allclose(lhs, rhs, atol=1e-12)


class TestParameterisationEquivalence:
    """Scheffe and slack span the same space, so their statistics must agree."""

    @pytest.mark.parametrize("degree", ["linear", "quadratic", "special_cubic"])
    def test_same_term_count_and_rank(self, degree: str) -> None:
        comp, proc = _crossed()
        a = evaluate_design(comp, proc, ModelSpec("scheffe", degree, 2))  # type: ignore[arg-type]
        b = evaluate_design(
            comp, proc, ModelSpec("slack", degree, 2, dropped_component="lactose")  # type: ignore[arg-type]
        )
        assert a.n_terms == b.n_terms
        assert a.rank == b.rank

    @pytest.mark.parametrize("degree", ["linear", "quadratic", "special_cubic"])
    def test_same_prediction_variance_and_leverage(self, degree: str) -> None:
        """Leverage and G-efficiency are basis-independent; they must match exactly."""
        comp, proc = _crossed()
        a = evaluate_design(comp, proc, ModelSpec("scheffe", degree, 2))  # type: ignore[arg-type]
        b = evaluate_design(
            comp, proc, ModelSpec("slack", degree, 2, dropped_component="lactose")  # type: ignore[arg-type]
        )
        assert np.allclose(a.leverage, b.leverage, atol=1e-8)
        assert a.g_efficiency == pytest.approx(b.g_efficiency, rel=1e-6)
        assert a.d_efficiency == pytest.approx(b.d_efficiency, rel=1e-6)

    def test_projections_span_the_same_column_space(self) -> None:
        """The hat matrices coincide, which is what 'same model' means."""
        comp, proc = _crossed()
        xa = build_model_matrix(comp, proc, ModelSpec("scheffe", "quadratic", 2))
        xb = build_model_matrix(
            comp, proc, ModelSpec("slack", "quadratic", 2, dropped_component="lactose")
        )
        ha = xa @ np.linalg.pinv(xa)
        hb = xb @ np.linalg.pinv(xb)
        assert np.allclose(ha, hb, atol=1e-8)


class TestLackOfFit:
    def test_not_estimable_without_replication(self) -> None:
        keys = [(c, g) for c in range(11) for g in range(3)]
        lof = lack_of_fit_estimability(keys, n_model_terms=18)
        assert not lof.estimable_at_replicate_level
        assert lof.pure_error_df_replicate == 0
        assert "NOT estimable" in lof.describe()

    def test_estimable_with_replicates_but_carries_the_caveat(self) -> None:
        keys = [(c, g) for c in range(11) for g in range(3) for _ in range(3)]
        lof = lack_of_fit_estimability(keys, n_model_terms=18)
        assert lof.estimable_at_replicate_level
        assert lof.pure_error_df_replicate == 66
        # The anti-conservatism must travel with the number, always.
        assert "anti-conservative" in lof.describe()
        assert not lof.estimable_at_design_point_level
