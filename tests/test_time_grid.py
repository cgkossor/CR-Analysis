"""Timepoint reconciliation (ragged sampling).

Real studies sample a few seconds to a minute off the nominal time, so every
profile ends up with its own time vector. Two failure modes matter, and they
pull in opposite directions:

* **Merging too little** — profiles never align, every cross-profile statistic
  comes back NaN, and the cross-validated error becomes unreportable.
* **Merging too much** — genuinely distinct pulls collapse into one point and
  the comparison silently runs at lower time resolution than the study design.

The second is the more dangerous, because it produces plausible numbers. When
the two cannot be told apart, splitting is the safe direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import config
from pipeline.profiles.grid import (
    build_time_grid,
    paired_finite,
    project_onto_grid,
)

#: The placeholder's schedule, in hours. Front-loaded, 1-minute spacing early.
NOMINAL_MIN = [
    0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 22, 30, 45,
    60, 90, 120, 180, 240, 360, 480, 600, 720, 1080, 1440,
]
NOMINAL_H = np.array(NOMINAL_MIN, dtype=float) / 60.0


def _jittered(n_profiles: int, jitter_min: float, seed: int = 3) -> list[np.ndarray]:
    """One time vector per profile, each offset by its own small amount."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_profiles):
        shift = rng.uniform(-jitter_min, jitter_min) / 60.0
        times = NOMINAL_H.copy()
        times[1:] += shift  # t=0 is always exactly zero
        out.append(times)
    return out


class TestScheduleRecovery:
    def test_identical_vectors_are_left_alone(self) -> None:
        grid = build_time_grid([NOMINAL_H.copy() for _ in range(10)])
        assert grid.n_points == len(NOMINAL_H)
        assert not grid.collapsed
        assert grid.max_shift_h == pytest.approx(0.0)

    @pytest.mark.parametrize("jitter_s", [2.0, 5.0, 15.0, 30.0])
    def test_recovers_the_nominal_schedule_from_jitter(self, jitter_s: float) -> None:
        """Sub-minute jitter must resolve back to the schedule that was designed."""
        grid = build_time_grid(_jittered(22, jitter_s / 60.0))
        assert grid.n_points == len(NOMINAL_H), (
            f"{jitter_s:g}s jitter produced {grid.n_points} points, expected "
            f"{len(NOMINAL_H)}"
        )
        assert grid.collapsed
        assert np.allclose(grid.times_h, NOMINAL_H, atol=jitter_s / 3600.0 + 1e-9)

    def test_never_merges_distinct_nominal_timepoints(self) -> None:
        """The dangerous direction: losing resolution while still producing numbers.

        Chaining is the mechanism — 15, 18 and 22 min form an unbroken run of
        close values once jittered, and naive gap clustering swallows all three.
        """
        grid = build_time_grid(_jittered(22, 20.0 / 60.0))
        for nominal in NOMINAL_H:
            assert np.any(np.abs(grid.times_h - nominal) < 1.0 / 60.0), (
                f"the {nominal * 60:g} min pull has no corresponding grid point"
            )

    def test_jitter_near_the_sample_spacing_splits_rather_than_merges(self) -> None:
        """When the two cannot be distinguished, err toward too many points."""
        grid = build_time_grid(_jittered(22, 1.0))
        assert grid.n_points >= len(NOMINAL_H)
        assert any("WARNING" in n for n in grid.notes) or grid.n_points >= len(NOMINAL_H)

    def test_warns_when_resolution_is_lost(self) -> None:
        """Merging distinct points must be announced, not absorbed silently."""
        original = config.TIME_CLUSTER_REL
        try:
            # Deliberately over-loose, to force the bad case.
            config.TIME_CLUSTER_REL = 0.5  # type: ignore[misc]
            grid = build_time_grid(_jittered(22, 5.0 / 60.0))
        finally:
            config.TIME_CLUSTER_REL = original  # type: ignore[misc]
        assert grid.n_points < len(NOMINAL_H), "expected over-merging at this tolerance"
        assert any("WARNING" in n for n in grid.notes), (
            "resolution was lost and nothing said so"
        )


class TestProjection:
    def test_interpolates_onto_the_grid(self) -> None:
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 25.0, 50.0, 75.0])
        out = project_onto_grid(t, y, np.array([0.5, 1.5, 2.5]))
        assert np.allclose(out, [12.5, 37.5, 62.5])

    def test_does_not_extrapolate_beyond_the_measured_window(self) -> None:
        """G3: a profile that stopped at 3 h has no opinion about 24 h."""
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 25.0, 50.0, 75.0])
        out = project_onto_grid(t, y, np.array([-1.0, 1.5, 24.0]))
        assert np.isnan(out[0])
        assert np.isfinite(out[1])
        assert np.isnan(out[2]), "value invented beyond the last observation"

    def test_empty_profile_yields_all_missing(self) -> None:
        out = project_onto_grid(np.array([]), np.array([]), np.array([1.0, 2.0]))
        assert np.all(np.isnan(out))


class TestPairedFinite:
    def test_keeps_only_mutually_present_points(self) -> None:
        a = np.array([1.0, np.nan, 3.0, 4.0])
        b = np.array([1.0, 2.0, np.nan, 4.0])
        x, y = paired_finite(a, b)
        assert np.allclose(x, [1.0, 4.0])
        assert np.allclose(y, [1.0, 4.0])

    def test_no_overlap_returns_empty_rather_than_nan(self) -> None:
        """The caller must be able to detect 'not computable' without a NaN check."""
        x, y = paired_finite(np.array([1.0, np.nan]), np.array([np.nan, 2.0]))
        assert x.size == 0 and y.size == 0
