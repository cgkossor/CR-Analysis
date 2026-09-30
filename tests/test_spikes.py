"""Spike filter: removes momentary glitches, never real release behaviour.

A bubble on a probe window or a burst of interference reads as one or two
points far off the curve. The filter must take those out while leaving alone a
steep early release, a genuine sustained excursion, ordinary noise, and sparse
manual pulls it has too little context to judge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.profiles.spikes import remove_spikes, spike_mask

T = np.arange(0.0, 24.0, 1 / 6)  # 10-min probe logging


def _curve(scale_h: float = 6.0, shape: float = 0.8) -> np.ndarray:
    return 100.0 * (1.0 - np.exp(-((T / scale_h) ** shape)))


def _noisy(seed: int = 1, sd: float = 0.6) -> np.ndarray:
    return _curve() + np.random.default_rng(seed).normal(0.0, sd, T.size)


def test_clean_and_noisy_curves_are_untouched() -> None:
    assert not spike_mask(_curve()).any()
    for seed in range(5):
        assert not spike_mask(_noisy(seed)).any()


def test_single_and_paired_spikes_are_removed() -> None:
    y = _noisy()
    y[[20, 60, 61, 100]] += [15.0, -12.0, -11.0, 8.0]
    assert np.flatnonzero(spike_mask(y)).tolist() == [20, 60, 61, 100]


def test_a_sustained_excursion_is_behaviour_not_a_spike() -> None:
    y = _noisy()
    y[80 : 80 + config.SPIKE_MAX_RUN + 4] += 10.0
    assert not spike_mask(y).any()


def test_steep_early_release_is_not_a_spike() -> None:
    fast = 100.0 * (1.0 - np.exp(-T / 0.3))
    rng = np.random.default_rng(2)
    assert not spike_mask(fast + rng.normal(0.0, 0.6, T.size)).any()


def test_first_and_last_readings_are_never_removed() -> None:
    y = _noisy()
    y[0] += 20.0
    y[-1] += 20.0
    assert not spike_mask(y)[[0, -1]].any()


def test_sparse_manual_pulls_are_left_alone() -> None:
    t = np.array([0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 18, 24], dtype=float)
    y = 100.0 * (1.0 - np.exp(-t / 6.0))
    y[5] += 25.0
    assert len(y) < config.SPIKE_MIN_READINGS
    assert not spike_mask(y).any()


def test_remove_spikes_drops_rows_per_replicate() -> None:
    frames = []
    for rep in (1, 2):
        y = _noisy(seed=rep)
        if rep == 2:
            y[50] += 20.0
        frames.append(pd.DataFrame(
            {"id": "A", "replicate": rep, "time_h": T, "pct_released": y}
        ))
    long = pd.concat(frames, ignore_index=True)
    cleaned, log = remove_spikes(long)
    assert len(log) == 1
    assert log["deviation"].iloc[0] > log["threshold"].iloc[0]
    assert len(cleaned) == len(long) - 1
    gone = long.loc[(long["replicate"] == 2)].iloc[50]
    left = cleaned[(cleaned["replicate"] == 2) & np.isclose(cleaned["time_h"], gone["time_h"])]
    assert left.empty
