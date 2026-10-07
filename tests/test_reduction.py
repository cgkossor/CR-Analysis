"""Model reduction must not throw away a real grade effect."""

from __future__ import annotations

import numpy as np

from pipeline.design.matrix import ModelSpec, code_process
from pipeline.surfaces.reduce import reduce_model


def _design() -> tuple[np.ndarray, np.ndarray]:
    comp = []
    for a in (0.1, 0.225, 0.35, 0.475, 0.6):
        for h in (0.2, 0.3, 0.4, 0.5, 0.6):
            lac = 1 - a - h
            if 0.05 <= lac <= 0.7:
                comp.append((a, h, lac))
    c = np.array(comp[:11])
    lv = np.log10([100.0, 4000.0, 100000.0])
    v = code_process(lv, lv)
    return np.repeat(c, 3, axis=0), np.tile(v, len(c))


def test_a_moderate_grade_effect_survives_reduction() -> None:
    """With three grades the v and v^2 terms overlap; each can look insignificant
    alone. A one-shot p-value cut dropped the whole grade effect in about four
    runs in ten at this effect size, so every grade predicted the same line."""
    comp, v = _design()
    kept_grade = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        y = 30 * comp[:, 0] + 80 * comp[:, 1] + 20 * comp[:, 2] + 2.5 * v
        y = y + rng.normal(0, 3, len(v))
        red = reduce_model(comp, v, y, ModelSpec("scheffe", "linear", 2))
        kept_grade += any(":" in t for t in red.kept_terms)
    assert kept_grade >= 36, f"grade effect lost in {40 - kept_grade} of 40 runs"


def test_no_grade_effect_is_still_removed() -> None:
    comp, v = _design()
    dropped = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        y = 30 * comp[:, 0] + 80 * comp[:, 1] + 20 * comp[:, 2] + rng.normal(0, 3, len(v))
        red = reduce_model(comp, v, y, ModelSpec("scheffe", "linear", 2))
        dropped += not any(":" in t for t in red.kept_terms)
    assert dropped >= 30, "reduction keeps grade terms that are pure noise"
