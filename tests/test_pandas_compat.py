"""The pipeline must work under pandas copy-on-write.

Copy-on-write became the default in pandas 3.0. Under it, ``DataFrame.to_numpy()``
returns a **read-only view** of the frame's own buffer rather than a fresh array,
so any in-place mutation of that result raises ``ValueError: underlying array is
read-only``.

This is a nasty class of bug because it is invisible on older pandas: the same
code runs fine on 2.x and fails on the recipient's machine. Enabling the flag
explicitly reproduces the newer behaviour on any version, so the incompatibility
is caught here rather than by whoever installs this next.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis import CANDIDATE_METRICS, run_analysis
from pipeline.io.load import load_database
from pipeline.profiles.table import build_replicate_table
from pipeline.responses.space import characterise

ROOT = Path(__file__).resolve().parents[1]


def _database() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture
def copy_on_write() -> Iterator[None]:
    """Force pandas copy-on-write for the duration of a test."""
    try:
        previous = pd.options.mode.copy_on_write
    except AttributeError:  # pragma: no cover - pandas too old to have the flag
        pytest.skip("this pandas has no copy_on_write option")
    pd.options.mode.copy_on_write = True
    try:
        yield
    finally:
        pd.options.mode.copy_on_write = previous


def test_to_numpy_is_read_only_under_copy_on_write(copy_on_write: None) -> None:
    """Confirm the fixture reproduces the condition the fix defends against.

    Without this, the tests below could pass simply because the flag did nothing.
    """
    frame = pd.DataFrame(np.eye(3), columns=list("abc"))
    array = frame.corr().abs().to_numpy()
    assert not array.flags.writeable, (
        "copy-on-write did not produce a read-only array, so the tests below are "
        "not actually exercising the failure mode"
    )


def test_response_space_survives_read_only_arrays(copy_on_write: None) -> None:
    """The original failure: np.fill_diagonal on a read-only correlation matrix."""
    db = load_database(_database())
    replicates = build_replicate_table(db)
    metrics = [m for m in CANDIDATE_METRICS if m in replicates.columns]
    frame = replicates.groupby(["case", "grade"])[metrics].mean().reset_index()

    space = characterise(frame, metrics)

    assert space.key_responses, "no key responses selected"
    assert space.n_components_90 >= 1
    assert len(space.groups) >= 1


def test_full_analysis_runs_under_copy_on_write(copy_on_write: None) -> None:
    """End to end, since one read-only array is rarely the only one."""
    analysis = run_analysis(load_database(_database()))

    assert np.isfinite(analysis.cross_validation.profile_rmse_pct)
    assert analysis.surfaces["log10_td"].fit.estimable
    assert analysis.equivalence_summary.n_targets > 0


def test_results_are_identical_with_and_without_copy_on_write() -> None:
    """The fix must not change any number, only where the bytes live."""
    database = _database()

    previous = pd.options.mode.copy_on_write
    try:
        pd.options.mode.copy_on_write = False
        without = run_analysis(load_database(database))
        pd.options.mode.copy_on_write = True
        with_cow = run_analysis(load_database(database))
    finally:
        pd.options.mode.copy_on_write = previous

    assert without.cross_validation.profile_rmse_pct == pytest.approx(
        with_cow.cross_validation.profile_rmse_pct, rel=1e-12
    )
    assert without.response_space.key_responses == with_cow.response_space.key_responses
    assert without.response_space.n_components_90 == with_cow.response_space.n_components_90
