"""Static figures: every one rendered, journal-styled, captioned, 300 dpi.

The figures are the part of the output most likely to be pasted into a slide or
a paper without anyone rereading the code, so the conventions are checked here
rather than trusted: no titles, closed box with mirrored minor ticks, unique
IDs, and a headline folder holding exactly the hard-coded set.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from pipeline.analysis import run_analysis
from pipeline.figures import publication as pub
from pipeline.figures.headlines import HEADLINES
from pipeline.io.load import load_database
from pipeline.run import _stress

ROOT = Path(__file__).resolve().parents[1]


def _database() -> Path:
    matches = sorted(ROOT.glob("*.xlsx"))
    if not matches:
        pytest.skip("no database workbook present")
    return matches[0]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Any]]:
    from pipeline.figures.render import render_all

    analysis = run_analysis(load_database(_database()))
    out = tmp_path_factory.mktemp("figures")
    pub.STRICT = True  # any title or suptitle raises inside save()
    try:
        records = render_all(analysis, _stress(analysis), out)
    finally:
        pub.STRICT = False
    return out, records


def test_style_is_journal() -> None:
    pub.apply_style()
    rc = plt.rcParams
    assert rc["xtick.top"] and rc["ytick.right"]
    assert rc["xtick.minor.visible"] and rc["ytick.minor.visible"]
    assert rc["xtick.direction"] == "in" and rc["ytick.direction"] == "in"
    assert rc["axes.spines.top"] and rc["axes.spines.right"]
    assert not rc["axes.grid"]


def test_every_record_is_a_300_dpi_png(rendered: tuple[Path, list[Any]]) -> None:
    out, records = rendered
    assert records
    for r in records:
        path = out / r.file
        assert path.suffix == ".png" and path.exists(), r.file
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
        assert dpi is not None and round(dpi[0]) == 300, (r.file, dpi)


def test_caption_ids_are_unique(rendered: tuple[Path, list[Any]]) -> None:
    out, _ = rendered
    manifest = json.loads((out / "captions.json").read_text(encoding="utf-8"))
    ids = [m["id"] for m in manifest]
    assert len(ids) == len(set(ids))
    assert all(m["caption"].strip() for m in manifest)
    assert (out / "captions.md").exists()


def test_headline_folder_holds_exactly_the_headlines(
    rendered: tuple[Path, list[Any]],
) -> None:
    out, _ = rendered
    folder = out / "headlines"
    assert sorted(p.stem for p in folder.glob("*.png")) == sorted(HEADLINES)
    manifest = json.loads((folder / "captions.json").read_text(encoding="utf-8"))
    assert [m["name"] for m in manifest] == list(HEADLINES)
    assert (folder / "captions.md").exists()


def test_strict_mode_rejects_titles(tmp_path: Path) -> None:
    fig, ax = pub.new_figure(pub.SINGLE)
    ax.set_title("a title")
    pub.STRICT = True
    try:
        with pytest.raises(AssertionError, match="title"):
            pub.save(fig, tmp_path, "titled")
    finally:
        pub.STRICT = False
        plt.close(fig)


def test_spread_labels_enforces_the_gap() -> None:
    ys = pub.spread_labels([1.0, 1.01, 1.02, 5.0], 0.5)
    ordered = sorted(ys)
    assert all(b - a >= 0.5 - 1e-9 for a, b in pairwise(ordered))
    assert ys[3] == pytest.approx(5.0)
