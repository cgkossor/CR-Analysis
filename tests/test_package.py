"""Packaging must produce artifacts that survive email and still work.

Both failure modes here are silent: a standalone HTML that still references a
sibling file looks fine on the machine that built it and is blank everywhere
else, and a base64 payload that does not round-trip only reveals itself in the
recipient's hands.
"""

from __future__ import annotations

import base64
import hashlib
import re
import zipfile
from pathlib import Path

import pytest

from pipeline.package import (
    build_standalone_html,
    build_zip,
    encode_to_text,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard() -> Path:
    path = ROOT / "dashboard"
    if not (path / "data.js").exists():
        pytest.skip("dashboard/data.js not generated; run python -m pipeline.run first")
    return path


class TestStandaloneHtml:
    def test_has_no_external_references(self, dashboard: Path, tmp_path: Path) -> None:
        out = build_standalone_html(dashboard, tmp_path / "standalone.html")
        html = out.read_text(encoding="utf-8")
        assert not re.search(r"<script[^>]+\bsrc\s*=", html), "still loads an external script"
        assert not re.search(r"<link[^>]+\bhref\s*=", html), "still loads an external stylesheet"

    def test_contains_the_whole_application(self, dashboard: Path, tmp_path: Path) -> None:
        out = build_standalone_html(dashboard, tmp_path / "standalone.html")
        html = out.read_text(encoding="utf-8")
        assert "window.CR_DATA" in html, "data payload missing"
        assert "CRModel" in html, "model code missing"
        assert "Database Explorer" in html, "application code missing"

    def test_escapes_nested_script_terminators(self, dashboard: Path, tmp_path: Path) -> None:
        """A literal </script> inside inlined JS would truncate the page."""
        out = build_standalone_html(dashboard, tmp_path / "standalone.html")
        html = out.read_text(encoding="utf-8")
        assert html.count("<script>") == html.count("</script>"), (
            "unbalanced script tags: an inlined payload closed the block early"
        )


class TestBase64Bundle:
    def test_round_trips_byte_for_byte(self, tmp_path: Path) -> None:
        zip_path = build_zip(ROOT, tmp_path / "CR.zip")
        original = zip_path.read_bytes()
        text_path, digest = encode_to_text(zip_path, tmp_path / "bundle.txt")

        # Decode exactly as the instructions tell the recipient to.
        decoded = base64.b64decode(text_path.read_text(encoding="utf-8"))
        assert decoded == original
        assert hashlib.sha256(decoded).hexdigest() == digest

    def test_payload_is_plain_text_only(self, tmp_path: Path) -> None:
        """Anything but base64 characters risks a mail gateway rewriting it."""
        zip_path = build_zip(ROOT, tmp_path / "CR.zip")
        text_path, _ = encode_to_text(zip_path, tmp_path / "bundle.txt")
        body = text_path.read_text(encoding="utf-8")
        assert re.fullmatch(r"[A-Za-z0-9+/=\n]+", body), "payload contains non-base64 characters"
        assert max(len(line) for line in body.splitlines()) <= 76, (
            "long lines may be hard-wrapped in transit"
        )

    def test_archive_excludes_regenerable_and_local_files(self, tmp_path: Path) -> None:
        zip_path = build_zip(ROOT, tmp_path / "CR.zip")
        with zipfile.ZipFile(zip_path) as archive:
            names = archive.namelist()
            assert archive.testzip() is None
        for unwanted in ("node_modules", "__pycache__", ".pytest_cache", "dist/"):
            assert not any(unwanted in n for n in names), f"{unwanted} was packaged"
        for required in (
            "CR/pipeline/run.py",
            "CR/dashboard/index.html",
            "CR/dashboard/data.js",
            "CR/README.md",
            "CR/pyproject.toml",
        ):
            assert required in names, f"{required} missing from the archive"

    def test_archive_is_reproducible(self, tmp_path: Path) -> None:
        """G10 extends to the deliverable: two builds must be byte-identical."""
        first = build_zip(ROOT, tmp_path / "a.zip").read_bytes()
        second = build_zip(ROOT, tmp_path / "b.zip").read_bytes()
        assert hashlib.sha256(first).hexdigest() == hashlib.sha256(second).hexdigest()


class TestDashboardTextFallback:
    """The .html itself may be blocked, so it also ships as plain text."""

    def test_encoded_dashboard_round_trips(self, dashboard: Path, tmp_path: Path) -> None:
        html = build_standalone_html(dashboard, tmp_path / "standalone.html")
        text_path, digest = encode_to_text(html, tmp_path / "dashboard.txt")

        decoded = base64.b64decode(text_path.read_text(encoding="utf-8"))
        assert decoded == html.read_bytes()
        assert hashlib.sha256(decoded).hexdigest() == digest
        # It must decode back to something a browser will open.
        assert decoded.lstrip().startswith(b"<!DOCTYPE html>")
