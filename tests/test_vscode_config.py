"""The VS Code setup must be valid and must ship to the recipient.

These configs are the recipient's first contact with the project. A malformed
JSON file here produces a silent, confusing failure — VS Code ignores it and
none of the tasks appear, with no error a non-specialist would recognise.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from pipeline.package import build_zip

ROOT = Path(__file__).resolve().parents[1]
VSCODE = ROOT / ".vscode"

CONFIGS = ("settings.json", "launch.json", "tasks.json", "extensions.json")


@pytest.mark.parametrize("name", CONFIGS)
def test_config_is_valid_json(name: str) -> None:
    path = VSCODE / name
    assert path.exists(), f".vscode/{name} is missing"
    json.loads(path.read_text(encoding="utf-8"))


def test_launch_configs_reference_real_modules() -> None:
    launch = json.loads((VSCODE / "launch.json").read_text(encoding="utf-8"))
    modules = {c.get("module") for c in launch["configurations"]}
    assert modules <= {"pipeline.run", "pipeline.package"}, (
        f"unknown module in launch.json: {modules}"
    )
    for module in modules:
        parts = module.split(".")
        assert (ROOT / parts[0] / f"{parts[1]}.py").exists(), f"{module} does not exist"


def test_tasks_cover_the_whole_workflow() -> None:
    tasks = json.loads((VSCODE / "tasks.json").read_text(encoding="utf-8"))
    labels = " ".join(t["label"].lower() for t in tasks["tasks"])
    for step in ("install", "run analysis", "dashboard", "test"):
        assert step in labels, f"no task covers '{step}'"


def test_requirements_match_the_declared_dependencies() -> None:
    """requirements.txt is what the recipient installs; it must not drift."""
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for package in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "openpyxl"):
        assert package in requirements, f"{package} missing from requirements.txt"
        assert package in pyproject, f"{package} missing from pyproject.toml"


def test_recipient_facing_files_are_packaged(tmp_path: Path) -> None:
    zip_path = build_zip(ROOT, tmp_path / "CR.zip")
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
    required = {
        "CR/START_HERE.md",
        "CR/requirements.txt",
        "CR/README.md",
        *(f"CR/.vscode/{name}" for name in CONFIGS),
    }
    missing = required - names
    assert not missing, f"not shipped to the recipient: {sorted(missing)}"
