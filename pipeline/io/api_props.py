"""Per-API properties, from a small shared ``apis.csv``.

The dissolution workbooks say which API they hold but nothing about it. What
cross-API analysis needs is supplied once, in one file:

    api,solubility_mg_ml,solubility_class
    API_1,33,high
    API_2,0.04,low

``api`` must match the workbook's API column. ``solubility_mg_ml`` is the
measured solubility. ``solubility_class`` (high / low) is optional. It is not
derived from the number, because the class also depends on dose and on the
medium. A blank class keeps the G1 gate closed for that API.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.io.schema import SchemaError

CLASSES = ("high", "low")


def load_api_props(path: str | Path) -> dict[str, dict[str, Any]]:
    """``{api: {"solubility_mg_ml": float | None, "solubility_class": str | None}}``."""
    source = Path(path)
    if not source.exists():
        raise SchemaError(f"API properties file not found: {source}", 1)
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    missing = [c for c in ("api", "solubility_mg_ml") if c not in frame.columns]
    if missing:
        raise SchemaError(
            f"{source.name} needs columns api and solubility_mg_ml; missing {missing}.", 34
        )
    out: dict[str, dict[str, Any]] = {}
    for i, row in frame.iterrows():
        api = str(row["api"]).strip()
        if not api:
            continue
        if api in out:
            raise SchemaError(f"{source.name}: API {api!r} is listed twice.", 34)
        raw = str(row["solubility_mg_ml"]).strip()
        solubility: float | None = None
        if raw:
            try:
                solubility = float(raw)
            except ValueError:
                raise SchemaError(
                    f"{source.name} row {int(str(i)) + 2}: solubility {raw!r} is not a number.",
                    34,
                ) from None
            if not math.isfinite(solubility) or solubility <= 0:
                raise SchemaError(
                    f"{source.name}: solubility for {api!r} must be positive.", 34
                )
        cls = str(row.get("solubility_class", "")).strip().lower() or None
        if cls is not None and cls not in CLASSES:
            raise SchemaError(
                f"{source.name}: solubility_class for {api!r} must be high, low or blank.", 34
            )
        out[api] = {"solubility_mg_ml": solubility, "solubility_class": cls}
    return out
