"""Locate small project assets in source checkouts and installed wheels."""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def asset_root() -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("PANDOSI_ASSETS")
    if configured:
        candidates.append(Path(configured))
    candidates.append(Path.cwd() / "assets")
    candidates.append(Path(__file__).resolve().parents[1] / "assets")
    data_path = sysconfig.get_path("data")
    if data_path:
        candidates.append(Path(data_path) / "share" / "pandosi" / "assets")
    candidates.append(Path(sys.prefix) / "share" / "pandosi" / "assets")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def asset_path(*parts: str) -> Path:
    return asset_root().joinpath(*parts)
