"""Load YAML experiment config and resolve paths relative to the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else REPO_ROOT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {cfg_path} is empty or not a mapping")
    cfg["_config_path"] = str(cfg_path)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolve_path(cfg: dict[str, Any], key_path: str) -> Path:
    node: Any = cfg
    for part in key_path.split("."):
        node = node[part]
    path = Path(node)
    if not path.is_absolute():
        path = Path(cfg["_repo_root"]) / path
    return path
