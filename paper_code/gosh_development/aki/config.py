"""Configuration loading.

Every run is driven by two YAML files:
    configs/common.yaml        – study-wide settings (windows, labelling, features, training)
    configs/sites/<site>.yaml  – where the raw data live and how to map them

A site file may declare `extends: other_site` to inherit and override
(e.g. gosh_silent.yaml extends gosh.yaml with different raw paths).
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_site(site: str) -> dict:
    path = CONFIG_DIR / "sites" / f"{site}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No site config at {path}")
    cfg = _read(path)
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_site(parent), cfg)
    cfg["site"] = site
    return cfg


def load_config(site: str, common_path: str | Path | None = None) -> dict:
    """Return {'common': {...}, 'site': {...}} with resolved data directories."""
    common = _read(Path(common_path) if common_path else CONFIG_DIR / "common.yaml")
    site_cfg = load_site(site)
    # site may override any common setting under `common_overrides`
    common = deep_merge(common, site_cfg.pop("common_overrides", {}))
    out_root = Path(site_cfg.get("output_dir", ROOT / "data" / site))
    site_cfg["dirs"] = {
        "root": out_root,
        "canonical": out_root / "canonical",
        "labelled": out_root / "labelled",
        "windows": out_root / "windows",
    }
    for d in site_cfg["dirs"].values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return {"common": common, "site": site_cfg}
