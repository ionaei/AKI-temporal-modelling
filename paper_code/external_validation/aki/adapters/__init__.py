from .gosh import GOSHAdapter
from .mimic_like import MimicLikeAdapter

ADAPTERS = {"gosh": GOSHAdapter, "mimic_like": MimicLikeAdapter}


def get_adapter(site_cfg: dict):
    kind = site_cfg.get("adapter")
    if kind not in ADAPTERS:
        raise ValueError(f"site config needs adapter: one of {list(ADAPTERS)}")
    return ADAPTERS[kind](site_cfg)
