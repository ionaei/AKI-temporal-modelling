"""Build analysis 'stays' from raw care episodes.

Contiguous or overlapping care episodes (ICU bed moves, split records) are
merged into one stay, so that the 24-hour 'AKI on arrival' exclusion and the
creatinine reference logic do not restart at every transfer. Stays are then
filtered by age at admission and minimum length of stay.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_stays(episodes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gap = pd.Timedelta(hours=cfg.get("merge_gap_hours", 24))
    ep = episodes.sort_values(["subject_id", "start"]).reset_index(drop=True)

    # running max of end within subject; a new stay starts when the next
    # episode begins more than `gap` after everything before it has ended
    run_end = ep.groupby("subject_id")["end"].cummax()
    prev_end = run_end.groupby(ep["subject_id"]).shift()
    new_stay = prev_end.isna() | (ep["start"] > prev_end + gap)
    ep["stay_n"] = new_stay.groupby(ep["subject_id"]).cumsum()

    stays = (ep.groupby(["subject_id", "stay_n"])
               .agg(admit=("start", "min"), discharge=("end", "max"),
                    dob=("dob", "first"), sex=("sex", "first"),
                    units=("unit", lambda s: "|".join(sorted(set(map(str, s))))),
                    n_episodes=("start", "size"))
               .reset_index())
    stays["stay_id"] = stays["subject_id"].astype(str) + "_" + stays["stay_n"].astype(str)
    stays["age_days"] = (stays["admit"] - stays["dob"]).dt.total_seconds() / 86400
    stays["los_h"] = (stays["discharge"] - stays["admit"]).dt.total_seconds() / 3600

    n0 = len(stays)
    lo, hi = cfg.get("min_age_days", 0), cfg.get("max_age_days", 18 * 365.25)
    keep = stays["age_days"].between(lo, hi, inclusive="left")
    keep &= stays["los_h"] >= cfg.get("min_los_hours", 6)
    stays = stays[keep].drop(columns="stay_n").reset_index(drop=True)
    log.info("episodes %d -> stays %d (merged) -> %d after age/LOS filters",
             len(ep), n0, len(stays))
    return stays[["stay_id", "subject_id", "admit", "discharge", "dob", "sex", "age_days",
                  "los_h", "n_episodes", "units"]]
