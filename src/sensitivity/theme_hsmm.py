# src/experiments/sensitivity/theme_hsmm.py
from __future__ import annotations

import json
import copy
from pathlib import Path
import numpy as np

from src.visualization.tornado import plot_tornado, TornadoConfig
from .common import run_from_meta

def mc_mean(meta, *, K=30, **kwargs):
    vals = []
    for i in range(K):
        m = run_from_meta(meta, seed_offset=i, **kwargs)
        vals.append(m["TTL_hours"])
    return float(np.mean(vals))

def tweak_dwell_mean(scenario, factor, states=None):
    sc = copy.deepcopy(scenario)
    dp = sc.get("dwell_params", {}) or {}
    if states is None:
        states = list(dp.keys())
    for s in states:
        if s in dp and "mean_sec" in dp[s]:
            dp[s]["mean_sec"] = float(dp[s]["mean_sec"]) * factor
    sc["dwell_params"] = dp
    return sc

def tweak_sigma_social(scenario, factor):
    sc = copy.deepcopy(scenario)
    dp = sc.get("dwell_params", {}) or {}
    if "SOCIAL" in dp and "sigma" in dp["SOCIAL"]:
        dp["SOCIAL"]["sigma"] = float(dp["SOCIAL"]["sigma"]) * factor
    sc["dwell_params"] = dp
    return sc

def main(timeseries_json: str, out_dir: str = "output/sensitivity/theme_hsmm", K: int = 30):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(timeseries_json).read_text(encoding="utf-8"))
    meta = payload["meta"]
    scenario0 = meta["scenario"]

    base_ttl = mc_mean(meta, K=K)

    effects = {}

    # dwell mean all
    lo_sc = tweak_dwell_mean(scenario0, 0.7)
    hi_sc = tweak_dwell_mean(scenario0, 1.3)
    lo = mc_mean(meta, K=K, scenario_override=lo_sc)
    hi = mc_mean(meta, K=K, scenario_override=hi_sc)
    effects["dwell_mean_all (×0.7 / ×1.3)"] = (lo - base_ttl, hi - base_ttl)

    # dwell mean IDLE
    lo_sc = tweak_dwell_mean(scenario0, 0.7, states=["IDLE"])
    hi_sc = tweak_dwell_mean(scenario0, 1.3, states=["IDLE"])
    lo = mc_mean(meta, K=K, scenario_override=lo_sc)
    hi = mc_mean(meta, K=K, scenario_override=hi_sc)
    effects["dwell_mean_IDLE (×0.7 / ×1.3)"] = (lo - base_ttl, hi - base_ttl)

    # dwell mean VIDEO
    lo_sc = tweak_dwell_mean(scenario0, 0.7, states=["VIDEO"])
    hi_sc = tweak_dwell_mean(scenario0, 1.3, states=["VIDEO"])
    lo = mc_mean(meta, K=K, scenario_override=lo_sc)
    hi = mc_mean(meta, K=K, scenario_override=hi_sc)
    effects["dwell_mean_VIDEO (×0.7 / ×1.3)"] = (lo - base_ttl, hi - base_ttl)

    # sigma SOCIAL (lognormal)
    lo_sc = tweak_sigma_social(scenario0, 0.8)
    hi_sc = tweak_sigma_social(scenario0, 1.2)
    lo = mc_mean(meta, K=K, scenario_override=lo_sc)
    hi = mc_mean(meta, K=K, scenario_override=hi_sc)
    effects["sigma_SOCIAL (×0.8 / ×1.2)"] = (lo - base_ttl, hi - base_ttl)

    plot_tornado(
        effects,
        cfg=TornadoConfig(
            title=f"Theme C: HSMM - Tornado (ΔTTL hours, MC mean K={K})",
            xlabel="ΔTTL (hours)",
            figsize=(10.8, 6.6),
        ),
        save_path=str(out / "tornado_hsmm_dTTL.pdf"),
        show=False,
    )

if __name__ == "__main__":
    main("output/population/timeseries/some_device.json", K=30)
