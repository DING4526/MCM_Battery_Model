# src/experiments/sensitivity/theme_usage_mapping.py
from __future__ import annotations

import json
import copy
from pathlib import Path

from src.visualization.tornado import plot_tornado, TornadoConfig
from .common import run_from_meta

ON_STATES = ("SOCIAL", "VIDEO", "GAME")

def mul_param(usage_states, key, factor, states=None, clamp=None):
    states = states or usage_states.keys()
    out = copy.deepcopy(usage_states)
    for s in states:
        if s not in out: 
            continue
        if key in out[s]:
            out[s][key] = out[s][key] * factor
            if clamp:
                lo, hi = clamp
                out[s][key] = max(lo, min(hi, out[s][key]))
    return out

def main(timeseries_json: str, out_dir: str = "output/sensitivity/theme_usage_mapping"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(timeseries_json).read_text(encoding="utf-8"))
    meta = payload["meta"]

    base = run_from_meta(meta)

    effects_ttl = {}
    effects_radio = {}

    usage0 = meta["usage_states"]

    def add_effect(name, low_states, high_states):
        lo = run_from_meta(meta, usage_states_override=low_states)
        hi = run_from_meta(meta, usage_states_override=high_states)
        effects_ttl[name] = (lo["TTL_hours"] - base["TTL_hours"], hi["TTL_hours"] - base["TTL_hours"])
        effects_radio[name] = (lo["energy_ratio_radio"] - base["energy_ratio_radio"], hi["energy_ratio_radio"] - base["energy_ratio_radio"])

    # brightness u (亮屏态)
    add_effect(
        "u (on states ×0.9 / ×1.1)",
        mul_param(usage0, "u", 0.9, states=ON_STATES, clamp=(0.0, 1.0)),
        mul_param(usage0, "u", 1.1, states=ON_STATES, clamp=(0.0, 1.0)),
    )

    # u_cpu (亮屏态)
    add_effect(
        "u_cpu (on states ×0.9 / ×1.1)",
        mul_param(usage0, "u_cpu", 0.9, states=ON_STATES, clamp=(0.0, 1.0)),
        mul_param(usage0, "u_cpu", 1.1, states=ON_STATES, clamp=(0.0, 1.0)),
    )

    # r_bg (全态)
    add_effect(
        "r_bg (all ×0.8 / ×1.2)",
        mul_param(usage0, "r_bg", 0.8, clamp=(0.0, 0.6)),
        mul_param(usage0, "r_bg", 1.2, clamp=(0.0, 0.6)),
    )

    # delta_signal (全态)
    add_effect(
        "delta_signal (all ×0.8 / ×1.2)",
        mul_param(usage0, "delta_signal", 0.8, clamp=(0.0, 1.0)),
        mul_param(usage0, "delta_signal", 1.2, clamp=(0.0, 1.0)),
    )

    # lambda_cell (全态)
    add_effect(
        "lambda_cell (all ×0.8 / ×1.2)",
        mul_param(usage0, "lambda_cell", 0.8, clamp=(0.0, 1.0)),
        mul_param(usage0, "lambda_cell", 1.2, clamp=(0.0, 1.0)),
    )

    plot_tornado(
        effects_ttl,
        cfg=TornadoConfig(title="Theme B: Usage Mapping - Tornado (ΔTTL hours)", xlabel="ΔTTL (hours)", figsize=(10.8, 6.6)),
        save_path=str(out / "tornado_usage_dTTL.pdf"),
        show=False,
    )
    plot_tornado(
        effects_radio,
        cfg=TornadoConfig(title="Theme B: Usage Mapping - Tornado (ΔRadio energy ratio)", xlabel="Δ energy_ratio_radio", figsize=(10.8, 6.6)),
        save_path=str(out / "tornado_usage_dRadioRatio.pdf"),
        show=False,
    )

if __name__ == "__main__":
    main("output/population/timeseries/some_device.json")
