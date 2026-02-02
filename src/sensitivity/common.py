# src/experiments/sensitivity/common.py
from __future__ import annotations

import copy
from typing import Dict, Any, Optional

from src.simulate import run_simulation

def run_from_meta(
    meta: Dict[str, Any],
    *,
    T_amb_c: Optional[float] = None,
    usage_states_override: Optional[Dict[str, Dict[str, float]]] = None,
    scenario_override: Optional[Dict[str, Any]] = None,
    battery_aging_loss: Optional[float] = None,
    battery_params_override: Optional[Dict[str, float]] = None,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    seed = int(meta.get("seed", 42)) + int(seed_offset)

    scenario = copy.deepcopy(meta.get("scenario", {}))
    if scenario_override is not None:
        scenario = scenario_override

    usage_states = copy.deepcopy(meta.get("usage_states", {}))
    if usage_states_override is not None:
        usage_states = usage_states_override

    cap_mah = float(meta.get("battery_capacity_mah", 5000.0))
    aging_loss = float(meta.get("aging_loss", meta.get("aging", {}).get("aging_loss", 0.15)))
    if battery_aging_loss is not None:
        aging_loss = float(battery_aging_loss)

    Tamb = float(meta.get("T_amb_c", 25.0)) if T_amb_c is None else float(T_amb_c)
    T_amb_K = Tamb + 273.15

    res = run_simulation(
        scenario=scenario,
        dt=float(meta.get("dt", 1.0)),
        T_amb=T_amb_K,
        seed=seed,
        record=True,
        record_breakdown=True,
        record_uncorrected=True,
        usage_states_override=usage_states,
        battery_capacity_mah=cap_mah,
        battery_aging_loss=aging_loss,
        fixed_dwell_sec=int(meta.get("fixed_dwell_sec", 900)),
        dwell_mode="hsmm",
        battery_params_override=battery_params_override,  # 你加的参数
    )

    # 指标提取
    ttl_h = res["TTL"] / 3600.0
    tb_max_c = (max(res["Tb"]) - 273.15) if res.get("Tb") else float("nan")

    # 能耗占比（需要 breakdown）
    dt_sec = float(meta.get("dt", 1.0))
    def energy(x): return (sum(x) * dt_sec) if x else 0.0
    E_total = energy(res.get("Power"))
    den = E_total if E_total > 1e-12 else 1.0

    metrics = {
        "TTL_hours": ttl_h,
        "max_Tb_C": tb_max_c,
        "energy_ratio_screen": energy(res.get("Power_screen")) / den,
        "energy_ratio_cpu": energy(res.get("Power_cpu")) / den,
        "energy_ratio_radio": energy(res.get("Power_radio")) / den,
        "energy_ratio_background": energy(res.get("Power_background")) / den,
    }
    return metrics
