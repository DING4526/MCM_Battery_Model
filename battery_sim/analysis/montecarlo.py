import random
from typing import List, Dict, Any

from battery_sim.config.device import sample_device
from battery_sim.config.scenarios import get_daily_profile
from battery_sim.core.integrator import simulate_day
from battery_sim.config.states import get_state_params


def run_mc(N: int, scenario: str, user: str = "Heavy", system: str = "default") -> List[Dict[str, Any]]:
    """
    进行蒙特卡洛模拟，返回每次模拟的 TTE 与能量分解。
    user、system 可扩展为策略，这里仅作占位。
    """
    results = []
    profile = get_daily_profile(scenario)
    for i in range(N):
        device = sample_device()
        TTE, e_state, e_module = simulate_day(
            profile=profile,
            G=1.0,
            T_amb=298.0,
            SOH0=0.9,
            device=device,
            get_state_params=get_state_params,
            network="wifi",
        )
        results.append({"TTE_h": TTE, "energy_by_state": e_state, "energy_by_module": e_module, "device": device})
    return results

