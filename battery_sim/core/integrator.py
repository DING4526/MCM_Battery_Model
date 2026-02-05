from typing import Dict, List, Tuple

from battery_sim.core.power import total_power
from battery_sim.core.aging import effective_energy


def simulate_day(
    profile: List[Tuple[str, int]],
    G: float,
    T_amb: float,
    SOH0: float,
    device: Dict[str, float],
    get_state_params,
    network: str = "wifi",
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    profile: [(state, 持续分钟)]
    G: 外部温度效率（由场景给定）
    T_amb: 环境温度 (K)
    SOH0: 初始 SOH (1-δ)
    device: 设备参数
    get_state_params: 状态参数获取函数
    返回 (TTE 小时, 按状态能量, 按子模块能量)
    """
    SOC = 1.0
    T_b = T_amb
    dt = 60  # 秒
    energy_by_state: Dict[str, float] = {}
    energy_by_module = {"screen": 0.0, "cpu": 0.0, "radio": 0.0, "gps": 0.0, "bg": 0.0}

    time_elapsed_s = 0
    for state, minutes in profile:
        params = get_state_params(state)
        steps = minutes * 60 // dt
        for _ in range(int(steps)):
            p = total_power(params, device, network)
            E_eff = effective_energy(G, T_b, 1 - SOH0, cycles=0, device=device)  # Wh
            if E_eff <= 1e-9:
                # 有效能量失效时，直接视为耗尽
                TTE_hours = time_elapsed_s / 3600
                return TTE_hours, energy_by_state, energy_by_module
            dSOC = -(p["total"] * dt) / (E_eff * 3600)  # SOC 变化
            SOC += dSOC
            # 热模型更新
            dT = (device["eta_heat"] * p["total"] - device["h"] * (T_b - T_amb)) * dt / device["C_th"]
            T_b += dT
            time_elapsed_s += dt

            for k in energy_by_module:
                energy_by_module[k] += p[k] * dt / 3600

            energy_by_state[state] = energy_by_state.get(state, 0.0) + p["total"] * dt / 3600

            if SOC <= 0:
                TTE_hours = time_elapsed_s / 3600
                return TTE_hours, energy_by_state, energy_by_module

    TTE_hours = time_elapsed_s / 3600
    return TTE_hours, energy_by_state, energy_by_module
