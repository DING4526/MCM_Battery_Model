from typing import Dict


def screen_power(screen: Dict, device: Dict[str, float]) -> float:
    """屏幕功耗模型，含亮度与刷新率影响。"""
    s = screen.get("s", 0)
    if s == 0:
        return device["P_off"]
    r = screen.get("r", 60)
    u = screen.get("u", 0.5)
    p_base = device["P_base_60"] * (1 + device["kappa"] * (r / 60 - 1))
    p_emit = device["A"] * device["p_u"] * u
    return p_base + p_emit


def cpu_power(cpu: Dict, device: Dict[str, float]) -> float:
    """CPU 功耗，空载 + 动态。"""
    u = cpu.get("u", 0)
    return device["P_cpu_idle"] + device["P_cpu_dyn"] * u


def radio_power(radio: Dict, device: Dict[str, float], network: str = "wifi") -> float:
    """无线功耗，按 idle/active/transmit 占比分解。"""
    Ri = radio.get("Ri", 0)
    Ra = radio.get("Ra", 0)
    Rt = radio.get("Rt", 0)
    alpha = device["alpha_wifi"] if network == "wifi" else device["alpha_cell"]
    return Ri * device["P_i"] + Ra * device["P_a"] * alpha + Rt * device["P_t"] * alpha


def gps_power(gps: Dict, device: Dict[str, float]) -> float:
    """GPS 功耗，按占空比线性放缩。"""
    r = gps.get("r", 0)
    return r * device["P_gps"]


def bg_power(bg: Dict, device: Dict[str, float]) -> float:
    """后台功耗，按活跃比例在 idle/active 之间插值。"""
    r = bg.get("r", 0)
    return (1 - r) * device["P_bg_idle"] + r * device["P_bg_active"]


def total_power(state_params: Dict, device: Dict[str, float], network: str = "wifi") -> Dict[str, float]:
    """返回功耗分解与总功耗。"""
    p_screen = screen_power(state_params.get("screen", {}), device)
    p_cpu = cpu_power(state_params.get("cpu", {}), device)
    p_radio = radio_power(state_params.get("radio", {}), device, network)
    p_gps = gps_power(state_params.get("gps", {}), device)
    p_bg = bg_power(state_params.get("bg", {}), device)
    p_tot = p_screen + p_cpu + p_radio + p_gps + p_bg
    return {
        "screen": p_screen,
        "cpu": p_cpu,
        "radio": p_radio,
        "gps": p_gps,
        "bg": p_bg,
        "total": p_tot,
    }

