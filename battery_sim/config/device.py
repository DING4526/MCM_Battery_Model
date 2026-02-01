import random
from typing import Dict, Any

# 标称设备参数（用于蒙特卡洛采样的中心值）
DEVICE: Dict[str, float] = {
    # 屏幕
    "A": 48.0,  # cm^2
    "p_u": 0.001,  # W/cm^2 每单位亮度
    "P_base_60": 0.10,  # W
    "P_off": 0.01,  # W
    "kappa": 0.2,
    # CPU
    "P_cpu_idle": 0.1,  # W
    "P_cpu_dyn": 2.0,  # W（u=1）
    # 无线
    "P_i": 0.1,  # 空闲
    "P_a": 0.9,  # 接收活跃
    "P_t": 0.6,  # 发送
    "alpha_wifi": 0.85,
    "alpha_cell": 1.1,
    # GPS
    "P_gps": 0.15,
    # 后台
    "P_bg_idle": 0.08,
    "P_bg_active": 0.30,
    # 电池与热
    "E0": 15.0,  # Wh
    "lambda_base": 4.19e-4,
    "k_FC": 8.77e-4,
    "k_T": 0.0417,
    "T_ref": 298.0,
    "C_th": 60.0,  # J/K
    "h": 1.0,  # W/K
    "eta_heat": 0.8,
}


def _jitter(value: float, rel: float = 0.05) -> float:
    """对参数加入相对扰动，用于蒙特卡洛采样。"""
    return random.gauss(value, rel * value)


def sample_device(seed: int | None = None) -> Dict[str, Any]:
    """进行一次设备参数的蒙特卡洛采样。"""
    if seed is not None:
        random.seed(seed)
    sampled = {k: _jitter(v, 0.05) for k, v in DEVICE.items()}
    # 保证功率非负
    for key in ("P_base_60", "P_off", "P_cpu_idle", "P_cpu_dyn", "P_i", "P_a", "P_t", "P_gps", "P_bg_idle", "P_bg_active"):
        sampled[key] = max(sampled[key], 0.0)
    return sampled
