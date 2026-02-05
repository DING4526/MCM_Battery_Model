import math
from typing import Dict


def effective_energy(G: float, T_b: float, FC: float, cycles: int, device: Dict[str, float]) -> float:
    """
    有效可用能量（Wh），考虑温度和老化。
    G: 温度低效因子（可选外部输入）
    T_b: 电池温度 (K)
    FC: 容量衰减比例 delta
    cycles: 历史循环数（可用于扩展）
    """
    E0 = device["E0"]
    # 温度修正，低温指数下降
    f_T = math.exp(-device["k_T"] * max(device["T_ref"] - T_b, 0))
    # 老化修正
    f_A = 1 - FC
    # 简单补偿外部温度效率 G
    return E0 * f_T * f_A * G

