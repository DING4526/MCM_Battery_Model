# src/aging_model.py
#
# 老化修正外接模型：从 CSV 用户级字段 -> 计算 SOH 与 aging_loss(δ)
#
# 依据你给的模型设计：
#   SOH(n) = exp(-lambda_eff * n)
#   lambda_eff = lambda_base * (1 + k_FC * FC% + k_T * ΔT_avg)
#
# 输出：
#   aging_loss = 1 - SOH  （直接接入 BatteryModel(aging_loss=...)）
#
# 注意：
# - FC% 按“百分数”使用（0~100）。因为文档解释为“每增加 1% 快充，老化速率微增 ~0.09%”
# - ΔT_avg = avg_battery_temp_celsius - T_ref_C（默认 25°C）
# - n 由 device_age_months 与 avg_charging_cycles_per_week 折算得到

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class AgingParams:
    # 文档给的回归结果（默认值）
    lambda_base: float = 4.19e-4   # /cycle
    k_FC: float = 8.77e-4         # per 1% (FC% in [0,100])
    k_T: float = 4.1687e-2        # per 1°C (ΔT in °C)

    # 折算用常数
    weeks_per_month: float = 4.345  # 平均每月周数（≈52.14/12）

    # 参考温度（对应你电池模型的 T_ref=298.15K）
    T_ref_C: float = 25.0

    # 数值保护
    min_cycles: float = 0.0
    max_cycles: float = 1e6
    min_delta: float = 0.0
    max_delta: float = 0.95  # δ 不要到 1，避免容量归零导致数值爆炸


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, str) and x.strip() == "":
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def estimate_aging_from_row(row: dict, params: AgingParams | None = None) -> dict:
    """
    输入：CSV dict row
    输出：{
      "n_cycles": float,
      "lambda_eff": float,
      "SOH": float,
      "aging_loss": float,
      "deltaT_C": float,
      "FC_percent": float
    }
    """
    p = params or AgingParams()

    age_months = _safe_float(row.get("device_age_months"), 0.0)
    cycles_per_week = _safe_float(row.get("avg_charging_cycles_per_week"), 0.0)

    # 折算累积循环数 n
    n = age_months * p.weeks_per_month * cycles_per_week
    n = _clamp(n, p.min_cycles, p.max_cycles)

    # 快充百分比（0~100）
    FC_percent = _safe_float(row.get("fast_charging_usage_percent"), 0.0)
    FC_percent = _clamp(FC_percent, 0.0, 100.0)

    # 平均电池温度（°C） -> ΔT
    T_avg_C = _safe_float(row.get("avg_battery_temp_celsius"), p.T_ref_C)
    deltaT = T_avg_C - p.T_ref_C

    # lambda_eff
    lambda_eff = p.lambda_base * (1.0 + p.k_FC * FC_percent + p.k_T * deltaT)

    # 数值保护：lambda_eff 不应为负
    if lambda_eff < 0:
        lambda_eff = 0.0

    # SOH & aging_loss
    SOH = math.exp(-lambda_eff * n) if n > 0 else 1.0
    SOH = _clamp(SOH, 0.0, 1.0)

    aging_loss = 1.0 - SOH
    aging_loss = _clamp(aging_loss, p.min_delta, p.max_delta)

    return {
        "n_cycles": n,
        "lambda_eff": lambda_eff,
        "SOH": SOH,
        "aging_loss": aging_loss,
        "deltaT_C": deltaT,
        "FC_percent": FC_percent,
        "T_avg_C": T_avg_C,
    }
