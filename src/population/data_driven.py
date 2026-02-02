# src/population/data_driven.py
from __future__ import annotations

def _norm_str(x: str) -> str:
    return str(x).strip().lower()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def build_state_ratio_from_row(row: dict) -> dict:
    """
    输入 row（来自 CSV 一行，dict-like）
    输出 4-state ratio，key: IDLE/SOCIAL/VIDEO/GAME
    """
    # 基础
    screen_on_day = float(row["avg_screen_on_hours_per_day"])
    screen_on_week = clamp(screen_on_day * 7.0, 0.0, 168.0)

    gaming_week = max(0.0, float(row["gaming_hours_per_week"]))
    video_week = max(0.0, float(row["video_streaming_hours_per_week"]))

    # 避免异常：游戏+视频超过亮屏总时长
    if gaming_week + video_week > screen_on_week:
        scale = screen_on_week / (gaming_week + video_week + 1e-9)
        gaming_week *= scale
        video_week *= scale

    social_week = max(screen_on_week - gaming_week - video_week, 0.0)
    idle_week = max(168.0 - screen_on_week, 0.0)

    ratio = {
        "IDLE": idle_week / 168.0,
        "SOCIAL": social_week / 168.0,
        "VIDEO": video_week / 168.0,
        "GAME": gaming_week / 168.0,
    }

    # 归一化（防止浮点误差或异常数据）
    s = sum(ratio.values())
    if s <= 0:
        return {"IDLE": 1.0, "SOCIAL": 0.0, "VIDEO": 0.0, "GAME": 0.0}
    for k in ratio:
        ratio[k] /= s
    return ratio

def build_scenario_from_row(row: dict) -> dict:
    return {
        "type": "mixed",
        "state_ratio": build_state_ratio_from_row(row),
    }

def build_usage_states_from_row(row: dict) -> dict:
    """
    生成 device-specific 的 USAGE_STATES（4态）
    数据驱动项：
      - background_app_usage_level -> r_bg
      - signal_strength_avg -> delta_signal, lambda_cell
    其它项用经验参数（与状态语义一致）
    """
    # --- background ---
    bg_map = {"low": 0.10, "medium": 0.18, "high": 0.28}
    bg_level = _norm_str(row["background_app_usage_level"])
    bg_base = bg_map.get(bg_level, 0.18)

    bg_scale = {
        "IDLE": 0.60,
        "SOCIAL": 1.00,
        "VIDEO": 0.80,
        "GAME": 0.70,
    }

    # --- signal ---
    sig_map = {"good": 0.05, "moderate": 0.15, "poor": 0.30}
    lam_map = {"good": 0.35, "moderate": 0.55, "poor": 0.75}
    sig = _norm_str(row["signal_strength_avg"])
    delta_signal = sig_map.get(sig, 0.15)
    lambda_cell = lam_map.get(sig, 0.55)

    # 你也可以把 usage_intensity_score 用来微调 u/u_cpu/r（这里先不做，按你要求先固定经验值）
    # intensity = clamp(float(row["usage_intensity_score"]) / 10.0, 0.0, 1.0)

    def rbg(state):
        return clamp(bg_base * bg_scale[state], 0.0, 0.6)

    # 经验参数：符合语义（你后续可再用 intensity 微调）
    return {
        "IDLE": {
            "s": 0, "u": 0.0, "r": 60, "u_cpu": 0.05,
            "R_i": 0.90, "R_a": 0.05, "R_t": 0.05,
            "lambda_cell": lambda_cell, "delta_signal": delta_signal,
            "r_bg": rbg("IDLE"),
        },
        "SOCIAL": {
            "s": 1, "u": 0.45, "r": 60, "u_cpu": 0.25,
            "R_i": 0.40, "R_a": 0.40, "R_t": 0.20,
            "lambda_cell": lambda_cell, "delta_signal": delta_signal,
            "r_bg": rbg("SOCIAL"),
        },
        "VIDEO": {
            "s": 1, "u": 0.80, "r": 120, "u_cpu": 0.40,
            "R_i": 0.20, "R_a": 0.60, "R_t": 0.20,
            "lambda_cell": lambda_cell, "delta_signal": delta_signal,
            "r_bg": rbg("VIDEO"),
        },
        "GAME": {
            "s": 1, "u": 0.90, "r": 120, "u_cpu": 0.85,
            "R_i": 0.50, "R_a": 0.30, "R_t": 0.20,
            "lambda_cell": lambda_cell, "delta_signal": delta_signal,
            "r_bg": rbg("GAME"),
        },
    }
