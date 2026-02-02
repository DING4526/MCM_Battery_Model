# src/population/data_driven.py
from __future__ import annotations

def _norm_str(x: str) -> str:
    return str(x).strip().lower()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def build_state_ratio_from_row(row: dict, awake_hours: float = 14.0, min_idle_awake: float = 0.02) -> dict:
    """
    从 24h 行为特征构造“续航仿真用”的 awake-window ratio（更贴近连续使用）
    """
    # ---- 原来的 24h ratio ----
    screen_on_day = float(row["avg_screen_on_hours_per_day"])
    screen_on_week = clamp(screen_on_day * 7.0, 0.0, 168.0)

    gaming_week = max(0.0, float(row["gaming_hours_per_week"]))
    video_week = max(0.0, float(row["video_streaming_hours_per_week"]))

    if gaming_week + video_week > screen_on_week:
        scale = screen_on_week / (gaming_week + video_week + 1e-9)
        gaming_week *= scale
        video_week *= scale

    social_week = max(screen_on_week - gaming_week - video_week, 0.0)
    idle_week = max(168.0 - screen_on_week, 0.0)

    ratio_24 = {
        "IDLE": idle_week / 168.0,
        "SOCIAL": social_week / 168.0,
        "VIDEO": video_week / 168.0,
        "GAME": gaming_week / 168.0,
    }

    # ---- 24h -> awake-window 映射 ----
    awake_hours = clamp(float(awake_hours), 4.0, 20.0)
    awake_frac = awake_hours / 24.0
    sleep_frac = 1.0 - awake_frac

    idle_24 = ratio_24["IDLE"]

    # “长 idle（睡觉/充电）”从 IDLE 里扣掉；剩下当作“短 idle（活跃窗口里）”
    idle_awake = max(idle_24 - sleep_frac, 0.0) / max(awake_frac, 1e-9)

    # 防止极端用户（screen_on 很高）导致 idle_awake≈0，HSMM 太“无喘息”
    idle_awake = max(idle_awake, float(min_idle_awake))

    # 其它三项按 awake_frac 放大到活跃窗口尺度
    social_awake = ratio_24["SOCIAL"] / max(awake_frac, 1e-9)
    video_awake = ratio_24["VIDEO"] / max(awake_frac, 1e-9)
    game_awake = ratio_24["GAME"] / max(awake_frac, 1e-9)

    ratio = {
        "IDLE": idle_awake,
        "SOCIAL": social_awake,
        "VIDEO": video_awake,
        "GAME": game_awake,
    }

    # 归一化
    s = sum(ratio.values())
    if s <= 0:
        return {"IDLE": 1.0, "SOCIAL": 0.0, "VIDEO": 0.0, "GAME": 0.0}
    for k in ratio:
        ratio[k] /= s
    return ratio


def build_scenario_from_row(row: dict) -> dict:
    return {
        "type": "hsmm",  # 如果你已经升级成 hsmm
        "state_ratio": build_state_ratio_from_row(row, awake_hours=14.0, min_idle_awake=0.02),
        "initial_state": "SOCIAL",
        # 可选：短 idle + 合理 session 驻留
        "dwell_params": {
            "IDLE":   {"dist": "gamma", "mean_sec": 120,  "shape_k": 2.0},   # 短 idle（2min 均值）
            "SOCIAL": {"dist": "lognormal", "mean_sec": 300, "sigma": 0.55},
            "VIDEO":  {"dist": "gamma", "mean_sec": 900,  "shape_k": 2.2},
            "GAME":   {"dist": "gamma", "mean_sec": 600,  "shape_k": 2.0},
        },
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
