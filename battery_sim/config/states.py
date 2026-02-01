from typing import Dict, Any

# 完整状态表
STATE: Dict[str, Dict[str, Any]] = {
    "S0": {
        "screen": {"s": 0},
        "cpu": {"u": 0.03},
        "radio": {"Ri": 0.9, "Ra": 0.05, "Rt": 0.05, "lambda": 0.1, "delta": 0},
        "gps": {"r": 0},
        "bg": {"r": 0.05},
    },
    "S1": {
        "screen": {"s": 0},
        "cpu": {"u": 0.1},
        "radio": {"Ri": 0.7, "Ra": 0.15, "Rt": 0.15, "lambda": 0.2, "delta": 0.1},
        "gps": {"r": 0},
        "bg": {"r": 0.15},
    },
    "S2": {
        "screen": {"s": 1, "u": 0.3, "r": 60},
        "cpu": {"u": 0.25},
        "radio": {"Ri": 0.3, "Ra": 0.4, "Rt": 0.3, "lambda": 0.4, "delta": 0.1},
        "gps": {"r": 0.05},
        "bg": {"r": 0.2},
    },
    "S3": {
        "screen": {"s": 1, "u": 0.45, "r": 60},
        "cpu": {"u": 0.35},
        "radio": {"Ri": 0.3, "Ra": 0.4, "Rt": 0.3, "lambda": 0.4, "delta": 0.1},
        "gps": {"r": 0},
        "bg": {"r": 0.15},
    },
    "S4": {
        "screen": {"s": 1, "u": 0.7, "r": 90},
        "cpu": {"u": 0.5},
        "radio": {"Ri": 0.2, "Ra": 0.6, "Rt": 0.2, "lambda": 0.6, "delta": 0.1},
        "gps": {"r": 0},
        "bg": {"r": 0.1},
    },
    "S5": {
        "screen": {"s": 1, "u": 0.6, "r": 60},
        "cpu": {"u": 0.4},
        "radio": {"Ri": 0.1, "Ra": 0.6, "Rt": 0.3, "lambda": 0.7, "delta": 0.3},
        "gps": {"r": 0.8},
        "bg": {"r": 0.1},
    },
    "S6": {
        "screen": {"s": 1, "u": 0.8, "r": 120},
        "cpu": {"u": 0.9},
        "radio": {"Ri": 0.4, "Ra": 0.4, "Rt": 0.2, "lambda": 0.5, "delta": 0},
        "gps": {"r": 0},
        "bg": {"r": 0.05},
    },
    "S7": {
        "screen": {"s": 1, "u": 0.7, "r": 60},
        "cpu": {"u": 0.6},
        "radio": {"Ri": 0.3, "Ra": 0.4, "Rt": 0.3, "lambda": 0.3, "delta": 0.1},
        "gps": {"r": 0},
        "bg": {"r": 0.05},
    },
}


def get_state_params(state: str, system_mode: str | None = None) -> Dict[str, Any]:
    """返回指定状态的参数，并根据系统模式做简单修正。"""
    base = STATE[state].copy()
    if system_mode == "battery_saver":
        if "screen" in base and base["screen"].get("s", 0) == 1:
            base["screen"] = {**base["screen"], "u": base["screen"].get("u", 0) * 0.7}
        if "cpu" in base:
            base["cpu"] = {**base["cpu"], "u": base["cpu"].get("u", 0) * 0.8}
        if "radio" in base:
            radio = base["radio"]
            base["radio"] = {
                **radio,
                "Ra": radio.get("Ra", 0) * 0.8,
                "Rt": radio.get("Rt", 0) * 0.8,
            }
    return base
