# usage/state.py
import copy

DEVICE_PARAMS = {
    # Screen
    "A": 45.0,
    "p_u": 0.01,
    "P_base60": 0.30,
    "kappa": 0.25,
    "P_off": 0.02,

    # CPU
    "P_idle": 0.08,
    "P_dyn": 2.5,

    # Radio
    "P_i": 0.08,
    "P_a": 0.95,
    "P_t": 0.65,
    "alpha_cell": 1.15,
    "alpha_wifi": 0.85,

    # Background
    "P_bg_idle": 0.07,
    "P_bg_act": 0.30,
}

# 4 states: IDLE / SOCIAL / VIDEO / GAME
USAGE_STATES = {
    "IDLE": {
        "s": 0, "u": 0.0, "r": 60, "u_cpu": 0.05,
        "R_i": 0.90, "R_a": 0.05, "R_t": 0.05,
        "lambda_cell": 0.40, "delta_signal": 0.10,
        "r_bg": 0.08,
    },
    "SOCIAL": {
        "s": 1, "u": 0.45, "r": 60, "u_cpu": 0.25,
        "R_i": 0.40, "R_a": 0.40, "R_t": 0.20,
        "lambda_cell": 0.55, "delta_signal": 0.15,
        "r_bg": 0.18,
    },
    "VIDEO": {
        "s": 1, "u": 0.80, "r": 120, "u_cpu": 0.40,
        "R_i": 0.20, "R_a": 0.60, "R_t": 0.20,
        "lambda_cell": 0.60, "delta_signal": 0.15,
        "r_bg": 0.14,
    },
    "GAME": {
        "s": 1, "u": 0.90, "r": 120, "u_cpu": 0.85,
        "R_i": 0.50, "R_a": 0.30, "R_t": 0.20,
        "lambda_cell": 0.55, "delta_signal": 0.20,
        "r_bg": 0.12,
    },
}


def set_usage_states(new_states: dict):
    """
    用设备级 data-driven 的 states 覆盖全局 USAGE_STATES。
    这是最小侵入式接入方式：simulate.py 不用改 get_state_params 的签名。
    """
    USAGE_STATES.clear()
    USAGE_STATES.update(copy.deepcopy(new_states))


def get_state_params(state_name: str, device_params_override: dict | None = None):
    params = dict(DEVICE_PARAMS)
    if device_params_override:
        params.update(device_params_override)
    params.update(USAGE_STATES[state_name])
    return params
