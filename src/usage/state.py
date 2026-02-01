# usage/state.py
import copy

DEVICE_PARAMS = {
    "A": 45.0,
    "p_u": 0.0008,
    "P_base60": 0.10,
    "kappa": 0.25,
    "P_off": 0.02,

    "P_idle": 0.08,
    "P_dyn": 2.5,

    "P_i": 0.08,
    "P_a": 0.95,
    "P_t": 0.65,
    "alpha_cell": 1.15,
    "alpha_wifi": 0.85,

    "P_on": 0.15,

    "P_bg_idle": 0.07,
    "P_bg_act": 0.30,
}

USAGE_STATES = {
    "DeepIdle": {
        "s": 0, "u": 0.0, "r": 60, "u_cpu": 0.05,
        "R_i": 0.9, "R_a": 0.05, "R_t": 0.05,
        "lambda_cell": 0.3, "delta_signal": 0.0,
        "r_on": 0.0, "r_bg": 0.1,
    },
    "Social": {
        "s": 1, "u": 0.4, "r": 60, "u_cpu": 0.25,
        "R_i": 0.4, "R_a": 0.4, "R_t": 0.2,
        "lambda_cell": 0.4, "delta_signal": 0.1,
        "r_on": 0.0, "r_bg": 0.2,
    },
    "Video": {
        "s": 1, "u": 0.8, "r": 120, "u_cpu": 0.4,
        "R_i": 0.2, "R_a": 0.6, "R_t": 0.2,
        "lambda_cell": 0.6, "delta_signal": 0.1,
        "r_on": 0.0, "r_bg": 0.15,
    },
    "Gaming": {
        "s": 1, "u": 0.9, "r": 120, "u_cpu": 0.85,
        "R_i": 0.5, "R_a": 0.3, "R_t": 0.2,
        "lambda_cell": 0.5, "delta_signal": 0.2,
        "r_on": 0.0, "r_bg": 0.1,
    },
    "Navigation": {
        "s": 1, "u": 0.7, "r": 60, "u_cpu": 0.35,
        "R_i": 0.1, "R_a": 0.6, "R_t": 0.3,
        "lambda_cell": 0.8, "delta_signal": 0.3,
        "r_on": 0.8, "r_bg": 0.15,
    },
}


def get_state_params(state_name):
    params = copy.deepcopy(DEVICE_PARAMS)
    params.update(USAGE_STATES[state_name])
    return params
