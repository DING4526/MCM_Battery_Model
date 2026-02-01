# power_model.py
# 根据“使用状态给定的一组参数”，计算瞬时功耗 P(t)


# =========================
# 屏幕功耗模型
# =========================
def screen_power(p):
    """
    AMOLED 屏幕功耗模型
    参数 p 中应包含：
        s       : 是否亮屏（0/1）
        u       : 平均归一化亮度（0~1）
        r       : 刷新率（Hz）
        A       : 屏幕发光面积（cm^2）
        p_u     : 单位面积单位亮度功耗密度（W/cm^2）
        P_base60: 60Hz 亮屏基线功耗（W）
        kappa   : 刷新率敏感系数
        P_off   : 息屏基线功耗（W）
    """
    if p["s"] == 0:
        return p["P_off"]

    P_base = p["P_base60"] * (1 + p["kappa"] * (p["r"] / 60.0 - 1))
    P_emit = p["A"] * p["p_u"] * p["u"]

    return P_base + P_emit


# =========================
# CPU 功耗模型
# =========================
def cpu_power(p):
    """
    CPU 功耗模型
    参数：
        u_cpu  : CPU 利用率（0~1）
        P_idle : 空载功耗（W）
        P_dyn  : 满负载动态功耗增量（W）
    """
    return p["P_idle"] + p["P_dyn"] * p["u_cpu"]


# =========================
# 无线通信功耗模型
# =========================
def radio_power(p):
    """
    无线通信功耗模型（三态 + 制式 + 信号修正）
    参数：
        R_i, R_a, R_t : Idle / Active / Tail 时间比例
        P_i, P_a, P_t : 各状态功耗
        lambda_cell  : 蜂窝网络使用比例
        alpha_cell   : 蜂窝功耗系数
        alpha_wifi   : WiFi 功耗系数
        delta_signal : 信号质量修正
    """
    P_avg = (
        p["R_i"] * p["P_i"]
        + p["R_a"] * p["P_a"]
        + p["R_t"] * p["P_t"]
    )

    alpha = (
        p["lambda_cell"] * p["alpha_cell"]
        + (1 - p["lambda_cell"]) * p["alpha_wifi"]
    )

    return P_avg * alpha * (1 + p["delta_signal"])


# =========================
# GPS 功耗模型
# =========================
def gps_power(p):
    """
    GPS 功耗模型（启用比例模型）
    参数：
        r_on : GPS 启用时间比例（0~1）
        P_on : 持续定位功耗（W）
    """
    return p["r_on"] * p["P_on"]


# =========================
# 后台功耗模型
# =========================
def background_power(p):
    """
    后台功耗模型（Idle / Active 比例）
    参数：
        r_bg     : 后台活跃时间比例
        P_bg_idle: 后台空闲功耗
        P_bg_act : 后台活跃功耗
    """
    return (
        p["r_bg"] * p["P_bg_act"]
        + (1 - p["r_bg"]) * p["P_bg_idle"]
    )


# =========================
# 瞬时总功耗
# =========================
def total_power(state_params):
    """
    瞬时总功耗
    所有子模块功耗的代数和
    """
    return (
        screen_power(state_params)
        + cpu_power(state_params)
        + radio_power(state_params)
        + gps_power(state_params)
        + background_power(state_params)
    )


def power_breakdown(state_params):
    """
    返回各子模块功耗分解
    
    返回：
        dict - 包含各子模块功耗的字典：
            - screen: 屏幕功耗 (W)
            - cpu: CPU 功耗 (W)
            - radio: 无线通信功耗 (W)
            - gps: GPS 功耗 (W)
            - background: 后台功耗 (W)
            - total: 总功耗 (W)
    """
    p_screen = screen_power(state_params)
    p_cpu = cpu_power(state_params)
    p_radio = radio_power(state_params)
    p_gps = gps_power(state_params)
    p_bg = background_power(state_params)
    
    return {
        "screen": p_screen,
        "cpu": p_cpu,
        "radio": p_radio,
        "gps": p_gps,
        "background": p_bg,
        "total": p_screen + p_cpu + p_radio + p_gps + p_bg,
    }
