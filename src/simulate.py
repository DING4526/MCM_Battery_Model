# simulate.py
#
# 核心仿真模块（无可视化）
#
# usage → power → battery
# 支持：
#   - 随机种子控制（可复现）
#   - 时间序列记录
#   - 子模块功耗分解记录
#   - 无修正版本 SOC 对比
#   - Monte Carlo 仿真

import random

from battery_model import BatteryModel
from power_model import total_power, power_breakdown
from usage.state import get_state_params
from usage.control import ScenarioController

from usage.scenario import *

# =====================================================
# 常量定义
# =====================================================
# 固定标称电压（V），用于无修正版本的 SOC 计算
NOMINAL_VOLTAGE = 3.7

# =====================================================
# 单次仿真
# =====================================================
def run_simulation(
    scenario=SCENARIO_STUDENT_DAILY_MIXED,
    dt=1.0,
    T_amb=298.15,
    seed=None,
    record=True,
    record_breakdown=False,
    record_uncorrected=False,
):
    """
    单次运行仿真直到电池耗尽

    参数：
        scenario : dict - 使用场景配置
        dt : float - 时间步长（秒）
        T_amb : float - 环境温度（K）
        seed : int - 随机种子
        record : bool - 是否记录时间序列
        record_breakdown : bool - 是否记录子模块功耗分解
        record_uncorrected : bool - 是否记录各种修正版本的 SOC 对比

    Returns
    -------
    result : dict
        {
          "TTL": float,
          "time": [...],
          "SOC": [...],
          "Tb": [...],
          "Power": [...],
          "State": [...],
          # 如果 record_breakdown=True:
          "Power_screen": [...],
          "Power_cpu": [...],
          "Power_radio": [...],
          "Power_gps": [...],
          "Power_background": [...],
          # 如果 record_uncorrected=True:
          "SOC_uncorrected": [...],     # 完全无修正（固定电压、无温度、无老化）
          "SOC_voltage_only": [...],    # 仅电压修正（OCV-SOC曲线，无温度、无老化）
          "SOC_temperature_only": [...], # 仅温度修正（固定电压，有温度，无老化）
          "SOC_aging_only": [...]       # 仅老化修正（固定电压，无温度，有老化）
        }
    """

    # ---------- 随机种子 ----------
    if seed is not None:
        random.seed(seed)

    # ---------- 初始化电池（带修正） ----------
    battery = BatteryModel(
        SOC0=1.0,
        Tb0=298.15,
        aging_loss=0.15,
    )

    # ---------- 初始化无修正电池（用于对比） ----------
    # 无修正电池使用简化模型：
    # - 无老化损失 (aging_loss=0)
    # - 无温度修正 (alpha=0，使 f_T(Tb)=1 恒定)
    # - 固定标称电压 (NOMINAL_VOLTAGE) 替代 OCV-SOC 曲线
    if record_uncorrected:
        # 完全无修正的电池
        battery_uncorrected = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,  # 无老化修正：f_A = 1
            alpha=0.0,       # 无温度修正：f_T(Tb) = 1（alpha=0 时 exp(0)=1）
        )
        
        # 仅电压修正的电池（使用 OCV-SOC 曲线，但无温度和老化修正）
        battery_voltage_only = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,  # 无老化修正
            alpha=0.0,       # 无温度修正
        )
        
        # 仅温度修正的电池（使用固定电压，但有温度修正，无老化修正）
        battery_temperature_only = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,  # 无老化修正
            alpha=0.03,      # 有温度修正（使用默认 alpha）
        )
        
        # 仅老化修正的电池（使用固定电压，无温度修正，但有老化修正）
        battery_aging_only = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.15, # 有老化修正
            alpha=0.0,       # 无温度修正
        )

    # ---------- 使用行为控制器 ----------
    controller = ScenarioController(scenario)

    # ---------- 记录 ----------
    time_list = []
    soc_list = []
    tb_list = []
    power_list = []
    state_list = []
    
    # 子模块功耗分解记录
    power_screen_list = []
    power_cpu_list = []
    power_radio_list = []
    power_gps_list = []
    power_bg_list = []
    
    # 无修正 SOC 记录
    soc_uncorrected_list = []
    soc_voltage_only_list = []
    soc_temperature_only_list = []
    soc_aging_only_list = []

    t = 0.0

    # ---------- 主循环 ----------
    while battery.SOC > 0:

        current_state = controller.step(dt)
        state_params = get_state_params(current_state)
        P = total_power(state_params)

        battery.step(P, T_amb, dt)
        
        # 更新无修正电池
        if record_uncorrected:
            # 完全无修正：使用固定电压
            # dSOC/dt = -P / (V_nom * Q_nom)
            Q_nom = battery_uncorrected.Q_nom
            dSOC = -P / (NOMINAL_VOLTAGE * Q_nom) * dt
            battery_uncorrected.SOC = max(0.0, battery_uncorrected.SOC + dSOC)
            
            # 仅电压修正：使用 OCV-SOC 曲线（需要调用 voc 方法）
            V_oc = battery_voltage_only.voc(battery_voltage_only.SOC)
            Q_eff_v = battery_voltage_only.Q_nom  # 无温度和老化修正，Q_eff = Q_nom
            dSOC_v = -P / (V_oc * Q_eff_v) * dt
            battery_voltage_only.SOC = max(0.0, battery_voltage_only.SOC + dSOC_v)
            
            # 仅温度修正：使用固定电压，但应用温度修正因子
            # 需要同步更新温度
            battery_temperature_only.Tb = battery.Tb  # 同步主电池的温度
            Q_eff_t = battery_temperature_only.Q_nom * battery_temperature_only.temperature_factor(battery.Tb)
            dSOC_t = -P / (NOMINAL_VOLTAGE * Q_eff_t) * dt
            battery_temperature_only.SOC = max(0.0, battery_temperature_only.SOC + dSOC_t)
            
            # 仅老化修正：使用固定电压，应用老化修正因子
            Q_eff_a = battery_aging_only.Q_nom * battery_aging_only.fA  # f_A = 1 - aging_loss
            dSOC_a = -P / (NOMINAL_VOLTAGE * Q_eff_a) * dt
            battery_aging_only.SOC = max(0.0, battery_aging_only.SOC + dSOC_a)

        if record:
            time_list.append(t)
            soc_list.append(battery.SOC)
            tb_list.append(battery.Tb)
            power_list.append(P)
            state_list.append(current_state)
            
            # 记录子模块功耗
            if record_breakdown:
                breakdown = power_breakdown(state_params)
                power_screen_list.append(breakdown["screen"])
                power_cpu_list.append(breakdown["cpu"])
                power_radio_list.append(breakdown["radio"])
                power_gps_list.append(breakdown["gps"])
                power_bg_list.append(breakdown["background"])
            
            # 记录无修正 SOC
            if record_uncorrected:
                soc_uncorrected_list.append(battery_uncorrected.SOC)
                soc_voltage_only_list.append(battery_voltage_only.SOC)
                soc_temperature_only_list.append(battery_temperature_only.SOC)
                soc_aging_only_list.append(battery_aging_only.SOC)

        t += dt

    result = {
        "TTL": t,
    }

    if record:
        result.update({
            "time": time_list,
            "SOC": soc_list,
            "Tb": tb_list,
            "Power": power_list,
            "State": state_list,
        })
        
        if record_breakdown:
            result.update({
                "Power_screen": power_screen_list,
                "Power_cpu": power_cpu_list,
                "Power_radio": power_radio_list,
                "Power_gps": power_gps_list,
                "Power_background": power_bg_list,
            })
        
        if record_uncorrected:
            result.update({
                "SOC_uncorrected": soc_uncorrected_list,
                "SOC_voltage_only": soc_voltage_only_list,
                "SOC_temperature_only": soc_temperature_only_list,
                "SOC_aging_only": soc_aging_only_list,
            })

    return result


# =====================================================
# Monte Carlo 仿真
# =====================================================
def run_monte_carlo(
    scenario,
    n_samples=100,
    base_seed=0,
    dt=1.0,
    T_amb=298.15,
):
    """
    Monte Carlo 多次随机仿真
    """

    ttl_list = []

    for i in range(n_samples):
        result = run_simulation(
            scenario=scenario,
            dt=dt,
            T_amb=T_amb,
            seed=base_seed + i,
            record=False,
        )
        ttl_list.append(result["TTL"])

    return ttl_list


# =====================================================
# 示例（仅测试核心仿真功能）
# =====================================================
if __name__ == "__main__":

    # ---------- 单次可复现仿真 ----------
    result = run_simulation(
        PURE_GAMING,
        seed=42,
        record=True,
    )
    print(f"TTL = {result['TTL'] / 3600:.2f} hours")

    # ---------- Monte Carlo ----------
    ttl_mc = run_monte_carlo(
        PURE_GAMING,
        n_samples=100,
        base_seed=1000,
    )
    
    import numpy as np
    print(f"Monte Carlo TTL: mean={np.mean(ttl_mc)/3600:.2f}h, std={np.std(ttl_mc)/3600:.3f}h")
