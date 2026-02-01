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
        record_uncorrected : bool - 是否记录无修正版本的 SOC

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
          "SOC_uncorrected": [...]  # 无电压/温度/老化修正的 SOC
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
    if record_uncorrected:
        # 无修正电池：无老化损失，固定电压
        battery_uncorrected = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,  # 无老化修正
            alpha=0.0,       # 无温度修正
        )
        # 固定电压用于简化计算（覆盖 voc 方法）
        battery_uncorrected._fixed_voltage = 3.7  # 固定标称电压

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

    t = 0.0

    # ---------- 主循环 ----------
    while battery.SOC > 0:

        current_state = controller.step(dt)
        state_params = get_state_params(current_state)
        P = total_power(state_params)

        battery.step(P, T_amb, dt)
        
        # 更新无修正电池
        if record_uncorrected:
            # 使用简化的 SOC 计算：dSOC/dt = -P / (V_nom * Q_nom)
            V_nom = 3.7  # 固定标称电压
            Q_nom = battery_uncorrected.Q_nom
            dSOC = -P / (V_nom * Q_nom) * dt
            battery_uncorrected.SOC = max(0.0, battery_uncorrected.SOC + dSOC)

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
