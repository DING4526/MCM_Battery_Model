# simulate.py
#
# 核心仿真模块（无可视化）
#
# usage → power → battery
# 支持：
#   - 随机种子控制（可复现）
#   - 时间序列记录
#   - 子模块功耗分解记录（screen/cpu/radio/background）
#   - 各种修正版本 SOC 对比（可选）
#   - Monte Carlo 仿真
#
# 升级点（面向“人群/CSV驱动”半重构）：
#   - 支持每个 device 注入：容量 battery_capacity_mah、usage_states_override、device_params_override
#   - 支持固定 dwell（默认 15min）用于 mixed 场景的状态驻留
#   - 移除 GPS：状态集建议为 IDLE/SOCIAL/VIDEO/GAME，对应 power_model 已删 GPS

import random
from typing import Optional, Dict, Any

from src.battery_model import BatteryModel
from src.power_model import total_power, power_breakdown
from src.usage.state import get_state_params, set_usage_states
from src.usage.control import ScenarioController

# 兼容：如果你还保留旧 scenario 常量，这里可以继续 import；人群模拟时可以传入自定义 scenario
try:
    from usage.scenario import *  # noqa
except Exception:
    pass

# =====================================================
# 常量定义
# =====================================================
# 固定标称电压（V），用于“固定电压版本”的 SOC 计算（无修正/仅温度/仅老化）
NOMINAL_VOLTAGE = 3.7



# =====================================================
# 单次仿真
# =====================================================
def run_simulation(
    scenario: Dict[str, Any],
    dt: float = 1.0,
    T_amb: float = 298.15,
    seed: Optional[int] = None,
    record: bool = True,
    record_breakdown: bool = False,
    record_uncorrected: bool = False,
    # ===== 新增：人群模拟/数据驱动注入 =====
    usage_states_override: Optional[Dict[str, Dict[str, float]]] = None,
    device_params_override: Optional[Dict[str, float]] = None,
    battery_capacity_mah: Optional[float] = None,
    battery_aging_loss: Optional[float] = None,
    fixed_dwell_sec: int = 15 * 60,
    dwell_mode: str = "fixed",  # "fixed" or "random"（随机驻留时间用于以后升级）
):
    """
    单次运行仿真直到电池耗尽

    参数：
        scenario : dict - 使用场景配置（建议 {"type":"mixed","state_ratio":{...}}）
        dt : float - 时间步长（秒）
        T_amb : float - 环境温度（K）
        seed : int - 随机种子
        record : bool - 是否记录时间序列
        record_breakdown : bool - 是否记录子模块功耗分解（screen/cpu/radio/background）
        record_uncorrected : bool - 是否记录各种修正版本的 SOC 对比

        usage_states_override : dict - 覆盖 usage.state.USAGE_STATES（设备级 data-driven）
        device_params_override : dict - 覆盖 usage.state.DEVICE_PARAMS（设备级差异可扩展）
        battery_capacity_mah : float - 电池容量（mAh），将映射为 BatteryModel(capacity_Ah)
        fixed_dwell_sec : int - mixed 场景每段状态固定驻留时长（默认 15min）
        dwell_mode : str - "fixed" or "random"

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
          "Power_background": [...],
          # 如果 record_uncorrected=True:
          "SOC_uncorrected": [...],        # 完全无修正（固定电压、无温度、无老化）
          "SOC_voltage_only": [...],       # 仅电压修正（OCV-SOC曲线，无温度、无老化）
          "SOC_temperature_only": [...],   # 仅温度修正（固定电压，有温度，无老化）
          "SOC_aging_only": [...],         # 仅老化修正（固定电压，无温度，有老化）
        }
    """

    # ---------- 输入检查 ----------
    if dt <= 0:
        raise ValueError("dt 必须 > 0")

    # ---------- 随机种子 ----------
    if seed is not None:
        random.seed(seed)

    # ---------- 覆盖 usage states（设备级 data-driven） ----------
    if usage_states_override is not None:
        set_usage_states(usage_states_override)

    # ---------- 容量注入：mAh -> Ah ----------
    cap_ah = None
    if battery_capacity_mah is not None:
        cap_ah = float(battery_capacity_mah) / 1000.0

    # ---------- 初始化电池（带修正：OCV+温度+老化） ----------
    aging_loss = 0.15 if battery_aging_loss is None else float(battery_aging_loss)

    battery = BatteryModel(
        SOC0=1.0,
        Tb0=298.15,
        aging_loss=aging_loss,
        capacity_Ah=cap_ah if cap_ah is not None else 5.0,
    )

    # ---------- 初始化对比电池（可选） ----------
    if record_uncorrected:
        # 完全无修正：固定电压 + 无温度 + 无老化
        battery_uncorrected = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,
            alpha=0.0,
            capacity_Ah=cap_ah if cap_ah is not None else 5.0,
        )

        # 仅电压修正：OCV-SOC（无温度、无老化）
        battery_voltage_only = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,
            alpha=0.0,
            capacity_Ah=cap_ah if cap_ah is not None else 5.0,
        )

        # 仅温度修正：固定电压 + 温度修正（无老化）
        battery_temperature_only = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=0.0,
            alpha=battery.alpha,  # 与主电池一致
            capacity_Ah=cap_ah if cap_ah is not None else 5.0,
        )

        # 仅老化修正：固定电压 + 老化修正（无温度）
        battery_aging_only = BatteryModel(
            SOC0=1.0,
            Tb0=298.15,
            aging_loss=aging_loss,  # 与主电池一致
            alpha=0.0,
            capacity_Ah=cap_ah if cap_ah is not None else 5.0,
        )

    # ---------- 使用行为控制器 ----------
    # 注意：你的 ScenarioController 需要支持 fixed_dwell_sec / dwell_mode 参数
    # 若你已按我之前建议修改 usage/control.py，则下面可直接工作
    try:
        controller = ScenarioController(
            scenario,
            dwell_mode=dwell_mode,
            fixed_dwell_sec=fixed_dwell_sec,
        )
    except TypeError:
        # 兼容旧版 ScenarioController（不支持固定驻留参数）
        controller = ScenarioController(scenario)

    # ---------- 记录 ----------
    time_list = []
    soc_list = []
    tb_list = []
    power_list = []
    state_list = []

    # 子模块功耗分解记录（已移除 GPS）
    power_screen_list = []
    power_cpu_list = []
    power_radio_list = []
    power_bg_list = []

    # 无修正/单因素修正 SOC 记录
    soc_uncorrected_list = []
    soc_voltage_only_list = []
    soc_temperature_only_list = []
    soc_aging_only_list = []

    t = 0.0

    # ---------- 主循环 ----------
    while battery.SOC > 0.0:

        current_state = controller.step(dt)
        state_params = get_state_params(current_state, device_params_override=device_params_override)
        P = total_power(state_params)

        # 主电池推进（包含 OCV+温度+老化）
        battery.step(P, T_amb, dt)

        # 对比电池推进
        if record_uncorrected:
            # 1) 完全无修正：固定电压
            Q_nom = battery_uncorrected.Q_nom
            dSOC = -P / (NOMINAL_VOLTAGE * Q_nom) * dt
            battery_uncorrected.SOC = max(0.0, battery_uncorrected.SOC + dSOC)

            # 2) 仅电压修正：使用 OCV-SOC 曲线
            V_oc = battery_voltage_only.voc(battery_voltage_only.SOC)
            Q_eff_v = battery_voltage_only.Q_nom  # 无温度/老化：Q_eff = Q_nom
            dSOC_v = -P / (V_oc * Q_eff_v) * dt
            battery_voltage_only.SOC = max(0.0, battery_voltage_only.SOC + dSOC_v)

            # 3) 仅温度修正：固定电压 + 温度因子（同步温度）
            battery_temperature_only.Tb = battery.Tb  # 同步主电池温度轨迹
            Q_eff_t = battery_temperature_only.effective_capacity(battery_temperature_only.Tb)
            # 注意：battery_temperature_only 的 fA=1（aging_loss=0），因此 Q_eff_t=Q_nom*f_T
            dSOC_t = -P / (NOMINAL_VOLTAGE * Q_eff_t) * dt
            battery_temperature_only.SOC = max(0.0, battery_temperature_only.SOC + dSOC_t)

            # 4) 仅老化修正：固定电压 + fA（无温度）
            Q_eff_a = battery_aging_only.Q_nom * battery_aging_only.fA
            dSOC_a = -P / (NOMINAL_VOLTAGE * Q_eff_a) * dt
            battery_aging_only.SOC = max(0.0, battery_aging_only.SOC + dSOC_a)

        # ---------- 写记录 ----------
        if record:
            time_list.append(t)
            soc_list.append(battery.SOC)
            tb_list.append(battery.Tb)
            power_list.append(P)
            state_list.append(current_state)

            if record_breakdown:
                br = power_breakdown(state_params)
                power_screen_list.append(br["screen"])
                power_cpu_list.append(br["cpu"])
                power_radio_list.append(br["radio"])
                power_bg_list.append(br["background"])

            if record_uncorrected:
                soc_uncorrected_list.append(battery_uncorrected.SOC)
                soc_voltage_only_list.append(battery_voltage_only.SOC)
                soc_temperature_only_list.append(battery_temperature_only.SOC)
                soc_aging_only_list.append(battery_aging_only.SOC)

        t += dt

    # ---------- 输出 ----------
    result = {"TTL": t}

    if record:
        result.update(
            {
                "time": time_list,
                "SOC": soc_list,
                "Tb": tb_list,
                "Power": power_list,
                "State": state_list,
            }
        )

        if record_breakdown:
            result.update(
                {
                    "Power_screen": power_screen_list,
                    "Power_cpu": power_cpu_list,
                    "Power_radio": power_radio_list,
                    "Power_background": power_bg_list,
                }
            )

        if record_uncorrected:
            result.update(
                {
                    "SOC_uncorrected": soc_uncorrected_list,
                    "SOC_voltage_only": soc_voltage_only_list,
                    "SOC_temperature_only": soc_temperature_only_list,
                    "SOC_aging_only": soc_aging_only_list,
                }
            )

    return result


# =====================================================
# Monte Carlo 仿真
# =====================================================
def run_monte_carlo(
    scenario: Dict[str, Any],
    n_samples: int = 100,
    base_seed: int = 0,
    dt: float = 1.0,
    T_amb: float = 298.15,
    # MC 也支持注入（方便对某类 device 做随机性评估）
    usage_states_override: Optional[Dict[str, Dict[str, float]]] = None,
    device_params_override: Optional[Dict[str, float]] = None,
    battery_capacity_mah: Optional[float] = None,
    fixed_dwell_sec: int = 15 * 60,
    dwell_mode: str = "fixed",
):
    """
    Monte Carlo 多次随机仿真（仅统计 TTL）

    返回：
        ttl_list : list[float]  # 秒
    """
    ttl_list = []

    for i in range(n_samples):
        result = run_simulation(
            scenario=scenario,
            dt=dt,
            T_amb=T_amb,
            seed=base_seed + i,
            record=False,
            record_breakdown=False,
            record_uncorrected=False,
            usage_states_override=usage_states_override,
            device_params_override=device_params_override,
            battery_capacity_mah=battery_capacity_mah,
            fixed_dwell_sec=fixed_dwell_sec,
            dwell_mode=dwell_mode,
        )
        ttl_list.append(result["TTL"])

    return ttl_list


# =====================================================
# 示例（仅测试核心仿真功能）
# =====================================================
if __name__ == "__main__":

    # 如果你还有旧的 PURE_GAMING，就能跑；否则请传一个 scenario dict
    try:
        test_scenario = PURE_GAMING  # noqa
    except Exception:
        test_scenario = {"type": "mixed", "state_ratio": {"GAME": 1.0}}

    result = run_simulation(
        scenario=test_scenario,
        seed=42,
        record=True,
        record_breakdown=True,
        record_uncorrected=True,
        battery_capacity_mah=5000,
        fixed_dwell_sec=15 * 60,
    )
    print(f"TTL = {result['TTL'] / 3600:.2f} hours")

    ttl_mc = run_monte_carlo(
        scenario=test_scenario,
        n_samples=50,
        base_seed=1000,
        battery_capacity_mah=5000,
        fixed_dwell_sec=15 * 60,
    )

    import numpy as np
    print(
        f"Monte Carlo TTL: mean={np.mean(ttl_mc)/3600:.2f}h, std={np.std(ttl_mc)/3600:.3f}h"
    )
