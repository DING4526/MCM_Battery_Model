# simulate.py
#
# usage → power → battery
# 支持：
#   - 随机种子控制（可复现）
#   - 时间序列记录
#   - Monte Carlo 仿真
#   - 基础可视化

import random
import matplotlib.pyplot as plt

from battery_model import BatteryModel
from power_model import total_power
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
):
    """
    单次运行仿真直到电池耗尽

    Returns
    -------
    result : dict
        {
          "TTL": float,
          "time": [...],
          "SOC": [...],
          "Tb": [...],
          "Power": [...],
          "State": [...]
        }
    """

    # ---------- 随机种子 ----------
    if seed is not None:
        random.seed(seed)

    # ---------- 初始化电池 ----------
    battery = BatteryModel(
        SOC0=1.0,
        Tb0=298.15,
        aging_loss=0.15,
    )

    # ---------- 使用行为控制器 ----------
    controller = ScenarioController(scenario)

    # ---------- 记录 ----------
    time_list = []
    soc_list = []
    tb_list = []
    power_list = []
    state_list = []

    t = 0.0

    # ---------- 主循环 ----------
    while battery.SOC > 0:

        current_state = controller.step(dt)
        state_params = get_state_params(current_state)
        P = total_power(state_params)

        battery.step(P, T_amb, dt)

        if record:
            time_list.append(t)
            soc_list.append(battery.SOC)
            tb_list.append(battery.Tb)
            power_list.append(P)
            state_list.append(current_state)

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
# 可视化工具
# =====================================================
def plot_single_run(result):
    """
    单次仿真时间序列
    """

    time_h = [t / 3600 for t in result["time"]]

    plt.figure()
    plt.plot(time_h, result["SOC"])
    plt.xlabel("Time (hours)")
    plt.ylabel("SOC")
    plt.title("SOC vs Time")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(time_h, result["Power"])
    plt.xlabel("Time (hours)")
    plt.ylabel("Power (W)")
    plt.title("Power vs Time")
    plt.grid(True)
    plt.show()


def plot_ttl_distribution(ttl_list):
    """
    TTL 分布直方图
    """

    ttl_h = [t / 3600 for t in ttl_list]

    plt.figure()
    plt.hist(ttl_h, bins=20, edgecolor="black")
    plt.xlabel("Time-to-Empty (hours)")
    plt.ylabel("Count")
    plt.title("TTL Distribution (Monte Carlo)")
    plt.grid(True)
    plt.show()


# =====================================================
# 示例
# =====================================================
if __name__ == "__main__":

    # ---------- 单次可复现仿真 ----------
    result = run_simulation(
        PURE_GAMING,
        seed=42,
        record=True,
    )
    print(f"TTL = {result['TTL'] / 3600:.2f} hours")

    plot_single_run(result)

    # ---------- Monte Carlo ----------
    ttl_mc = run_monte_carlo(
        PURE_GAMING,
        n_samples=200,
        base_seed=1000,
    )

    plot_ttl_distribution(ttl_mc)
