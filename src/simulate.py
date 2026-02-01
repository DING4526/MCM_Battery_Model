# simulate.py
#
# 把 usage → power → battery 串成一条线
# 显式时间推进
# 不做任何多余封装
# 使用 ScenarioController 统一 mixed / markov 行为逻辑

from battery_model import BatteryModel
from power_model import total_power

from usage.state import get_state_params
from usage.control import ScenarioController


def run_simulation(
    scenario,
    dt=1.0,
    T_amb=298.15,
):
    """
    运行仿真直到电池耗尽

    scenario:
        {
          "type": "mixed" / "markov",
          ...
        }
    """

    # -----------------------------
    # 初始化电池
    # -----------------------------
    battery = BatteryModel(
        SOC0=1.0,
        Tb0=298.15,
        aging_loss=0.15,
    )

    t = 0.0

    # -----------------------------
    # 初始化使用行为控制器
    # -----------------------------
    controller = ScenarioController(scenario)

    # -----------------------------
    # 主循环：直到 SOC 跑空
    # -----------------------------
    while battery.SOC > 0:

        # ===== 使用动态（统一入口）=====
        current_state = controller.step(dt)

        # ===== 功耗计算 =====
        state_params = get_state_params(current_state)
        P = total_power(state_params)

        # ===== 电池推进 =====
        battery.step(P, T_amb, dt)
        t += dt

    return t  # TTL（秒）


# -----------------------------
# 示例运行
# -----------------------------
if __name__ == "__main__":

    from usage.scenario import (
        SCENARIO_STUDENT_MIXED,
        SCENARIO_COMMUTE_MARKOV,
    )

    # ttl = run_simulation(SCENARIO_COMMUTE_MARKOV)
    ttl = run_simulation(SCENARIO_STUDENT_MIXED)

    print(f"TTL = {ttl / 3600:.2f} hours")
