# sensitivity.py
#
# Usage 参数敏感度分析模块
#
# 基于有限差分：
#   对单个参数 ±扰动
#   观察 TTL 变化
#

import copy

from src.simulate import run_monte_carlo
from src.usage.state import USAGE_STATES


# =====================================================
# 可选敏感参数（usage 相关）
# =====================================================

SENS_PARAMS = [
    "u",            # 屏幕亮度
    "r",            # 刷新率
    "u_cpu",        # CPU 利用率
    "lambda_cell",  # 蜂窝比例
    "delta_signal", # 信号质量
    "r_on",         # GPS 开启比例
]


def perturb_usage(param, factor):
    """
    对所有 usage 状态的某个参数进行比例扰动
    """

    for state in USAGE_STATES.values():
        if param in state:
            state[param] *= factor


def sensitivity_analysis(
    scenario,
    param_list=SENS_PARAMS,
    eps=0.2,
    n_mc=100,
):
    """
    参数敏感度分析

    eps : 扰动比例（默认 ±10%）
    """

    # 保存原始 usage 参数
    original_states = copy.deepcopy(USAGE_STATES)

    print("\n===== Baseline =====")
    ttl_base = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc

    results = {}

    for p in param_list:

        print(f"\n--- 参数: {p} ---")

        # 正扰动
        perturb_usage(p, 1 + eps)
        ttl_plus = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc

        # 恢复
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))

        # 负扰动
        perturb_usage(p, 1 - eps)
        ttl_minus = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc

        # 恢复
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))

        # 中心差分敏感度
        S = (ttl_plus - ttl_minus) / (2 * eps)

        # 归一化敏感度
        S_norm = S / ttl_base

        results[p] = {
            "TTL+": ttl_plus,
            "TTL-": ttl_minus,
            "S": S,
            "S_norm": S_norm,
        }

        print(f"TTL+ = {ttl_plus/3600:.2f} h")
        print(f"TTL- = {ttl_minus/3600:.2f} h")
        print(f"S_norm = {S_norm:.3f}")

    return results

if __name__ == "__main__":
    from usage.scenario import *

    res = sensitivity_analysis(
        SCENARIO_STUDENT_DAILY_MARKOV,
        n_mc=100,
    )