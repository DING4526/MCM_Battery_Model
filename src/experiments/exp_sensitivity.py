# experiments/exp_sensitivity.py
# 敏感度分析实验模块

import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_monte_carlo
from usage.state import USAGE_STATES
from visualization import plot_sensitivity_bar
from visualization.sensitivity_plot import plot_sensitivity_comprehensive
from visualization.config import smart_savefig
from usage.scenario import *


# 可分析的敏感度参数
SENS_PARAMS = ["u", "r", "u_cpu", "lambda_cell", "delta_signal", "r_on"]

PARAM_DESCRIPTIONS = {
    "u": "屏幕亮度",
    "r": "刷新率",
    "u_cpu": "CPU 利用率",
    "lambda_cell": "蜂窝网络比例",
    "delta_signal": "信号质量修正",
    "r_on": "GPS 开启比例",
}


def _perturb_usage(param, factor):
    """对所有 usage 状态的某个参数进行比例扰动"""
    for state in USAGE_STATES.values():
        if param in state:
            state[param] *= factor


def run_sensitivity_experiment(
    scenario=None,
    scenario_name="默认场景",
    param_list=None,
    eps=0.2,
    n_mc=100,
    verbose=True,
    output_dir="sensitivity",
):
    """
    运行敏感度分析实验
    
    参数：
        scenario : dict - 使用场景配置
        scenario_name : str - 场景名称
        param_list : list - 要分析的参数列表
        eps : float - 扰动比例（默认 ±20%）
        n_mc : int - 每次扰动的 Monte Carlo 样本数
        verbose : bool - 是否输出详细信息
        output_dir : str - 输出子目录名
    
    返回：
        results : dict - 敏感度分析结果
    """
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MARKOV
        scenario_name = "学生日常 Markov"
    
    if param_list is None:
        param_list = SENS_PARAMS
    
    if verbose:
        print("=" * 60)
        print("📊 敏感度分析实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"扰动幅度: ±{eps*100:.0f}%")
        print(f"Monte Carlo 样本数: {n_mc}")
        print("-" * 60)
    
    # 保存原始 usage 参数
    original_states = copy.deepcopy(USAGE_STATES)
    
    # 计算基准 TTL
    if verbose:
        print("计算基准 TTL...")
    
    ttl_base_list = run_monte_carlo(scenario, n_samples=n_mc)
    ttl_base = sum(ttl_base_list) / n_mc
    
    if verbose:
        print(f"基准 TTL: {ttl_base/3600:.2f} 小时")
    
    results = {}
    
    for p in param_list:
        if verbose:
            print(f"分析参数: {PARAM_DESCRIPTIONS.get(p, p)}...")
        
        # 正扰动
        _perturb_usage(p, 1 + eps)
        ttl_plus = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc
        
        # 恢复
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))
        
        # 负扰动
        _perturb_usage(p, 1 - eps)
        ttl_minus = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc
        
        # 恢复
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))
        
        # 中心差分敏感度
        S = (ttl_plus - ttl_minus) / (2 * eps)
        S_norm = S / ttl_base
        
        results[p] = {
            "TTL+": ttl_plus,
            "TTL-": ttl_minus,
            "S": S,
            "S_norm": S_norm,
        }
    
    if verbose:
        print("-" * 60)
        print("✅ 敏感度分析完成！")
        sorted_params = sorted(results.keys(), key=lambda p: abs(results[p]["S_norm"]), reverse=True)
        for i, p in enumerate(sorted_params, 1):
            sign = "+" if results[p]["S_norm"] > 0 else "-"
            print(f"  {i}. {PARAM_DESCRIPTIONS.get(p, p)}: {sign}{abs(results[p]['S_norm']):.4f}")
        print("=" * 60)
    
    # 保存图片
    plot_sensitivity_bar(results, filename="sensitivity_bar.png", subdir=output_dir, show=False)
    plot_sensitivity_comprehensive(results, ttl_base, filename="sensitivity_comprehensive.png", subdir=output_dir, show=False)
    
    results["_baseline_ttl"] = ttl_base
    return results


def run_multi_eps_sensitivity(
    scenario=None,
    scenario_name="默认场景",
    param_list=None,
    eps_list=None,
    n_mc=50,
    verbose=True,
    output_dir="sensitivity",
):
    """运行多扰动幅度敏感度分析"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MARKOV
    
    if param_list is None:
        param_list = SENS_PARAMS[:3]
    
    if eps_list is None:
        eps_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    multi_results = {p: {"eps": [], "S_norm": []} for p in param_list}
    
    for eps in eps_list:
        results = run_sensitivity_experiment(
            scenario=scenario,
            param_list=param_list,
            eps=eps,
            n_mc=n_mc,
            verbose=False,
            output_dir=output_dir,
        )
        
        for p in param_list:
            multi_results[p]["eps"].append(eps * 100)
            multi_results[p]["S_norm"].append(results[p]["S_norm"])
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, p in enumerate(param_list):
        ax.plot(multi_results[p]["eps"], multi_results[p]["S_norm"], 
                'o-', color=colors[i % len(colors)], linewidth=2,
                label=PARAM_DESCRIPTIONS.get(p, p))
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("扰动幅度 (%)")
    ax.set_ylabel("归一化敏感度")
    ax.set_title("敏感度与扰动幅度关系")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    smart_savefig("multi_eps.png", output_dir)
    
    return multi_results


if __name__ == "__main__":
    run_sensitivity_experiment()
