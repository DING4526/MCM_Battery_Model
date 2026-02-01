# experiments/exp_sensitivity.py
# 敏感度分析实验模块
#
# 提供敏感度分析实验功能：
# - 参数敏感度分析
# - 多种可视化方式
# - 分析报告生成

import sys
import os
import copy

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_monte_carlo
from usage.state import USAGE_STATES
from visualization import (
    plot_sensitivity_bar,
    plot_sensitivity_tornado,
    plot_sensitivity_spider,
    plot_sensitivity_heatmap,
)
from visualization.sensitivity_plot import plot_sensitivity_comprehensive
from usage.scenario import *


# =====================================================
# 可分析的敏感度参数
# =====================================================

SENS_PARAMS = [
    "u",            # 屏幕亮度
    "r",            # 刷新率
    "u_cpu",        # CPU 利用率
    "lambda_cell",  # 蜂窝比例
    "delta_signal", # 信号质量
    "r_on",         # GPS 开启比例
]

PARAM_DESCRIPTIONS = {
    "u": "屏幕亮度",
    "r": "刷新率",
    "u_cpu": "CPU 利用率",
    "lambda_cell": "蜂窝网络比例",
    "delta_signal": "信号质量修正",
    "r_on": "GPS 开启比例",
}


def _perturb_usage(param, factor):
    """
    对所有 usage 状态的某个参数进行比例扰动
    """
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
    visualize=True,
    comprehensive_plot=False,
    save_prefix=None,
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
        visualize : bool - 是否可视化结果
        comprehensive_plot : bool - 是否显示综合分析图
        save_prefix : str - 图片保存路径前缀
    
    返回：
        results : dict - 敏感度分析结果
    """
    
    # 默认场景
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MARKOV
        scenario_name = "学生日常 Markov 场景"
    
    # 默认参数列表
    if param_list is None:
        param_list = SENS_PARAMS
    
    if verbose:
        print("=" * 60)
        print("📊 敏感度分析实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"扰动幅度: ±{eps*100:.0f}%")
        print(f"Monte Carlo 样本数: {n_mc}")
        print(f"分析参数: {[PARAM_DESCRIPTIONS.get(p, p) for p in param_list]}")
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
        print("-" * 60)
    
    results = {}
    
    for p in param_list:
        if verbose:
            print(f"分析参数: {PARAM_DESCRIPTIONS.get(p, p)}...")
        
        # 正扰动
        _perturb_usage(p, 1 + eps)
        ttl_plus_list = run_monte_carlo(scenario, n_samples=n_mc)
        ttl_plus = sum(ttl_plus_list) / n_mc
        
        # 恢复
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))
        
        # 负扰动
        _perturb_usage(p, 1 - eps)
        ttl_minus_list = run_monte_carlo(scenario, n_samples=n_mc)
        ttl_minus = sum(ttl_minus_list) / n_mc
        
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
        
        if verbose:
            print(f"  TTL+{eps*100:.0f}%: {ttl_plus/3600:.2f} h")
            print(f"  TTL-{eps*100:.0f}%: {ttl_minus/3600:.2f} h")
            print(f"  归一化敏感度: {S_norm:.4f}")
    
    if verbose:
        print("-" * 60)
        print("✅ 敏感度分析完成！")
        print("-" * 60)
        
        # 排序输出
        sorted_params = sorted(results.keys(), key=lambda p: abs(results[p]["S_norm"]), reverse=True)
        print("📈 敏感度排名（按绝对值）:")
        for i, p in enumerate(sorted_params, 1):
            sign = "+" if results[p]["S_norm"] > 0 else "-"
            print(f"  {i}. {PARAM_DESCRIPTIONS.get(p, p)}: {sign}{abs(results[p]['S_norm']):.4f}")
        
        print("=" * 60)
    
    # 可视化
    if visualize:
        if comprehensive_plot:
            # 综合分析图
            save_path = f"{save_prefix}_sens_comprehensive.png" if save_prefix else None
            plot_sensitivity_comprehensive(results, ttl_base, save_path=save_path)
        else:
            # 简单柱状图
            save_path = f"{save_prefix}_sens_bar.png" if save_prefix else None
            plot_sensitivity_bar(results, save_path=save_path)
    
    # 添加基准 TTL 到结果
    results["_baseline_ttl"] = ttl_base
    
    return results


def run_multi_eps_sensitivity(
    scenario=None,
    scenario_name="默认场景",
    param_list=None,
    eps_list=None,
    n_mc=50,
    verbose=True,
    visualize=True,
):
    """
    运行多扰动幅度敏感度分析
    
    参数：
        scenario : dict - 使用场景配置
        scenario_name : str - 场景名称
        param_list : list - 要分析的参数列表
        eps_list : list - 扰动幅度列表
        n_mc : int - Monte Carlo 样本数
        verbose : bool - 是否输出详细信息
        visualize : bool - 是否可视化结果
    
    返回：
        multi_results : dict - 多扰动幅度分析结果
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MARKOV
        scenario_name = "学生日常 Markov 场景"
    
    if param_list is None:
        param_list = SENS_PARAMS[:3]  # 默认只分析前三个参数
    
    if eps_list is None:
        eps_list = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    if verbose:
        print("=" * 60)
        print("📊 多扰动幅度敏感度分析")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"扰动幅度: {[f'{e*100:.0f}%' for e in eps_list]}")
        print("-" * 60)
    
    multi_results = {p: {"eps": [], "S_norm": []} for p in param_list}
    
    for eps in eps_list:
        if verbose:
            print(f"\n扰动幅度 ±{eps*100:.0f}%:")
        
        results = run_sensitivity_experiment(
            scenario=scenario,
            param_list=param_list,
            eps=eps,
            n_mc=n_mc,
            verbose=False,
            visualize=False,
        )
        
        for p in param_list:
            multi_results[p]["eps"].append(eps * 100)
            multi_results[p]["S_norm"].append(results[p]["S_norm"])
            
            if verbose:
                print(f"  {PARAM_DESCRIPTIONS.get(p, p)}: S_norm = {results[p]['S_norm']:.4f}")
    
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#DC3545', '#6C757D']
        
        for i, p in enumerate(param_list):
            ax.plot(multi_results[p]["eps"], multi_results[p]["S_norm"], 
                    'o-', color=colors[i % len(colors)], linewidth=2, markersize=8,
                    label=PARAM_DESCRIPTIONS.get(p, p))
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("扰动幅度 (%)")
        ax.set_ylabel("归一化敏感度")
        ax.set_title(f"敏感度与扰动幅度关系 - {scenario_name}", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print("=" * 60)
    
    return multi_results


def run_quick_demo():
    """
    快速演示敏感度分析
    """
    print("\n" + "🚀 快速演示：敏感度分析\n")
    
    # 基础敏感度分析
    results = run_sensitivity_experiment(
        scenario=SCENARIO_STUDENT_DAILY_MARKOV,
        scenario_name="学生日常 Markov",
        n_mc=50,  # 演示用较少样本
        verbose=True,
        visualize=True,
        comprehensive_plot=True,
    )


if __name__ == "__main__":
    run_quick_demo()
