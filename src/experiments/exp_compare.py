# experiments/exp_compare.py
# 场景对比实验模块
#
# 提供场景对比实验功能：
# - 多场景 Monte Carlo 对比
# - 统计对比分析
# - 对比可视化

import sys
import os
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_simulation, run_monte_carlo
from visualization import (
    plot_scenario_comparison,
    plot_scenario_boxplot,
    plot_scenario_radar,
    plot_multi_scenario_timeline,
)
from visualization.comparison import plot_scenario_comprehensive_comparison
from usage.scenario import *


# =====================================================
# 预定义场景组合
# =====================================================

SCENARIO_GROUPS = {
    "日常场景": {
        "学生日常": SCENARIO_STUDENT_DAILY_MIXED,
        "通勤": SCENARIO_COMMUTE_MIXED,
        "周末娱乐": SCENARIO_WEEKEND_MIXED,
        "旅行": SCENARIO_TRAVEL_MIXED,
    },
    "极端场景": {
        "纯待机": PURE_DEEPIDLE,
        "纯社交": PURE_SOCIAL,
        "纯视频": PURE_VIDEO,
        "纯游戏": PURE_GAMING,
        "纯导航": PURE_NAVIGATION,
    },
    "混合 vs Markov": {
        "学生日常 Mixed": SCENARIO_STUDENT_DAILY_MIXED,
        "学生日常 Markov": SCENARIO_STUDENT_DAILY_MARKOV,
        "通勤 Mixed": SCENARIO_COMMUTE_MIXED,
        "通勤 Markov": SCENARIO_COMMUTE_MARKOV,
    },
}


def run_comparison_experiment(
    scenarios=None,
    group_name=None,
    n_mc=100,
    base_seed=0,
    dt=1.0,
    T_amb=298.15,
    verbose=True,
    visualize=True,
    comprehensive_plot=False,
    include_timeline=False,
    save_prefix=None,
):
    """
    运行场景对比实验
    
    参数：
        scenarios : dict - 场景字典 {name: scenario_config}
        group_name : str - 预定义场景组名称
        n_mc : int - Monte Carlo 样本数
        base_seed : int - 基础随机种子
        dt : float - 时间步长（秒）
        T_amb : float - 环境温度（K）
        verbose : bool - 是否输出详细信息
        visualize : bool - 是否可视化结果
        comprehensive_plot : bool - 是否显示综合分析图
        include_timeline : bool - 是否包含时间线对比
        save_prefix : str - 图片保存路径前缀
    
    返回：
        results : dict - 对比结果
    """
    
    # 获取场景
    if scenarios is None:
        if group_name is not None and group_name in SCENARIO_GROUPS:
            scenarios = SCENARIO_GROUPS[group_name]
        else:
            scenarios = SCENARIO_GROUPS["日常场景"]
            group_name = "日常场景"
    
    if group_name is None:
        group_name = "自定义场景组"
    
    if verbose:
        print("=" * 60)
        print("🔬 场景对比实验")
        print("=" * 60)
        print(f"场景组: {group_name}")
        print(f"包含场景: {list(scenarios.keys())}")
        print(f"Monte Carlo 样本数: {n_mc}")
        print(f"环境温度: {T_amb - 273.15:.1f} °C")
        print("-" * 60)
    
    results = {}
    single_results = {}  # 单次仿真结果（用于时间线）
    
    for name, scenario in scenarios.items():
        if verbose:
            print(f"分析场景: {name}...")
        
        # Monte Carlo 仿真
        ttl_list = run_monte_carlo(
            scenario=scenario,
            n_samples=n_mc,
            base_seed=base_seed,
            dt=dt,
            T_amb=T_amb,
        )
        
        # 计算统计量
        results[name] = {
            "ttl_list": ttl_list,
            "mean": np.mean(ttl_list),
            "std": np.std(ttl_list),
            "min": np.min(ttl_list),
            "max": np.max(ttl_list),
            "median": np.median(ttl_list),
            "q1": np.percentile(ttl_list, 25),
            "q3": np.percentile(ttl_list, 75),
        }
        
        if verbose:
            print(f"  均值: {results[name]['mean']/3600:.2f} h, 标准差: {results[name]['std']/3600:.3f} h")
        
        # 单次仿真（用于时间线）
        if include_timeline:
            single_result = run_simulation(
                scenario=scenario,
                dt=dt,
                T_amb=T_amb,
                seed=base_seed,
                record=True,
            )
            single_results[name] = single_result
    
    if verbose:
        print("-" * 60)
        print("✅ 场景对比分析完成！")
        print("-" * 60)
        
        # 排序输出
        sorted_scenarios = sorted(results.keys(), key=lambda s: results[s]["mean"], reverse=True)
        print("📊 续航排名（由高到低）:")
        best_ttl = results[sorted_scenarios[0]]["mean"]
        for i, name in enumerate(sorted_scenarios, 1):
            ttl = results[name]["mean"]
            relative = (ttl / best_ttl - 1) * 100
            print(f"  {i}. {name}: {ttl/3600:.2f} h ({relative:+.1f}%)")
        
        print("=" * 60)
    
    # 可视化
    if visualize:
        if comprehensive_plot:
            # 综合分析图
            save_path = f"{save_prefix}_compare_comprehensive.png" if save_prefix else None
            plot_scenario_comprehensive_comparison(
                results, 
                results_dict=single_results if include_timeline else None,
                save_path=save_path
            )
        else:
            # 简单柱状图
            save_path = f"{save_prefix}_compare_bar.png" if save_prefix else None
            plot_scenario_comparison(results, save_path=save_path)
    
    return results


def run_sensitivity_to_temperature(
    scenario=None,
    scenario_name="默认场景",
    T_amb_range=None,
    n_mc=50,
    verbose=True,
    visualize=True,
):
    """
    运行温度敏感性分析
    
    参数：
        scenario : dict - 使用场景配置
        scenario_name : str - 场景名称
        T_amb_range : list - 环境温度范围（K）
        n_mc : int - Monte Carlo 样本数
        verbose : bool - 是否输出详细信息
        visualize : bool - 是否可视化结果
    
    返回：
        results : dict - 温度敏感性结果
    """
    import matplotlib.pyplot as plt
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常"
    
    if T_amb_range is None:
        # 0°C 到 40°C
        T_amb_range = [273.15 + t for t in [0, 10, 20, 25, 30, 35, 40]]
    
    if verbose:
        print("=" * 60)
        print("🌡️ 环境温度敏感性分析")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"温度范围: {[t-273.15 for t in T_amb_range]} °C")
        print("-" * 60)
    
    results = {
        "temperatures": [t - 273.15 for t in T_amb_range],
        "means": [],
        "stds": [],
    }
    
    for T_amb in T_amb_range:
        ttl_list = run_monte_carlo(
            scenario=scenario,
            n_samples=n_mc,
            T_amb=T_amb,
        )
        
        mean_ttl = np.mean(ttl_list) / 3600
        std_ttl = np.std(ttl_list) / 3600
        
        results["means"].append(mean_ttl)
        results["stds"].append(std_ttl)
        
        if verbose:
            print(f"  {T_amb-273.15:5.1f}°C: 均值={mean_ttl:.2f} h, 标准差={std_ttl:.3f} h")
    
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        temps = results["temperatures"]
        means = results["means"]
        stds = results["stds"]
        
        ax.plot(temps, means, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='平均 TTL')
        ax.fill_between(temps, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3, color='#2E86AB', label='±1σ')
        
        ax.set_xlabel("环境温度 (°C)")
        ax.set_ylabel("续航时间 TTL (小时)")
        ax.set_title(f"环境温度对续航时间的影响 - {scenario_name}", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加参考线
        ax.axvline(x=25, color='green', linestyle='--', alpha=0.5, label='室温 (25°C)')
        
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print("=" * 60)
    
    return results


def run_all_group_comparisons(n_mc=50, verbose=True):
    """
    运行所有预定义场景组的对比实验
    """
    all_results = {}
    
    for group_name in SCENARIO_GROUPS.keys():
        if verbose:
            print(f"\n{'='*60}")
            print(f"场景组: {group_name}")
            print('='*60)
        
        results = run_comparison_experiment(
            group_name=group_name,
            n_mc=n_mc,
            verbose=verbose,
            visualize=True,
            comprehensive_plot=True,
        )
        
        all_results[group_name] = results
    
    return all_results


def run_quick_demo():
    """
    快速演示场景对比
    """
    print("\n" + "🚀 快速演示：场景对比分析\n")
    
    # 日常场景对比
    results = run_comparison_experiment(
        group_name="日常场景",
        n_mc=50,
        verbose=True,
        visualize=True,
        comprehensive_plot=True,
        include_timeline=True,
    )
    
    print("\n进行极端场景对比...")
    
    # 极端场景对比
    results_extreme = run_comparison_experiment(
        group_name="极端场景",
        n_mc=50,
        verbose=True,
        visualize=True,
        comprehensive_plot=True,
    )


if __name__ == "__main__":
    run_quick_demo()
