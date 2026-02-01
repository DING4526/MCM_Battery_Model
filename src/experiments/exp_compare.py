# experiments/exp_compare.py
# 场景对比实验模块

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_simulation, run_monte_carlo
from visualization import plot_scenario_comparison
from visualization.comparison import plot_scenario_comprehensive_comparison
from visualization.config import smart_savefig
from usage.scenario import *


# 预定义场景组合
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
    output_dir="compare",
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
        output_dir : str - 输出子目录名
    
    返回：
        results : dict - 对比结果
    """
    
    # 获取场景
    if scenarios is None:
        if group_name and group_name in SCENARIO_GROUPS:
            scenarios = SCENARIO_GROUPS[group_name]
        else:
            scenarios = SCENARIO_GROUPS["日常场景"]
            group_name = "日常场景"
    
    if verbose:
        print("=" * 60)
        print("🔬 场景对比实验")
        print("=" * 60)
        print(f"场景组: {group_name}")
        print(f"Monte Carlo 样本数: {n_mc}")
        print(f"环境温度: {T_amb - 273.15:.1f} °C")
        print("-" * 60)
    
    results = {}
    
    for name, scenario in scenarios.items():
        if verbose:
            print(f"分析场景: {name}...")
        
        ttl_list = run_monte_carlo(
            scenario=scenario,
            n_samples=n_mc,
            base_seed=base_seed,
            dt=dt,
            T_amb=T_amb,
        )
        
        results[name] = {
            "ttl_list": ttl_list,
            "mean": np.mean(ttl_list),
            "std": np.std(ttl_list),
            "min": np.min(ttl_list),
            "max": np.max(ttl_list),
            "median": np.median(ttl_list),
        }
        
        if verbose:
            print(f"  均值: {results[name]['mean']/3600:.2f} h")
    
    if verbose:
        print("-" * 60)
        print("✅ 场景对比完成！")
        sorted_scenarios = sorted(results.keys(), key=lambda s: results[s]["mean"], reverse=True)
        for i, name in enumerate(sorted_scenarios, 1):
            print(f"  {i}. {name}: {results[name]['mean']/3600:.2f} h")
        print("=" * 60)
    
    # 保存图片
    plot_scenario_comparison(results, filename="scenario_comparison.png", subdir=output_dir, show=False)
    plot_scenario_comprehensive_comparison(results, filename="scenario_comprehensive.png", subdir=output_dir, show=False)
    
    return results


def run_sensitivity_to_temperature(
    scenario=None,
    scenario_name="默认场景",
    T_amb_range=None,
    n_mc=50,
    verbose=True,
    output_dir="compare",
):
    """运行温度敏感性分析"""
    import matplotlib.pyplot as plt
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常"
    
    if T_amb_range is None:
        T_amb_range = [273.15 + t for t in [0, 10, 20, 25, 30, 35, 40]]
    
    if verbose:
        print("🌡️ 环境温度敏感性分析")
    
    results = {"temperatures": [], "means": [], "stds": []}
    
    for T_amb in T_amb_range:
        ttl_list = run_monte_carlo(scenario=scenario, n_samples=n_mc, T_amb=T_amb)
        results["temperatures"].append(T_amb - 273.15)
        results["means"].append(np.mean(ttl_list) / 3600)
        results["stds"].append(np.std(ttl_list) / 3600)
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["temperatures"], results["means"], 'o-', color='#2E86AB', linewidth=2)
    ax.fill_between(results["temperatures"],
                    [m - s for m, s in zip(results["means"], results["stds"])],
                    [m + s for m, s in zip(results["means"], results["stds"])],
                    alpha=0.3, color='#2E86AB')
    ax.set_xlabel("环境温度 (°C)")
    ax.set_ylabel("续航时间 TTL (小时)")
    ax.set_title(f"环境温度对续航时间的影响 - {scenario_name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    smart_savefig("temperature_sensitivity.png", output_dir)
    
    return results


if __name__ == "__main__":
    run_comparison_experiment()
