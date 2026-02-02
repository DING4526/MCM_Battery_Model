# experiments/exp_compare.py
# 场景对比实验模块（Plotly 版本）

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_simulation, run_monte_carlo
from visualization.comparison import (
    plot_scenario_comparison,
    plot_scenario_boxplot,
    plot_scenario_radar,
)
from visualization.config import save_plotly_figure, get_output_dir
from usage.scenario import *


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
    """运行场景对比实验"""
    
    if scenarios is None:
        if group_name is not None and group_name in SCENARIO_GROUPS:
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
            "q1": np.percentile(ttl_list, 25),
            "q3": np.percentile(ttl_list, 75),
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
    
    # 保存图表（Plotly 版本）
    if verbose:
        print("保存图表...")
    
    # 1. 场景对比柱状图
    fig = plot_scenario_comparison(results, show=False)
    save_plotly_figure(fig, "scenario_comparison", output_dir, size_type="default")
    
    # 2. 场景箱线图
    fig = plot_scenario_boxplot(results, show=False)
    save_plotly_figure(fig, "scenario_boxplot", output_dir, size_type="default")
    
    # 3. 雷达图
    fig = plot_scenario_radar(results, show=False)
    save_plotly_figure(fig, "scenario_radar", output_dir, size_type="square")
    
    if verbose:
        out_path = get_output_dir(output_dir)
        print(f"图表已保存到 {out_path}/ 目录")
    
    return results


def run_sensitivity_to_temperature(
    scenario=None,
    scenario_name="默认场景",
    T_amb_range=None,
    n_mc=50,
    verbose=True,
    output_dir="compare",
):
    """运行温度敏感性分析（Plotly 版本）"""
    import plotly.graph_objects as go
    from visualization.config import COLORS, LINE_WIDTHS, FONT_SIZES, FIGURE_SIZES
    
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
    
    fig = go.Figure()
    
    # 均值线
    fig.add_trace(go.Scatter(
        x=results["temperatures"],
        y=results["means"],
        mode='lines+markers',
        name='均值',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        marker=dict(size=8),
    ))
    
    # 误差带
    upper = [m + s for m, s in zip(results["means"], results["stds"])]
    lower = [m - s for m, s in zip(results["means"], results["stds"])]
    
    fig.add_trace(go.Scatter(
        x=results["temperatures"] + results["temperatures"][::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(41, 128, 185, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1σ',
        showlegend=True,
    ))
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(
            text=f"环境温度对续航时间的影响 - {scenario_name}",
            font=dict(size=FONT_SIZES["title"]),
        ),
        xaxis_title="环境温度 (°C)",
        yaxis_title="续航时间 TTL (小时)",
        width=width,
        height=height,
        legend=dict(font=dict(size=FONT_SIZES["legend"])),
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    save_plotly_figure(fig, "temperature_sensitivity", output_dir, size_type="default")
    
    return results


if __name__ == "__main__":
    run_comparison_experiment()
