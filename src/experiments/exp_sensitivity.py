# experiments/exp_sensitivity.py
# 敏感度分析实验模块（Plotly 版本）

import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_monte_carlo
from usage.state import USAGE_STATES
from visualization.sensitivity_plot import (
    plot_sensitivity_bar,
    plot_sensitivity_tornado,
    plot_sensitivity_spider,
    plot_sensitivity_heatmap,
)
from visualization.config import save_plotly_figure, get_output_dir
from usage.scenario import *


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
    """运行敏感度分析实验"""
    
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
    
    original_states = copy.deepcopy(USAGE_STATES)
    
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
        
        _perturb_usage(p, 1 + eps)
        ttl_plus = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc
        
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))
        
        _perturb_usage(p, 1 - eps)
        ttl_minus = sum(run_monte_carlo(scenario, n_samples=n_mc)) / n_mc
        
        USAGE_STATES.clear()
        USAGE_STATES.update(copy.deepcopy(original_states))
        
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
    
    # 保存图表（Plotly 版本）
    if verbose:
        print("保存图表...")
    
    # 1. 敏感度柱状图
    fig = plot_sensitivity_bar(results, show=False)
    save_plotly_figure(fig, "sensitivity_bar", output_dir, size_type="default")
    
    # 2. 龙卷风图
    fig = plot_sensitivity_tornado(results, ttl_base, show=False)
    save_plotly_figure(fig, "sensitivity_tornado", output_dir, size_type="default")
    
    # 3. 蜘蛛图
    fig = plot_sensitivity_spider(results, show=False)
    save_plotly_figure(fig, "sensitivity_spider", output_dir, size_type="square")
    
    if verbose:
        out_path = get_output_dir(output_dir)
        print(f"图表已保存到 {out_path}/ 目录")
    
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
    """运行多扰动幅度敏感度分析（Plotly 版本）"""
    import plotly.graph_objects as go
    from visualization.config import COLORS, LINE_WIDTHS, FONT_SIZES, FIGURE_SIZES
    
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
    
    fig = go.Figure()
    
    colors = [COLORS["accent"], COLORS["secondary"], COLORS["warning"]]
    
    for i, p in enumerate(param_list):
        fig.add_trace(go.Scatter(
            x=multi_results[p]["eps"],
            y=multi_results[p]["S_norm"],
            mode='lines+markers',
            name=PARAM_DESCRIPTIONS.get(p, p),
            line=dict(color=colors[i % len(colors)], width=LINE_WIDTHS["main"]),
            marker=dict(size=6),
        ))
    
    fig.add_hline(y=0, line_color=COLORS["primary"], line_width=1)
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="敏感度与扰动幅度关系", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="扰动幅度 (%)",
        yaxis_title="归一化敏感度",
        width=width,
        height=height,
        legend=dict(font=dict(size=FONT_SIZES["legend"])),
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    save_plotly_figure(fig, "multi_eps", output_dir, size_type="default")
    
    return multi_results


if __name__ == "__main__":
    run_sensitivity_experiment()
