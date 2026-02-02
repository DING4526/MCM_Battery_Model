# experiments/exp_monte_carlo.py
# Monte Carlo 仿真实验模块（Plotly 版本）

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_monte_carlo
from visualization.distribution import (
    plot_ttl_distribution,
    plot_ttl_boxplot,
    plot_ttl_violin,
    plot_ttl_kde,
)
from visualization.config import save_plotly_figure, get_output_dir
from usage.scenario import *


def run_monte_carlo_experiment(
    scenario=None,
    scenario_name="默认场景",
    n_samples=100,
    base_seed=0,
    dt=1.0,
    T_amb=298.15,
    verbose=True,
    output_dir="monte_carlo",
):
    """
    运行 Monte Carlo 仿真实验
    """
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常"
    
    if verbose:
        print("=" * 60)
        print("🎲 Monte Carlo 仿真实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"仿真次数: {n_samples}")
        print(f"环境温度: {T_amb - 273.15:.1f} °C")
        print("-" * 60)
        print("正在运行仿真...")
    
    # 运行 Monte Carlo 仿真
    ttl_list = run_monte_carlo(
        scenario=scenario,
        n_samples=n_samples,
        base_seed=base_seed,
        dt=dt,
        T_amb=T_amb,
    )
    
    # 计算统计量
    results = {
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
        print("-" * 60)
        print("✅ 仿真完成！")
        print(f"   平均 TTL: {results['mean']/3600:.2f} 小时")
        print(f"   标准差: {results['std']/3600:.3f} 小时")
        print(f"   范围: [{results['min']/3600:.2f}, {results['max']/3600:.2f}] 小时")
        print("=" * 60)
    
    # 保存图表（Plotly 版本）
    if verbose:
        print("保存图表...")
    
    # 1. TTL 分布直方图
    fig = plot_ttl_distribution(ttl_list, show=False)
    save_plotly_figure(fig, "ttl_histogram", output_dir, size_type="default")
    
    # 2. 箱线图
    fig = plot_ttl_boxplot(ttl_list, show=False)
    save_plotly_figure(fig, "ttl_boxplot", output_dir, size_type="square")
    
    # 3. 小提琴图
    fig = plot_ttl_violin(ttl_list, show=False)
    save_plotly_figure(fig, "ttl_violin", output_dir, size_type="square")
    
    # 4. 核密度估计
    fig = plot_ttl_kde(ttl_list, show=False)
    save_plotly_figure(fig, "ttl_kde", output_dir, size_type="default")
    
    if verbose:
        out_path = get_output_dir(output_dir)
        print(f"图表已保存到 {out_path}/ 目录")
    
    return results


def run_convergence_analysis(
    scenario=None,
    scenario_name="默认场景",
    max_samples=500,
    step=50,
    base_seed=0,
    verbose=True,
    output_dir="monte_carlo",
):
    """运行收敛性分析（Plotly 版本）"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from visualization.config import COLORS, LINE_WIDTHS, FONT_SIZES, FIGURE_SIZES
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
    
    if verbose:
        print("📈 Monte Carlo 收敛性分析")
        print(f"最大样本数: {max_samples}")
    
    sample_sizes = list(range(step, max_samples + 1, step))
    means = []
    stds = []
    
    all_ttl = run_monte_carlo(scenario=scenario, n_samples=max_samples, base_seed=base_seed)
    
    for n in sample_sizes:
        ttl_subset = all_ttl[:n]
        means.append(np.mean(ttl_subset) / 3600)
        stds.append(np.std(ttl_subset) / 3600)
    
    # 绘制收敛图
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("均值收敛性", "标准差稳定性"),
        horizontal_spacing=0.12,
    )
    
    fig.add_trace(go.Scatter(
        x=sample_sizes, y=means,
        mode='lines+markers',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        marker=dict(size=5),
        name='均值',
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=sample_sizes, y=stds,
        mode='lines+markers',
        line=dict(color=COLORS["success"], width=LINE_WIDTHS["main"]),
        marker=dict(size=5),
        name='标准差',
    ), row=1, col=2)
    
    fig.update_xaxes(title_text="样本数", row=1, col=1)
    fig.update_yaxes(title_text="TTL 均值 (小时)", row=1, col=1)
    fig.update_xaxes(title_text="样本数", row=1, col=2)
    fig.update_yaxes(title_text="TTL 标准差 (小时)", row=1, col=2)
    
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="Monte Carlo 收敛性分析", font=dict(size=FONT_SIZES["title"])),
        width=width + 100,
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=45),
    )
    
    save_plotly_figure(fig, "convergence", output_dir, size_type="wide")
    
    return {"sample_sizes": sample_sizes, "means": means, "stds": stds}


if __name__ == "__main__":
    run_monte_carlo_experiment()
