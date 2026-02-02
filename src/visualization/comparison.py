# visualization/comparison.py
# 场景对比可视化模块（Plotly 版本 - 单栏论文优化）
#
# 提供专业的场景对比可视化：
# - 场景对比柱状图
# - 多场景箱线图
# - 雷达图
# - 多场景时间线对比

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from .config import (
    COLORS, SCENARIO_COLORS, DEFAULT_COLORS,
    FONT_SIZES, LINE_WIDTHS, FIGURE_SIZES,
    to_hours, get_show_plots, save_plotly_figure,
    hex_to_rgba,
    setup_style,
)

setup_style()


def _get_color(scenario_name, index):
    """获取场景颜色"""
    if scenario_name in SCENARIO_COLORS:
        return SCENARIO_COLORS[scenario_name]
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]


# =====================================================
# 场景对比柱状图
# =====================================================

def plot_scenario_comparison(comparison_results, filename=None, subdir="", ax=None,
                              show=None, save_path=None, error_bars=True):
    """
    绘制场景对比柱状图
    """
    scenarios = list(comparison_results.keys())
    values = [comparison_results[s]["mean"] / 3600 for s in scenarios]
    errors = [comparison_results[s]["std"] / 3600 for s in scenarios] if error_bars else None
    colors = [_get_color(s, i) for i, s in enumerate(scenarios)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=scenarios,
        y=values,
        marker=dict(color=colors, line=dict(color='white', width=1)),
        error_y=dict(type='data', array=errors, visible=error_bars) if errors else None,
        text=[f'{v:.2f} h' for v in values],
        textposition='outside',
        textfont=dict(size=FONT_SIZES["annotation"]),
        hovertemplate='%{x}<br>TTL: %{y:.2f} h<extra></extra>',
    ))
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="使用场景续航时间对比", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="使用场景",
        yaxis_title="平均续航时间 (小时)",
        xaxis=dict(tickangle=-15),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=50, b=70),
    )
    
    path = filename or save_path
    if path:
        save_plotly_figure(fig, path, subdir, size_type="default")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig


# =====================================================
# 多场景箱线图
# =====================================================

def plot_scenario_boxplot(comparison_results, ax=None, show=True, save_path=None):
    """
    绘制多场景箱线图对比
    """
    scenarios = list(comparison_results.keys())
    
    fig = go.Figure()
    
    for i, s in enumerate(scenarios):
        ttl_h = to_hours(comparison_results[s]["ttl_list"])
        color = _get_color(s, i)
        
        fig.add_trace(go.Box(
            y=ttl_h,
            name=s,
            marker=dict(color=color),
            boxmean='sd',
            fillcolor=hex_to_rgba(color, 0.4),
            line=dict(color=color, width=LINE_WIDTHS["main"]),
        ))
    
    # 均值标记
    means = [np.mean(to_hours(comparison_results[s]["ttl_list"])) for s in scenarios]
    fig.add_trace(go.Scatter(
        x=scenarios,
        y=means,
        mode='markers',
        name='均值',
        marker=dict(color=COLORS["danger"], size=8, symbol='diamond',
                   line=dict(color='white', width=1)),
    ))
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="使用场景 TTL 分布对比", font=dict(size=FONT_SIZES["title"])),
        yaxis_title="续航时间 TTL (小时)",
        xaxis=dict(tickangle=-15),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=70),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="default")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 场景雷达图
# =====================================================

def plot_scenario_radar(comparison_results, metrics=None, ax=None, show=True, save_path=None):
    """
    绘制场景对比雷达图
    """
    scenarios = list(comparison_results.keys())
    
    if metrics is None:
        metrics = ["mean", "std", "min", "max", "median"]
    
    metric_labels = {
        "mean": "均值", "std": "标准差", "min": "最小值",
        "max": "最大值", "median": "中位数"
    }
    
    # 准备数据
    data = {}
    for s in scenarios:
        ttl_h = to_hours(comparison_results[s]["ttl_list"])
        data[s] = {
            "mean": np.mean(ttl_h),
            "std": np.std(ttl_h),
            "min": np.min(ttl_h),
            "max": np.max(ttl_h),
            "median": np.median(ttl_h)
        }
    
    # 归一化
    normalized = {}
    for m in metrics:
        vals = [data[s][m] for s in scenarios]
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v if max_v != min_v else 1
        for s in scenarios:
            if s not in normalized:
                normalized[s] = []
            normalized[s].append((data[s][m] - min_v) / range_v)
    
    labels = [metric_labels.get(m, m) for m in metrics]
    labels_closed = labels + [labels[0]]
    
    fig = go.Figure()
    
    for i, s in enumerate(scenarios):
        values = normalized[s] + [normalized[s][0]]
        color = _get_color(s, i)
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels_closed,
            fill='toself',
            fillcolor=hex_to_rgba(color, 0.15),
            line=dict(color=color, width=LINE_WIDTHS["main"]),
            marker=dict(size=5, color=color),
            name=s,
        ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="场景多维度对比雷达图", font=dict(size=FONT_SIZES["title"])),
        polar=dict(
            radialaxis=dict(visible=True, tickfont=dict(size=FONT_SIZES["axis_tick"])),
            angularaxis=dict(tickfont=dict(size=FONT_SIZES["axis_tick"])),
        ),
        legend=dict(
            font=dict(size=FONT_SIZES["legend"]),
            x=1.1, y=0.9,
        ),
        width=width + 100,
        height=height,
        margin=dict(l=60, r=100, t=60, b=60),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="square")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 多场景时间线对比
# =====================================================

def plot_multi_scenario_timeline(results_dict, ax=None, show=True, save_path=None):
    """
    绘制多场景时间线对比
    """
    scenarios = list(results_dict.keys())
    
    fig = go.Figure()
    
    for i, (scenario, result) in enumerate(results_dict.items()):
        time_h = to_hours(result["time"])
        soc_percent = [s * 100 for s in result["SOC"]]
        color = _get_color(scenario, i)
        
        fig.add_trace(go.Scatter(
            x=time_h,
            y=soc_percent,
            mode='lines',
            name=scenario,
            line=dict(color=color, width=LINE_WIDTHS["main"]),
            hovertemplate=f'{scenario}<br>时间: %{{x:.2f}} h<br>SOC: %{{y:.1f}}%<extra></extra>'
        ))
    
    # 关键电量线
    fig.add_hline(y=20, line_dash="dash", line_color=COLORS["warning"],
                  line_width=1, opacity=0.5)
    fig.add_hline(y=5, line_dash="dash", line_color=COLORS["danger"],
                  line_width=1, opacity=0.5)
    
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="多场景 SOC 变化曲线对比", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="时间 (小时)",
        yaxis_title="电量 SOC (%)",
        yaxis=dict(range=[0, 105]),
        width=width,
        height=height + 50,
        legend=dict(
            font=dict(size=FONT_SIZES["legend"]),
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
        ),
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="wide")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 场景综合对比
# =====================================================

def plot_scenario_comprehensive_comparison(comparison_results, results_dict=None,
                                            filename=None, subdir="", save_path=None, show=None):
    """
    绘制场景对比综合图表
    """
    scenarios = list(comparison_results.keys())
    has_timeline = results_dict is not None
    
    if has_timeline:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("场景对比柱状图", "TTL 分布箱线图", "SOC 时间线对比", "统计分析"),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )
    else:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("场景对比柱状图", "TTL 分布箱线图", "统计分析", ""),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "table", "colspan": 2}, None],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )
    
    # 1. 柱状图
    values = [comparison_results[s]["mean"] / 3600 for s in scenarios]
    errors = [comparison_results[s]["std"] / 3600 for s in scenarios]
    colors = [_get_color(s, i) for i, s in enumerate(scenarios)]
    
    fig.add_trace(go.Bar(
        x=scenarios, y=values,
        marker=dict(color=colors),
        error_y=dict(type='data', array=errors),
        text=[f'{v:.2f}' for v in values],
        textposition='outside',
        showlegend=False,
    ), row=1, col=1)
    
    # 2. 箱线图
    for i, s in enumerate(scenarios):
        ttl_h = to_hours(comparison_results[s]["ttl_list"])
        color = _get_color(s, i)
        
        fig.add_trace(go.Box(
            y=ttl_h, name=s,
            marker=dict(color=color),
            boxmean='sd',
            showlegend=False,
        ), row=1, col=2)
    
    if has_timeline:
        # 3. 时间线对比
        for i, (scenario, result) in enumerate(results_dict.items()):
            time_h = to_hours(result["time"])
            soc_percent = [s * 100 for s in result["SOC"]]
            color = _get_color(scenario, i)
            
            fig.add_trace(go.Scatter(
                x=time_h, y=soc_percent,
                mode='lines', name=scenario,
                line=dict(color=color, width=LINE_WIDTHS["secondary"]),
                showlegend=True,
            ), row=2, col=1)
        
        table_row, table_col = 2, 2
    else:
        table_row, table_col = 2, 1
    
    # 统计表格
    baseline_mean = comparison_results[scenarios[0]]["mean"]
    
    table_scenarios = []
    table_means = []
    table_stds = []
    table_mins = []
    table_maxs = []
    table_relatives = []
    
    for s in scenarios:
        table_scenarios.append(s)
        table_means.append(f'{comparison_results[s]["mean"]/3600:.2f}')
        table_stds.append(f'{comparison_results[s]["std"]/3600:.2f}')
        table_mins.append(f'{comparison_results[s]["min"]/3600:.2f}')
        table_maxs.append(f'{comparison_results[s]["max"]/3600:.2f}')
        rel = (comparison_results[s]["mean"] / baseline_mean - 1) * 100
        table_relatives.append(f'{rel:+.1f}%')
    
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>场景</b>", "<b>均值(h)</b>", "<b>标准差</b>", "<b>最小</b>", "<b>最大</b>", "<b>相对基准</b>"],
            fill_color=COLORS["accent"],
            font=dict(color='white', size=FONT_SIZES["axis_tick"]),
            align='center', height=24,
        ),
        cells=dict(
            values=[table_scenarios, table_means, table_stds, table_mins, table_maxs, table_relatives],
            fill_color=['rgba(236, 240, 241, 0.8)'] + ['white'] * 5,
            font=dict(size=FONT_SIZES["annotation"]),
            align='center', height=22,
        )
    ), row=table_row, col=table_col)
    
    # 布局
    fig.update_xaxes(tickangle=-15, row=1, col=1)
    fig.update_yaxes(title_text="TTL (h)", row=1, col=1)
    fig.update_yaxes(title_text="TTL (h)", row=1, col=2)
    
    if has_timeline:
        fig.update_xaxes(title_text="时间 (h)", row=2, col=1)
        fig.update_yaxes(title_text="SOC (%)", range=[0, 105], row=2, col=1)
    
    width, height = FIGURE_SIZES["composite"]
    fig.update_layout(
        title=dict(text="使用场景对比分析报告", font=dict(size=FONT_SIZES["title"])),
        width=width + 100,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZES["legend"]),
        ),
        margin=dict(l=50, r=20, t=60, b=70),
    )
    
    path = filename or save_path
    if path:
        save_plotly_figure(fig, path, subdir, size_type="composite")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig
