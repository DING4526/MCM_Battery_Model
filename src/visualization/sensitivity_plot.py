# visualization/sensitivity_plot.py
# 敏感度分析可视化模块（Plotly 版本 - 单栏论文优化）
#
# 提供专业的敏感度可视化：
# - 柱状图
# - 龙卷风图
# - 蜘蛛图
# - 热力图
# - 综合分析

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from .config import (
    COLORS, DEFAULT_COLORS, PARAM_LABELS,
    FONT_SIZES, LINE_WIDTHS, FIGURE_SIZES,
    get_show_plots, save_plotly_figure,
    setup_style,
)

setup_style()


def _get_label(param):
    """获取参数的中文标签"""
    return PARAM_LABELS.get(param, param)


# =====================================================
# 敏感度柱状图
# =====================================================

def plot_sensitivity_bar(sens_results, filename=None, subdir="", ax=None, show=None,
                         save_path=None, normalized=True, sort=True):
    """
    绘制敏感度柱状图
    """
    # 排除内部字段
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    
    if normalized:
        values = [sens_results[p]["S_norm"] for p in params]
        ylabel = "归一化敏感度"
    else:
        values = [sens_results[p]["S"] / 3600 for p in params]
        ylabel = "敏感度 (小时)"
    
    labels = [_get_label(p) for p in params]
    
    # 排序
    if sort:
        sorted_indices = np.argsort(np.abs(values))[::-1]
        params = [params[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    # 颜色（正负不同）
    colors = [COLORS["danger"] if v < 0 else COLORS["success"] for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        textfont=dict(size=FONT_SIZES["annotation"]),
        hovertemplate='%{y}<br>敏感度: %{x:.4f}<extra></extra>',
    ))
    
    fig.add_vline(x=0, line_color=COLORS["primary"], line_width=LINE_WIDTHS["axis"])
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="参数敏感度分析", font=dict(size=FONT_SIZES["title"])),
        xaxis_title=ylabel,
        yaxis=dict(autorange="reversed"),
        width=width,
        height=height,
        margin=dict(l=100, r=60, t=50, b=45),
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
# 龙卷风图
# =====================================================

def plot_sensitivity_tornado(sens_results, baseline_ttl, ax=None, show=True,
                              save_path=None, sort=True):
    """
    绘制敏感度龙卷风图
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    baseline_h = baseline_ttl / 3600
    
    ttl_plus = [sens_results[p]["TTL+"] / 3600 for p in params]
    ttl_minus = [sens_results[p]["TTL-"] / 3600 for p in params]
    labels = [_get_label(p) for p in params]
    
    # 计算影响范围并排序
    ranges = [abs(ttl_plus[i] - ttl_minus[i]) for i in range(len(params))]
    
    if sort:
        sorted_indices = np.argsort(ranges)[::-1]
        params = [params[i] for i in sorted_indices]
        ttl_plus = [ttl_plus[i] for i in sorted_indices]
        ttl_minus = [ttl_minus[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    fig = go.Figure()
    
    # 负扰动
    fig.add_trace(go.Bar(
        y=labels,
        x=[t - baseline_h for t in ttl_minus],
        orientation='h',
        name='参数 -20%',
        marker=dict(color=COLORS["accent"]),
        base=baseline_h,
        hovertemplate='%{y}<br>TTL: %{x:.2f} h<extra></extra>',
    ))
    
    # 正扰动
    fig.add_trace(go.Bar(
        y=labels,
        x=[t - baseline_h for t in ttl_plus],
        orientation='h',
        name='参数 +20%',
        marker=dict(color=COLORS["warning"]),
        base=baseline_h,
        hovertemplate='%{y}<br>TTL: %{x:.2f} h<extra></extra>',
    ))
    
    # 基准线
    fig.add_vline(x=baseline_h, line_dash="dash", line_color=COLORS["primary"],
                  line_width=LINE_WIDTHS["main"],
                  annotation_text=f"基准: {baseline_h:.2f} h",
                  annotation_position="top",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="参数敏感度龙卷风图", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="续航时间 TTL (小时)",
        yaxis=dict(autorange="reversed"),
        barmode='overlay',
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZES["legend"]),
        ),
        margin=dict(l=100, r=20, t=50, b=70),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="default")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 蜘蛛图（雷达图）
# =====================================================

def plot_sensitivity_spider(sens_results, ax=None, show=True, save_path=None):
    """
    绘制敏感度蜘蛛图（雷达图）
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    values = [abs(sens_results[p]["S_norm"]) for p in params]
    labels = [_get_label(p) for p in params]
    
    # 闭合图形
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(41, 128, 185, 0.25)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        marker=dict(size=6, color=COLORS["accent"]),
        hovertemplate='%{theta}<br>|S|: %{r:.4f}<extra></extra>',
    ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="参数敏感度雷达图", font=dict(size=FONT_SIZES["title"])),
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickfont=dict(size=FONT_SIZES["axis_tick"]),
            ),
            angularaxis=dict(
                tickfont=dict(size=FONT_SIZES["axis_tick"]),
            ),
        ),
        width=width,
        height=height,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="square")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 敏感度热力图
# =====================================================

def plot_sensitivity_heatmap(sens_results, ax=None, show=True, save_path=None):
    """
    绘制敏感度热力图
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    labels = [_get_label(p) for p in params]
    
    # 数据矩阵
    data = np.zeros((len(params), 3))
    for i, p in enumerate(params):
        data[i, 0] = sens_results[p]["TTL-"] / 3600
        data[i, 1] = (sens_results[p]["TTL+"] + sens_results[p]["TTL-"]) / 2 / 3600
        data[i, 2] = sens_results[p]["TTL+"] / 3600
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=data,
        x=['参数 -20%', '基准', '参数 +20%'],
        y=labels,
        colorscale='RdYlGn',
        text=[[f'{v:.2f}' for v in row] for row in data],
        texttemplate='%{text}',
        textfont=dict(size=FONT_SIZES["annotation"]),
        hovertemplate='%{y}<br>%{x}: %{z:.3f} h<extra></extra>',
        colorbar=dict(
            title=dict(text='TTL (h)', font=dict(size=FONT_SIZES["axis_tick"])),
            tickfont=dict(size=FONT_SIZES["axis_tick"]),
        ),
    ))
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="参数敏感度热力图", font=dict(size=FONT_SIZES["title"])),
        width=width,
        height=height,
        margin=dict(l=100, r=80, t=50, b=45),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="default")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 敏感度综合分析
# =====================================================

def plot_sensitivity_comprehensive(sens_results, baseline_ttl, filename=None, subdir="",
                                    save_path=None, show=None):
    """
    绘制敏感度分析综合图表
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("归一化敏感度柱状图", "敏感度龙卷风图", "敏感度雷达图", "分析洞察"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "polar"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. 柱状图
    values = [sens_results[p]["S_norm"] for p in params]
    labels = [_get_label(p) for p in params]
    sorted_indices = np.argsort(np.abs(values))[::-1]
    sorted_values = [values[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    colors = [COLORS["danger"] if v < 0 else COLORS["success"] for v in sorted_values]
    
    fig.add_trace(go.Bar(
        y=sorted_labels, x=sorted_values, orientation='h',
        marker=dict(color=colors),
        showlegend=False,
    ), row=1, col=1)
    
    # 2. 龙卷风图
    baseline_h = baseline_ttl / 3600
    ttl_plus = [sens_results[p]["TTL+"] / 3600 for p in params]
    ttl_minus = [sens_results[p]["TTL-"] / 3600 for p in params]
    
    ranges = [abs(ttl_plus[i] - ttl_minus[i]) for i in range(len(params))]
    sorted_idx2 = np.argsort(ranges)[::-1]
    
    fig.add_trace(go.Bar(
        y=[labels[i] for i in sorted_idx2],
        x=[ttl_minus[i] - baseline_h for i in sorted_idx2],
        orientation='h', name='-20%',
        marker=dict(color=COLORS["accent"]),
        base=baseline_h,
        showlegend=True,
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        y=[labels[i] for i in sorted_idx2],
        x=[ttl_plus[i] - baseline_h for i in sorted_idx2],
        orientation='h', name='+20%',
        marker=dict(color=COLORS["warning"]),
        base=baseline_h,
        showlegend=True,
    ), row=1, col=2)
    
    # 3. 雷达图
    radar_values = [abs(sens_results[p]["S_norm"]) for p in params]
    radar_values_closed = radar_values + [radar_values[0]]
    labels_closed = labels + [labels[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=radar_values_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(41, 128, 185, 0.25)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=2, col=1)
    
    # 4. 洞察表格
    s_norms = {p: sens_results[p]["S_norm"] for p in params}
    most_sensitive = max(params, key=lambda p: abs(s_norms[p]))
    least_sensitive = min(params, key=lambda p: abs(s_norms[p]))
    
    positive_sens = [_get_label(p) for p in params if s_norms[p] > 0]
    negative_sens = [_get_label(p) for p in params if s_norms[p] < 0]
    
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>分析项</b>", "<b>结果</b>"],
            fill_color=COLORS["accent"],
            font=dict(color='white', size=FONT_SIZES["axis_tick"]),
            align='center', height=24,
        ),
        cells=dict(
            values=[
                ["基准 TTL", "最敏感参数", "最不敏感参数", "负敏感度参数", "正敏感度参数"],
                [f"{baseline_h:.2f} h",
                 f"{_get_label(most_sensitive)} ({s_norms[most_sensitive]:.4f})",
                 f"{_get_label(least_sensitive)} ({s_norms[least_sensitive]:.4f})",
                 ", ".join(negative_sens) if negative_sens else "无",
                 ", ".join(positive_sens) if positive_sens else "无"]
            ],
            fill_color=['rgba(236, 240, 241, 0.8)', 'white'],
            font=dict(size=FONT_SIZES["annotation"]),
            align=['left', 'left'], height=22,
        )
    ), row=2, col=2)
    
    # 布局
    fig.update_xaxes(title_text="归一化敏感度", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="TTL (h)", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    width, height = FIGURE_SIZES["composite"]
    fig.update_layout(
        title=dict(text="参数敏感度分析综合报告", font=dict(size=FONT_SIZES["title"])),
        width=width + 100,
        height=height,
        barmode='overlay',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZES["legend"]),
        ),
        margin=dict(l=80, r=20, t=60, b=70),
    )
    
    path = filename or save_path
    if path:
        save_plotly_figure(fig, path, subdir, size_type="composite")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig
