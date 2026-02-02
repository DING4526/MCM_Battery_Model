# visualization/distribution.py
# 分布可视化模块（Plotly 版本 - 单栏论文优化）
#
# 提供专业的分布可视化：
# - 直方图
# - 箱线图
# - 小提琴图
# - 核密度估计
# - 综合统计摘要

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats

from .config import (
    COLORS, DEFAULT_COLORS,
    FONT_SIZES, LINE_WIDTHS, FIGURE_SIZES,
    to_hours, get_show_plots, save_plotly_figure,
    setup_style,
)

setup_style()


# =====================================================
# TTL 分布直方图
# =====================================================

def plot_ttl_distribution(ttl_list, filename=None, subdir="", ax=None, show=None, save_path=None, bins=20):
    """
    绘制 TTL 分布直方图
    """
    ttl_h = to_hours(ttl_list)
    
    fig = go.Figure()
    
    # 直方图
    fig.add_trace(go.Histogram(
        x=ttl_h,
        nbinsx=bins,
        name='TTL 分布',
        marker=dict(
            color=COLORS["accent"],
            line=dict(color='white', width=1),
        ),
        opacity=0.75,
        hovertemplate='TTL: %{x:.2f} h<br>频数: %{y}<extra></extra>'
    ))
    
    # KDE 曲线
    kde = stats.gaussian_kde(ttl_h)
    x_range = np.linspace(min(ttl_h) - 0.2, max(ttl_h) + 0.2, 150)
    kde_values = kde(x_range)
    
    hist_counts, _ = np.histogram(ttl_h, bins=bins)
    bin_width = (max(ttl_h) - min(ttl_h)) / bins
    scale_factor = max(hist_counts) / max(kde_values) if max(kde_values) > 0 else 1
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_values * scale_factor,
        mode='lines',
        name='核密度估计',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        hoverinfo='skip',
    ))
    
    # 统计线
    mean_val = np.mean(ttl_h)
    median_val = np.median(ttl_h)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS["warning"],
                  line_width=LINE_WIDTHS["secondary"],
                  annotation_text=f"均值: {mean_val:.2f} h",
                  annotation_position="top",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    fig.add_vline(x=median_val, line_dash="dot", line_color=COLORS["success"],
                  line_width=LINE_WIDTHS["secondary"],
                  annotation_text=f"中位数: {median_val:.2f} h",
                  annotation_position="top left",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="Monte Carlo TTL 分布", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="续航时间 TTL (小时)",
        yaxis_title="频数",
        width=width,
        height=height,
        bargap=0.05,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    # 统计信息
    std_val = np.std(ttl_h)
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>统计摘要</b><br>n={len(ttl_h)}<br>μ={mean_val:.3f} h<br>σ={std_val:.3f} h",
        showarrow=False,
        font=dict(size=FONT_SIZES["annotation"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        borderpad=4,
        align="left",
        xanchor="right", yanchor="top",
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
# TTL 箱线图
# =====================================================

def plot_ttl_boxplot(ttl_list, ax=None, show=True, save_path=None, label="TTL"):
    """
    绘制 TTL 箱线图
    """
    ttl_h = to_hours(ttl_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=ttl_h,
        name=label,
        marker=dict(color=COLORS["accent"], outliercolor=COLORS["secondary"]),
        boxmean='sd',
        fillcolor='rgba(41, 128, 185, 0.4)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        hovertemplate='%{y:.2f} h<extra></extra>',
    ))
    
    # 散点（抖动）
    jitter = np.random.normal(0, 0.03, size=len(ttl_h))
    fig.add_trace(go.Scatter(
        x=jitter,
        y=ttl_h,
        mode='markers',
        name='数据点',
        marker=dict(color=COLORS["secondary"], size=4, opacity=0.4),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # 均值
    mean_val = np.mean(ttl_h)
    fig.add_trace(go.Scatter(
        x=[0], y=[mean_val],
        mode='markers',
        name=f'均值: {mean_val:.2f} h',
        marker=dict(color=COLORS["success"], size=10, symbol='diamond',
                   line=dict(color='white', width=1.5)),
    ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="TTL 分布箱线图", font=dict(size=FONT_SIZES["title"])),
        yaxis_title="续航时间 TTL (小时)",
        xaxis=dict(showticklabels=False, zeroline=False),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="square")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# TTL 小提琴图
# =====================================================

def plot_ttl_violin(ttl_list, ax=None, show=True, save_path=None, label="TTL"):
    """
    绘制 TTL 小提琴图
    """
    ttl_h = to_hours(ttl_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        y=ttl_h,
        name=label,
        box_visible=True,
        meanline_visible=True,
        fillcolor='rgba(41, 128, 185, 0.4)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        points='all',
        pointpos=-0.5,
        jitter=0.3,
        marker=dict(color=COLORS["secondary"], size=3, opacity=0.4),
        hovertemplate='%{y:.2f} h<extra></extra>',
    ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="TTL 分布小提琴图", font=dict(size=FONT_SIZES["title"])),
        yaxis_title="续航时间 TTL (小时)",
        xaxis=dict(showticklabels=False),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    # 统计信息
    mean_val = np.mean(ttl_h)
    std_val = np.std(ttl_h)
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>分布特征</b><br>μ={mean_val:.3f} h<br>σ={std_val:.3f} h<br>偏度={stats.skew(ttl_h):.3f}",
        showarrow=False,
        font=dict(size=FONT_SIZES["annotation"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        borderpad=4,
        align="left",
        xanchor="right", yanchor="top",
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="square")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# TTL 核密度估计图
# =====================================================

def plot_ttl_kde(ttl_list, ax=None, show=True, save_path=None, fill=True):
    """
    绘制 TTL 核密度估计图
    """
    ttl_h = to_hours(ttl_list)
    
    kde = stats.gaussian_kde(ttl_h)
    x_range = np.linspace(min(ttl_h) - 0.5, max(ttl_h) + 0.5, 200)
    density = kde(x_range)
    
    fig = go.Figure()
    
    fill_mode = 'tozeroy' if fill else None
    fig.add_trace(go.Scatter(
        x=x_range,
        y=density,
        mode='lines',
        name='核密度估计',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        fill=fill_mode,
        fillcolor='rgba(41, 128, 185, 0.25)',
        hovertemplate='TTL: %{x:.2f} h<br>密度: %{y:.4f}<extra></extra>',
    ))
    
    # Rug plot
    fig.add_trace(go.Scatter(
        x=ttl_h,
        y=[-0.01 * max(density)] * len(ttl_h),
        mode='markers',
        name='数据点',
        marker=dict(color=COLORS["secondary"], size=6, symbol='line-ns-open', opacity=0.5),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # 统计线
    mean_val = np.mean(ttl_h)
    std_val = np.std(ttl_h)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS["warning"],
                  line_width=LINE_WIDTHS["secondary"],
                  annotation_text=f"μ={mean_val:.2f} h",
                  annotation_position="top",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    # ±1σ 区域
    fig.add_vrect(
        x0=mean_val - std_val, x1=mean_val + std_val,
        fillcolor="rgba(243, 156, 18, 0.08)",
        line_width=0,
    )
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="TTL 核密度估计分布", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="续航时间 TTL (小时)",
        yaxis_title="概率密度",
        yaxis=dict(rangemode='tozero'),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="default")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# TTL 综合统计摘要
# =====================================================

def plot_ttl_statistical_summary(ttl_list, filename=None, subdir="", save_path=None, show=None):
    """
    绘制 TTL 综合统计摘要图
    """
    ttl_h = to_hours(ttl_list)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("分布直方图 + KDE", "箱线图", "Q-Q 图", "统计报告"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. 直方图 + KDE
    fig.add_trace(go.Histogram(
        x=ttl_h, nbinsx=20,
        marker=dict(color=COLORS["accent"], line=dict(color='white', width=1)),
        opacity=0.7, histnorm='probability density',
        showlegend=False,
    ), row=1, col=1)
    
    kde = stats.gaussian_kde(ttl_h)
    x_range = np.linspace(min(ttl_h) - 0.3, max(ttl_h) + 0.3, 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=kde(x_range),
        mode='lines', line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=1, col=1)
    
    # 2. 箱线图
    fig.add_trace(go.Box(
        y=ttl_h, name='TTL', boxmean='sd',
        fillcolor='rgba(41, 128, 185, 0.4)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=1, col=2)
    
    jitter = np.random.normal(0, 0.02, size=len(ttl_h))
    fig.add_trace(go.Scatter(
        x=jitter, y=ttl_h, mode='markers',
        marker=dict(color=COLORS["secondary"], size=3, opacity=0.4),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=2)
    
    # 3. Q-Q 图
    sorted_data = np.sort(ttl_h)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(ttl_h)))
    
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles, y=sorted_data,
        mode='markers',
        marker=dict(color=COLORS["accent"], size=5, opacity=0.6),
        showlegend=False,
    ), row=2, col=1)
    
    mean_val = np.mean(ttl_h)
    std_val = np.std(ttl_h)
    ref_x = np.array([min(theoretical_quantiles), max(theoretical_quantiles)])
    ref_y = mean_val + std_val * ref_x
    fig.add_trace(go.Scatter(
        x=ref_x, y=ref_y, mode='lines',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["secondary"], dash='dash'),
        showlegend=False,
    ), row=2, col=1)
    
    # 4. 统计表格
    n_samples = len(ttl_h)
    min_val = np.min(ttl_h)
    max_val = np.max(ttl_h)
    median_val = np.median(ttl_h)
    q1 = np.percentile(ttl_h, 25)
    q3 = np.percentile(ttl_h, 75)
    iqr = q3 - q1
    skewness = stats.skew(ttl_h)
    kurtosis = stats.kurtosis(ttl_h)
    
    if n_samples <= 5000:
        _, p_value = stats.shapiro(ttl_h)
    else:
        _, p_value = stats.normaltest(ttl_h)
    
    ci_low = mean_val - 1.96 * std_val / np.sqrt(n_samples)
    ci_high = mean_val + 1.96 * std_val / np.sqrt(n_samples)
    
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>指标</b>", "<b>数值</b>"],
            fill_color=COLORS["accent"],
            font=dict(color='white', size=FONT_SIZES["axis_tick"]),
            align='center', height=24,
        ),
        cells=dict(
            values=[
                ["样本数", "均值", "中位数", "标准差", "CV(%)", "最小值", "最大值",
                 "Q1", "Q3", "IQR", "偏度", "峰度", "正态性 p", "95% CI"],
                [f"{n_samples}", f"{mean_val:.4f}", f"{median_val:.4f}",
                 f"{std_val:.4f}", f"{std_val/mean_val*100:.2f}",
                 f"{min_val:.4f}", f"{max_val:.4f}",
                 f"{q1:.4f}", f"{q3:.4f}", f"{iqr:.4f}",
                 f"{skewness:.4f}", f"{kurtosis:.4f}",
                 f"{p_value:.4f}", f"[{ci_low:.3f}, {ci_high:.3f}]"]
            ],
            fill_color=['rgba(236, 240, 241, 0.8)', 'white'],
            font=dict(size=FONT_SIZES["annotation"]),
            align=['left', 'center'], height=20,
        )
    ), row=2, col=2)
    
    # 布局
    fig.update_xaxes(title_text="TTL (h)", row=1, col=1)
    fig.update_yaxes(title_text="概率密度", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="TTL (h)", row=1, col=2)
    fig.update_xaxes(title_text="理论分位数", row=2, col=1)
    fig.update_yaxes(title_text="样本分位数", row=2, col=1)
    
    width, height = FIGURE_SIZES["tall"]
    fig.update_layout(
        title=dict(
            text=f"Monte Carlo TTL 统计分析 (n={n_samples})",
            font=dict(size=FONT_SIZES["title"]),
        ),
        width=width + 100,
        height=height + 150,
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=45),
    )
    
    path = filename or save_path
    if path:
        save_plotly_figure(fig, path, subdir, size_type="tall")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig
