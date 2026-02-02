# visualization/distribution.py
# Distribution Visualization Module (Plotly - Single Column Paper Optimized)
#
# Professional distribution visualizations:
# - Histogram
# - Box Plot
# - Violin Plot
# - Kernel Density Estimation
# - Comprehensive Statistical Summary

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
# TTL Distribution Histogram
# =====================================================

def plot_ttl_distribution(ttl_list, filename=None, subdir="", ax=None, show=None, save_path=None, bins=20):
    """
    Plot TTL distribution histogram
    """
    ttl_h = to_hours(ttl_list)
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=ttl_h,
        nbinsx=bins,
        name='TTL Distribution',
        marker=dict(
            color=COLORS["accent"],
            line=dict(color='white', width=1),
        ),
        opacity=0.75,
        hovertemplate='TTL: %{x:.2f} h<br>Count: %{y}<extra></extra>'
    ))
    
    # KDE curve
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
        name='KDE',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        hoverinfo='skip',
    ))
    
    # Statistics lines
    mean_val = np.mean(ttl_h)
    median_val = np.median(ttl_h)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS["warning"],
                  line_width=LINE_WIDTHS["secondary"],
                  annotation_text=f"Mean: {mean_val:.2f} h",
                  annotation_position="top",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    fig.add_vline(x=median_val, line_dash="dot", line_color=COLORS["success"],
                  line_width=LINE_WIDTHS["secondary"],
                  annotation_text=f"Median: {median_val:.2f} h",
                  annotation_position="top left",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="Monte Carlo TTL Distribution", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time-to-Live TTL (hours)",
        yaxis_title="Count",
        width=width,
        height=height,
        bargap=0.05,
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    # Statistics info
    std_val = np.std(ttl_h)
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>Statistics</b><br>n={len(ttl_h)}<br>μ={mean_val:.3f} h<br>σ={std_val:.3f} h",
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
# TTL Box Plot
# =====================================================

def plot_ttl_boxplot(ttl_list, ax=None, show=True, save_path=None, label="TTL"):
    """
    Plot TTL box plot
    """
    ttl_h = to_hours(ttl_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Box(
        y=ttl_h,
        name=label,
        marker=dict(color=COLORS["accent"], outliercolor=COLORS["secondary"]),
        boxmean='sd',
        fillcolor='rgba(8, 145, 178, 0.4)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        hovertemplate='%{y:.2f} h<extra></extra>',
    ))
    
    # Scatter (jitter)
    jitter = np.random.normal(0, 0.03, size=len(ttl_h))
    fig.add_trace(go.Scatter(
        x=jitter,
        y=ttl_h,
        mode='markers',
        name='Data Points',
        marker=dict(color=COLORS["secondary"], size=4, opacity=0.4),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Mean
    mean_val = np.mean(ttl_h)
    fig.add_trace(go.Scatter(
        x=[0], y=[mean_val],
        mode='markers',
        name=f'Mean: {mean_val:.2f} h',
        marker=dict(color=COLORS["success"], size=10, symbol='diamond',
                   line=dict(color='white', width=1.5)),
    ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="TTL Distribution Box Plot", font=dict(size=FONT_SIZES["title"])),
        yaxis_title="Time-to-Live TTL (hours)",
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
# TTL Violin Plot
# =====================================================

def plot_ttl_violin(ttl_list, ax=None, show=True, save_path=None, label="TTL"):
    """
    Plot TTL violin plot
    """
    ttl_h = to_hours(ttl_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        y=ttl_h,
        name=label,
        box_visible=True,
        meanline_visible=True,
        fillcolor='rgba(8, 145, 178, 0.4)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        points='all',
        pointpos=-0.5,
        jitter=0.3,
        marker=dict(color=COLORS["secondary"], size=3, opacity=0.4),
        hovertemplate='%{y:.2f} h<extra></extra>',
    ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="TTL Distribution Violin Plot", font=dict(size=FONT_SIZES["title"])),
        yaxis_title="Time-to-Live TTL (hours)",
        xaxis=dict(showticklabels=False),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    # Statistics info
    mean_val = np.mean(ttl_h)
    std_val = np.std(ttl_h)
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>Distribution</b><br>μ={mean_val:.3f} h<br>σ={std_val:.3f} h<br>Skew={stats.skew(ttl_h):.3f}",
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
# TTL Kernel Density Estimation
# =====================================================

def plot_ttl_kde(ttl_list, ax=None, show=True, save_path=None, fill=True):
    """
    Plot TTL kernel density estimation
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
        name='KDE',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        fill=fill_mode,
        fillcolor='rgba(8, 145, 178, 0.25)',
        hovertemplate='TTL: %{x:.2f} h<br>Density: %{y:.4f}<extra></extra>',
    ))
    
    # Rug plot
    fig.add_trace(go.Scatter(
        x=ttl_h,
        y=[-0.01 * max(density)] * len(ttl_h),
        mode='markers',
        name='Data Points',
        marker=dict(color=COLORS["secondary"], size=6, symbol='line-ns-open', opacity=0.5),
        showlegend=False,
        hoverinfo='skip',
    ))
    
    # Statistics lines
    mean_val = np.mean(ttl_h)
    std_val = np.std(ttl_h)
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS["warning"],
                  line_width=LINE_WIDTHS["secondary"],
                  annotation_text=f"μ={mean_val:.2f} h",
                  annotation_position="top",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    # ±1σ region
    fig.add_vrect(
        x0=mean_val - std_val, x1=mean_val + std_val,
        fillcolor="rgba(245, 158, 11, 0.08)",
        line_width=0,
    )
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="TTL Kernel Density Estimation", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time-to-Live TTL (hours)",
        yaxis_title="Probability Density",
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
# TTL Comprehensive Statistical Summary
# =====================================================

def plot_ttl_statistical_summary(ttl_list, filename=None, subdir="", save_path=None, show=None):
    """
    Plot TTL comprehensive statistical summary
    """
    ttl_h = to_hours(ttl_list)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Histogram + KDE", "Box Plot", "Q-Q Plot", "Statistics Report"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. Histogram + KDE
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
    
    # 2. Box plot
    fig.add_trace(go.Box(
        y=ttl_h, name='TTL', boxmean='sd',
        fillcolor='rgba(8, 145, 178, 0.4)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=1, col=2)
    
    jitter = np.random.normal(0, 0.02, size=len(ttl_h))
    fig.add_trace(go.Scatter(
        x=jitter, y=ttl_h, mode='markers',
        marker=dict(color=COLORS["secondary"], size=3, opacity=0.4),
        showlegend=False, hoverinfo='skip',
    ), row=1, col=2)
    
    # 3. Q-Q plot
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
    
    # 4. Statistics table
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
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color=COLORS["accent"],
            font=dict(color='white', size=FONT_SIZES["axis_tick"]),
            align='center', height=24,
        ),
        cells=dict(
            values=[
                ["Samples", "Mean", "Median", "Std Dev", "CV(%)", "Min", "Max",
                 "Q1", "Q3", "IQR", "Skewness", "Kurtosis", "Normality p", "95% CI"],
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
    
    # Layout
    fig.update_xaxes(title_text="TTL (h)", row=1, col=1)
    fig.update_yaxes(title_text="Prob. Density", row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="TTL (h)", row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    
    width, height = FIGURE_SIZES["tall"]
    fig.update_layout(
        title=dict(
            text=f"Monte Carlo TTL Statistical Analysis (n={n_samples})",
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
