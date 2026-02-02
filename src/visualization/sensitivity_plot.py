# visualization/sensitivity_plot.py
# Sensitivity Analysis Visualization Module (Plotly - Single Column Paper Optimized)
#
# Professional sensitivity visualizations:
# - Bar Chart
# - Tornado Chart
# - Spider Chart
# - Heatmap
# - Comprehensive Analysis

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
    """Get parameter label"""
    return PARAM_LABELS.get(param, param)


# =====================================================
# Sensitivity Bar Chart
# =====================================================

def plot_sensitivity_bar(sens_results, filename=None, subdir="", ax=None, show=None,
                         save_path=None, normalized=True, sort=True):
    """
    Plot sensitivity bar chart
    """
    # Exclude internal fields
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    
    if normalized:
        values = [sens_results[p]["S_norm"] for p in params]
        ylabel = "Normalized Sensitivity"
    else:
        values = [sens_results[p]["S"] / 3600 for p in params]
        ylabel = "Sensitivity (hours)"
    
    labels = [_get_label(p) for p in params]
    
    # Sort
    if sort:
        sorted_indices = np.argsort(np.abs(values))[::-1]
        params = [params[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    # Colors (positive/negative)
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
        hovertemplate='%{y}<br>Sensitivity: %{x:.4f}<extra></extra>',
    ))
    
    fig.add_vline(x=0, line_color=COLORS["primary"], line_width=LINE_WIDTHS["axis"])
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="Parameter Sensitivity Analysis", font=dict(size=FONT_SIZES["title"])),
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
# Tornado Chart
# =====================================================

def plot_sensitivity_tornado(sens_results, baseline_ttl, ax=None, show=True,
                              save_path=None, sort=True):
    """
    Plot sensitivity tornado chart
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    baseline_h = baseline_ttl / 3600
    
    ttl_plus = [sens_results[p]["TTL+"] / 3600 for p in params]
    ttl_minus = [sens_results[p]["TTL-"] / 3600 for p in params]
    labels = [_get_label(p) for p in params]
    
    # Calculate impact range and sort
    ranges = [abs(ttl_plus[i] - ttl_minus[i]) for i in range(len(params))]
    
    if sort:
        sorted_indices = np.argsort(ranges)[::-1]
        params = [params[i] for i in sorted_indices]
        ttl_plus = [ttl_plus[i] for i in sorted_indices]
        ttl_minus = [ttl_minus[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    fig = go.Figure()
    
    # Negative perturbation
    fig.add_trace(go.Bar(
        y=labels,
        x=[t - baseline_h for t in ttl_minus],
        orientation='h',
        name='Param -20%',
        marker=dict(color=COLORS["accent"]),
        base=baseline_h,
        hovertemplate='%{y}<br>TTL: %{x:.2f} h<extra></extra>',
    ))
    
    # Positive perturbation
    fig.add_trace(go.Bar(
        y=labels,
        x=[t - baseline_h for t in ttl_plus],
        orientation='h',
        name='Param +20%',
        marker=dict(color=COLORS["warning"]),
        base=baseline_h,
        hovertemplate='%{y}<br>TTL: %{x:.2f} h<extra></extra>',
    ))
    
    # Baseline line
    fig.add_vline(x=baseline_h, line_dash="dash", line_color=COLORS["primary"],
                  line_width=LINE_WIDTHS["main"],
                  annotation_text=f"Baseline: {baseline_h:.2f} h",
                  annotation_position="top",
                  annotation_font_size=FONT_SIZES["annotation"])
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(text="Parameter Sensitivity Tornado Chart", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time-to-Live TTL (hours)",
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
# Spider Chart (Radar Chart)
# =====================================================

def plot_sensitivity_spider(sens_results, ax=None, show=True, save_path=None):
    """
    Plot sensitivity spider chart (radar chart)
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    values = [abs(sens_results[p]["S_norm"]) for p in params]
    labels = [_get_label(p) for p in params]
    
    # Close the figure
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(8, 145, 178, 0.25)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        marker=dict(size=6, color=COLORS["accent"]),
        hovertemplate='%{theta}<br>|S|: %{r:.4f}<extra></extra>',
    ))
    
    width, height = FIGURE_SIZES["square"]
    fig.update_layout(
        title=dict(text="Parameter Sensitivity Radar Chart", font=dict(size=FONT_SIZES["title"])),
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
# Sensitivity Heatmap
# =====================================================

def plot_sensitivity_heatmap(sens_results, ax=None, show=True, save_path=None):
    """
    Plot sensitivity heatmap
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    labels = [_get_label(p) for p in params]
    
    # Data matrix
    data = np.zeros((len(params), 3))
    for i, p in enumerate(params):
        data[i, 0] = sens_results[p]["TTL-"] / 3600
        data[i, 1] = (sens_results[p]["TTL+"] + sens_results[p]["TTL-"]) / 2 / 3600
        data[i, 2] = sens_results[p]["TTL+"] / 3600
    
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=data,
        x=['Param -20%', 'Baseline', 'Param +20%'],
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
        title=dict(text="Parameter Sensitivity Heatmap", font=dict(size=FONT_SIZES["title"])),
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
# Comprehensive Sensitivity Analysis
# =====================================================

def plot_sensitivity_comprehensive(sens_results, baseline_ttl, filename=None, subdir="",
                                    save_path=None, show=None):
    """
    Plot comprehensive sensitivity analysis chart
    """
    params = [p for p in sens_results.keys() if not p.startswith('_')]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Normalized Sensitivity", "Tornado Chart", "Radar Chart", "Analysis Insights"),
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "polar"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # 1. Bar chart
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
    
    # 2. Tornado chart
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
    
    # 3. Radar chart
    radar_values = [abs(sens_results[p]["S_norm"]) for p in params]
    radar_values_closed = radar_values + [radar_values[0]]
    labels_closed = labels + [labels[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=radar_values_closed,
        theta=labels_closed,
        fill='toself',
        fillcolor='rgba(8, 145, 178, 0.25)',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=2, col=1)
    
    # 4. Insights table
    s_norms = {p: sens_results[p]["S_norm"] for p in params}
    most_sensitive = max(params, key=lambda p: abs(s_norms[p]))
    least_sensitive = min(params, key=lambda p: abs(s_norms[p]))
    
    positive_sens = [_get_label(p) for p in params if s_norms[p] > 0]
    negative_sens = [_get_label(p) for p in params if s_norms[p] < 0]
    
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Analysis</b>", "<b>Result</b>"],
            fill_color=COLORS["accent"],
            font=dict(color='white', size=FONT_SIZES["axis_tick"]),
            align='center', height=24,
        ),
        cells=dict(
            values=[
                ["Baseline TTL", "Most Sensitive", "Least Sensitive", "Negative Sens.", "Positive Sens."],
                [f"{baseline_h:.2f} h",
                 f"{_get_label(most_sensitive)} ({s_norms[most_sensitive]:.4f})",
                 f"{_get_label(least_sensitive)} ({s_norms[least_sensitive]:.4f})",
                 ", ".join(negative_sens) if negative_sens else "None",
                 ", ".join(positive_sens) if positive_sens else "None"]
            ],
            fill_color=['rgba(236, 240, 241, 0.8)', 'white'],
            font=dict(size=FONT_SIZES["annotation"]),
            align=['left', 'left'], height=22,
        )
    ), row=2, col=2)
    
    # Layout
    fig.update_xaxes(title_text="Normalized Sens.", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="TTL (h)", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    
    width, height = FIGURE_SIZES["composite"]
    fig.update_layout(
        title=dict(text="Parameter Sensitivity Analysis Report", font=dict(size=FONT_SIZES["title"])),
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
