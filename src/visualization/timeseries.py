# visualization/timeseries.py
# Time Series Visualization Module (Plotly - Single Column Paper Optimized)
#
# Professional time series visualizations:
# - SOC Curve
# - Power Curve
# - Temperature Curve
# - Usage State Timeline
# - Composite Charts

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from .config import (
    COLORS, STATE_COLORS, DEFAULT_COLORS, POWER_BREAKDOWN_COLORS,
    FONT_SIZES, LINE_WIDTHS, FIGURE_SIZES,
    to_hours, get_show_plots, save_plotly_figure,
    hex_to_rgba,
    setup_style,
)

# 确保样式已初始化
setup_style()


# =====================================================
# SOC 曲线
# =====================================================

def plot_soc_curve(result, ax=None, show=True, save_path=None):
    """
    Plot SOC (State of Charge) over time
    
    Parameters:
        result : dict - Simulation result
        ax : Compatibility parameter (ignored)
        show : bool - Whether to display the figure
        save_path : str - Save path
    
    Returns:
        plotly.graph_objects.Figure
    """
    time_h = to_hours(result["time"])
    soc_percent = [s * 100 for s in result["SOC"]]
    
    fig = go.Figure()
    
    # Main SOC curve
    fig.add_trace(go.Scatter(
        x=time_h,
        y=soc_percent,
        mode='lines',
        name='Battery SOC',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        fill='tozeroy',
        fillcolor='rgba(8, 145, 178, 0.15)',
        hovertemplate='Time: %{x:.2f} h<br>SOC: %{y:.1f}%<extra></extra>'
    ))
    
    # Low battery warning line
    fig.add_hline(
        y=20, line_dash="dash", line_color=COLORS["warning"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text="Low Battery (20%)",
        annotation_position="top right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    # Critical battery warning line
    fig.add_hline(
        y=5, line_dash="dot", line_color=COLORS["danger"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text="Critical (5%)",
        annotation_position="bottom right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    # Layout
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="Battery SOC Over Time", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time (hours)",
        yaxis_title="State of Charge (%)",
        yaxis=dict(range=[0, 105]),
        xaxis=dict(range=[0, max(time_h)]),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=45, b=45),
    )
    
    # Add TTL annotation
    ttl_hours = result["TTL"] / 3600
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>TTL: {ttl_hours:.2f} h</b>",
        showarrow=False,
        font=dict(size=FONT_SIZES["annotation"]),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        borderpad=3,
        xanchor="right", yanchor="top",
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="wide")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 功耗曲线
# =====================================================

def plot_power_curve(result, ax=None, show=True, save_path=None):
    """
    Plot power consumption over time
    """
    time_h = to_hours(result["time"])
    power = np.array(result["Power"])
    
    # Calculate moving average
    window_size = min(100, len(power) // 10) if len(power) > 100 else 1
    if window_size > 1:
        power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
        time_smooth = time_h[:len(power_smooth)]
    else:
        power_smooth = power
        time_smooth = time_h
    
    fig = go.Figure()
    
    # Instantaneous power (semi-transparent)
    fig.add_trace(go.Scatter(
        x=time_h,
        y=power,
        mode='lines',
        name='Instantaneous',
        line=dict(color=COLORS["neutral"], width=0.5),
        opacity=0.3,
        hoverinfo='skip',
    ))
    
    # Smoothed power curve
    fig.add_trace(go.Scatter(
        x=time_smooth,
        y=power_smooth,
        mode='lines',
        name='Smoothed',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        hovertemplate='Time: %{x:.2f} h<br>Power: %{y:.3f} W<extra></extra>'
    ))
    
    # Average power line
    avg_power = np.mean(power)
    fig.add_hline(
        y=avg_power, line_dash="dash", line_color=COLORS["accent"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text=f"Avg: {avg_power:.2f} W",
        annotation_position="top right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="System Power Consumption", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time (hours)",
        yaxis_title="Power (W)",
        xaxis=dict(range=[0, max(time_h)]),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=45, b=45),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="wide")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 温度曲线
# =====================================================

def plot_temperature_curve(result, ax=None, show=True, save_path=None, T_amb=298.15):
    """
    Plot battery temperature over time
    """
    time_h = to_hours(result["time"])
    temp_c = [tb - 273.15 for tb in result["Tb"]]
    T_amb_c = T_amb - 273.15
    
    fig = go.Figure()
    
    # Temperature curve
    fig.add_trace(go.Scatter(
        x=time_h,
        y=temp_c,
        mode='lines',
        name='Battery Temp',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        fill='tonexty',
        fillcolor='rgba(220, 38, 38, 0.12)',
        hovertemplate='Time: %{x:.2f} h<br>Temp: %{y:.1f}°C<extra></extra>'
    ))
    
    # Ambient temperature reference line
    fig.add_hline(
        y=T_amb_c, line_dash="dash", line_color=COLORS["neutral"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text=f"Ambient: {T_amb_c:.1f}°C",
        annotation_position="bottom right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    # High temperature warning line
    fig.add_hline(
        y=45, line_dash="dot", line_color=COLORS["danger"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text="Warning (45°C)",
        annotation_position="top right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="Battery Temperature Over Time", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time (hours)",
        yaxis_title="Temperature (°C)",
        xaxis=dict(range=[0, max(time_h)]),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=45, b=45),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="wide")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 状态时间线
# =====================================================

def plot_state_timeline(result, ax=None, show=True, save_path=None):
    """
    Plot usage state timeline
    """
    time_h = to_hours(result["time"])
    states = result["State"]
    
    # Preserve state appearance order
    unique_states = []
    for s in states:
        if s not in unique_states:
            unique_states.append(s)
    
    fig = go.Figure()
    
    # Plot state segments
    prev_state = states[0]
    start_time = time_h[0]
    state_spans = []
    
    for i, (t, state) in enumerate(zip(time_h, states)):
        is_last = (i == len(states) - 1)
        if state != prev_state or is_last:
            end_t = t if not is_last else time_h[-1]
            state_spans.append((prev_state, start_time, end_t))
            start_time = t
            prev_state = state
    
    # Draw color blocks
    for state, start_t, end_t in state_spans:
        color = STATE_COLORS.get(state, COLORS["neutral"])
        fig.add_shape(
            type="rect",
            x0=start_t, x1=end_t, y0=0, y1=1,
            fillcolor=color,
            opacity=0.85,
            line_width=0,
        )
    
    # Add legend
    for state in unique_states:
        color = STATE_COLORS.get(state, COLORS["neutral"])
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            name=state,
            showlegend=True,
        ))
    
    width, height = FIGURE_SIZES["timeline"]
    fig.update_layout(
        title=dict(text="Usage State Timeline", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="Time (hours)",
        xaxis=dict(range=[0, max(time_h)]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        width=width,
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.5,
            xanchor="center",
            x=0.5,
            font=dict(size=FONT_SIZES["legend"]),
        ),
        margin=dict(l=50, r=20, t=45, b=60),
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="timeline")
    
    if show and get_show_plots():
        fig.show()
    
    return fig


# =====================================================
# 单次仿真基础图表
# =====================================================

def plot_single_run(result, filename=None, subdir="", save_path=None, show=None):
    """
    Plot basic charts for a single simulation run (SOC + Power)
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Battery SOC", "System Power"),
        vertical_spacing=0.12,
    )
    
    time_h = to_hours(result["time"])
    soc_percent = [s * 100 for s in result["SOC"]]
    power = np.array(result["Power"])
    
    # SOC curve
    fig.add_trace(go.Scatter(
        x=time_h, y=soc_percent,
        mode='lines', name='SOC',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        fill='tozeroy',
        fillcolor='rgba(8, 145, 178, 0.15)',
        showlegend=False,
    ), row=1, col=1)
    
    fig.add_hline(y=20, line_dash="dash", line_color=COLORS["warning"],
                  line_width=1, row=1, col=1)
    
    # Power curve (smoothed)
    window_size = min(100, len(power) // 10) if len(power) > 100 else 1
    if window_size > 1:
        power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
        time_smooth = time_h[:len(power_smooth)]
    else:
        power_smooth = power
        time_smooth = time_h
    
    fig.add_trace(go.Scatter(
        x=time_smooth, y=power_smooth,
        mode='lines', name='Power',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=2, col=1)
    
    avg_power = np.mean(power)
    fig.add_hline(y=avg_power, line_dash="dash", line_color=COLORS["accent"],
                  line_width=1, row=2, col=1)
    
    # Update layout
    ttl_hours = result["TTL"] / 3600
    width, height = FIGURE_SIZES["tall"]
    
    fig.update_layout(
        title=dict(
            text=f"Battery Simulation Results | TTL: {ttl_hours:.2f} hours",
            font=dict(size=FONT_SIZES["title"]),
        ),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=60, b=45),
    )
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=2, col=1)
    
    path = filename or save_path
    if path:
        save_plotly_figure(fig, path, subdir, size_type="tall")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig


# =====================================================
# Comprehensive Dashboard (Interface compatibility)
# =====================================================

def plot_comprehensive_dashboard(result, save_path=None, T_amb=298.15, show=None):
    """
    Plot comprehensive charts (multiple independent subplots)
    Maintains interface compatibility, actually plots multiple separate charts
    """
    # Directly call plot_single_run
    return plot_single_run(result, save_path=save_path, show=show)


# =====================================================
# Composite Chart: System Analysis (Temperature + Power Breakdown + State Timeline)
# Design philosophy: Compact, integrated, paper-quality system analysis
# =====================================================

def plot_composite_power_temperature(result, save_path=None, T_amb=298.15, show=None):
    """
    Plot system analysis composite chart: Temperature + Power Breakdown + State Timeline
    
    Design features:
    - Shared time axis, compact vertical layout
    - Temperature chart: line + light fill + threshold lines
    - Power chart: stacked area + boundary lines
    - State chart: pure color blocks, minimal cognitive load
    - Legends distributed in respective subplot areas
    - Bright, high-contrast color scheme
    """
    time_h = np.array(to_hours(result["time"]))
    temp_c = np.array([tb - 273.15 for tb in result["Tb"]])
    T_amb_c = T_amb - 273.15
    states = result["State"]
    ttl_hours = result["TTL"] / 3600
    max_time = max(time_h)
    
    # Create compact subplot layout
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.38, 0.48, 0.14],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )
    
    # ========== 1. Temperature Curve (line + light fill + threshold) ==========
    
    # Ambient temperature reference fill (very light)
    fig.add_trace(go.Scatter(
        x=time_h, y=[T_amb_c] * len(time_h),
        mode='lines', name='',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=1)
    
    # Battery temperature curve (with light fill to ambient)
    fig.add_trace(go.Scatter(
        x=time_h, y=temp_c,
        mode='lines', name='Battery Temp',
        line=dict(color=COLORS["secondary"], width=2.0),
        fill='tonexty',
        fillcolor=hex_to_rgba(COLORS["secondary"], 0.12),
        showlegend=False,
    ), row=1, col=1)
    
    # Ambient temperature reference line (dashed)
    fig.add_hline(
        y=T_amb_c, line_dash="dash", line_color=COLORS["neutral"],
        line_width=1.0, row=1, col=1,
    )
    # Ambient temperature annotation (dynamic offset)
    temp_range = max(temp_c) - min(temp_c) if max(temp_c) > min(temp_c) else 5
    annotation_offset = max(temp_range * 0.05, 0.5)
    fig.add_trace(go.Scatter(
        x=[max_time * 0.03], y=[T_amb_c + annotation_offset],
        mode='text',
        text=[f'Ambient {T_amb_c:.0f}°C'],
        textfont=dict(size=7, color=COLORS["neutral"]),
        textposition='middle right',
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=1)
    
    # High temperature warning line (45°C)
    TEMP_THRESHOLD = 45
    if max(temp_c) > 40:
        fig.add_hline(
            y=TEMP_THRESHOLD, line_dash="dot", line_color=COLORS["danger"],
            line_width=1.0, row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[max_time * 0.03], y=[TEMP_THRESHOLD + annotation_offset],
            mode='text',
            text=[f'Warning {TEMP_THRESHOLD}°C'],
            textfont=dict(size=7, color=COLORS["danger"]),
            textposition='middle right',
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)
    
    # ========== 2. Power Breakdown (stacked area + boundary lines) ==========
    
    has_breakdown = "Power_screen" in result
    
    if has_breakdown:
        # Stack order: bottom to top
        layers = [
            ("Background", result["Power_background"]),
            ("GPS", result["Power_gps"]),
            ("Radio", result["Power_radio"]),
            ("CPU", result["Power_cpu"]),
            ("Screen", result["Power_screen"]),
        ]
        
        # Line styles (differentiate by boundary lines, not just color)
        line_styles = {
            "Background": dict(width=0.8, color="#7C3AED", dash="dot"),
            "GPS": dict(width=0.8, color="#EA580C"),
            "Radio": dict(width=0.8, color="#D97706"),
            "CPU": dict(width=0.8, color="#059669"),
            "Screen": dict(width=1.0, color="#2563EB"),
        }
        
        for name, data in layers:
            fig.add_trace(go.Scatter(
                x=time_h, y=data,
                mode='lines', name=name,
                stackgroup='power',
                fillcolor=hex_to_rgba(POWER_BREAKDOWN_COLORS[name], 0.75),
                line=line_styles[name],
                showlegend=False,
            ), row=2, col=1)
        
        total_power = (np.array(result["Power_screen"]) +
                      np.array(result["Power_cpu"]) +
                      np.array(result["Power_radio"]) +
                      np.array(result["Power_gps"]) +
                      np.array(result["Power_background"]))
        avg_power = np.mean(total_power)
        max_power = np.max(total_power)
        
        # Power module legend position configuration
        LEGEND_START_Y = 0.92
        LEGEND_SPACING = 0.12
        LEGEND_X_POS = 0.88
        
        layer_names = ["Screen", "CPU", "Radio", "GPS", "Bkgnd"]
        layer_full = ["Screen", "CPU", "Radio", "GPS", "Background"]
        for i, (name, full_name) in enumerate(zip(layer_names, layer_full)):
            y_pos = max_power * (LEGEND_START_Y - i * LEGEND_SPACING)
            fig.add_trace(go.Scatter(
                x=[max_time * LEGEND_X_POS], y=[y_pos],
                mode='text',
                text=[f'■{name}'],
                textfont=dict(size=7, color=POWER_BREAKDOWN_COLORS[full_name]),
                textposition='middle right',
                showlegend=False,
                hoverinfo='skip',
            ), row=2, col=1)
    else:
        power = np.array(result["Power"])
        fig.add_trace(go.Scatter(
            x=time_h, y=power,
            mode='lines', name='Total Power',
            fill='tozeroy',
            fillcolor=hex_to_rgba(COLORS["secondary"], 0.15),
            line=dict(color=COLORS["secondary"], width=1.5),
            showlegend=False,
        ), row=2, col=1)
        avg_power = np.mean(power)
        max_power = np.max(power)
    
    # Average power reference line (dynamic offset)
    power_annotation_offset = max(max_power * 0.05, 0.1)
    fig.add_hline(
        y=avg_power, line_dash="dash", line_color=COLORS["primary"],
        line_width=1.0, row=2, col=1,
    )
    fig.add_trace(go.Scatter(
        x=[max_time * 0.03], y=[avg_power + power_annotation_offset],
        mode='text',
        text=[f'Avg {avg_power:.2f}W'],
        textfont=dict(size=7, color=COLORS["primary"]),
        textposition='middle right',
        showlegend=False,
        hoverinfo='skip',
    ), row=2, col=1)
    
    # ========== 3. State Timeline (pure color blocks, minimal cognitive load) ==========
    
    unique_states = []
    for s in states:
        if s not in unique_states:
            unique_states.append(s)
    
    # Draw state color blocks
    prev_state = states[0]
    start_time = time_h[0]
    
    for i, (t, state) in enumerate(zip(time_h, states)):
        is_last = (i == len(states) - 1)
        if state != prev_state or is_last:
            end_t = t if not is_last else time_h[-1]
            color = STATE_COLORS.get(prev_state, COLORS["neutral"])
            
            fig.add_shape(
                type="rect",
                x0=start_time, x1=end_t, y0=0, y1=1,
                fillcolor=color,
                opacity=0.9,
                line_width=0,
                row=3, col=1,
            )
            start_time = t
            prev_state = state
    
    # State legend position configuration
    STATE_LEGEND_START_X = 0.08
    STATE_LEGEND_WIDTH = 0.85
    
    # State legend (horizontal at bottom, using paper coordinates)
    n_states = len(unique_states)
    for i, state in enumerate(unique_states):
        color = STATE_COLORS.get(state, COLORS["neutral"])
        x_pos = STATE_LEGEND_START_X + i * (STATE_LEGEND_WIDTH / n_states)
        fig.add_annotation(
            x=x_pos, y=-0.02,
            xref="paper", yref="paper",
            text=f"■ {state}",
            showarrow=False,
            font=dict(size=7, color=color),
            xanchor="left",
        )
    
    # ========== Layout Settings ==========
    
    # Title as small annotation in top right
    fig.add_annotation(
        x=0.95, y=1.01,
        xref="paper", yref="paper",
        text=f"TTL {ttl_hours:.2f} h | Avg Power {avg_power:.2f} W",
        showarrow=False,
        font=dict(size=9, color=COLORS["primary"]),
        xanchor="right",
    )
    
    # Compact layout (increased right margin for legend)
    fig.update_layout(
        width=700,
        height=520,
        margin=dict(l=50, r=45, t=22, b=38),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # Y-axis labels (side, small font)
    fig.update_yaxes(
        title_text="Temp(°C)", title_font_size=9, title_standoff=5,
        tickfont_size=8, row=1, col=1,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(100,100,100,0.15)",
    )
    fig.update_yaxes(
        title_text="Power(W)", title_font_size=9, title_standoff=5,
        tickfont_size=8, row=2, col=1,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(100,100,100,0.15)",
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, row=3, col=1,
        range=[0, 1], fixedrange=True,
    )
    
    # X-axis (shared time axis, only bottom shows ticks and labels)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(
        title_text="Time (hours)", title_font_size=9, title_standoff=3,
        tickfont_size=8, row=3, col=1,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(100,100,100,0.2)",
    )
    
    # Save (static format)
    if save_path:
        save_plotly_figure(fig, save_path, size_type="composite")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig


# =====================================================
# SOC Comparison Chart
# =====================================================

def plot_soc_comparison(result, save_path=None, show=None):
    """
    Plot SOC comparison: showing effects of different correction factors
    """
    time_h = np.array(to_hours(result["time"]))
    
    fig = go.Figure()
    
    # Line style configuration
    line_configs = {
        "SOC_uncorrected": dict(color=COLORS["neutral"], dash="solid", width=1.8, name="No Correction"),
        "SOC_voltage_only": dict(color=COLORS["accent"], dash="dash", width=1.8, name="Voltage Only"),
        "SOC_temperature_only": dict(color=COLORS["success"], dash="dashdot", width=1.8, name="Temperature Only"),
        "SOC_aging_only": dict(color="#A855F7", dash="dot", width=2.0, name="Aging Only"),
        "SOC": dict(color=COLORS["secondary"], dash="solid", width=LINE_WIDTHS["main"], name="All Corrections"),
    }
    
    ttl_map = {}
    x_max = 0
    
    for key, config in line_configs.items():
        if key in result:
            soc_data = np.array(result[key]) * 100
            n = min(len(soc_data), len(time_h))
            
            fig.add_trace(go.Scatter(
                x=time_h[:n],
                y=soc_data[:n],
                mode='lines',
                name=config["name"],
                line=dict(color=config["color"], dash=config["dash"], width=config["width"]),
                hovertemplate=f'{config["name"]}<br>Time: %{{x:.2f}} h<br>SOC: %{{y:.1f}}%<extra></extra>'
            ))
            
            # Estimate TTL
            idx_zero = np.where(soc_data[:n] <= 0)[0]
            if len(idx_zero) > 0:
                ttl_map[config["name"]] = time_h[idx_zero[0]]
            else:
                ttl_map[config["name"]] = time_h[n-1]
            
            x_max = max(x_max, ttl_map[config["name"]])
    
    # Critical battery lines
    fig.add_hline(y=20, line_dash="dot", line_color=COLORS["warning"],
                  line_width=1, opacity=0.7)
    fig.add_hline(y=5, line_dash="dot", line_color=COLORS["danger"],
                  line_width=1, opacity=0.7)
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(
            text="SOC Comparison: Correction Factor Analysis",
            font=dict(size=FONT_SIZES["title"]),
        ),
        xaxis_title="Time (hours)",
        yaxis_title="State of Charge (%)",
        xaxis=dict(range=[0, x_max * 1.02]),
        yaxis=dict(range=[0, 105]),
        width=width,
        height=height,
        legend=dict(
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            font=dict(size=FONT_SIZES["legend"]),
        ),
        margin=dict(l=50, r=20, t=50, b=45),
    )
    
    # Add statistics box
    stats_lines = ["<b>TTL Comparison</b>"]
    for name, ttl in sorted(ttl_map.items(), key=lambda x: -x[1]):
        stats_lines.append(f"{name}: {ttl:.2f} h")
    
    fig.add_annotation(
        x=0.02, y=0.02, xref="paper", yref="paper",
        text="<br>".join(stats_lines),
        showarrow=False,
        font=dict(family="monospace", size=FONT_SIZES["annotation"]-1),
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        borderpad=4,
        align="left",
        xanchor="left", yanchor="bottom",
    )
    
    if save_path:
        save_plotly_figure(fig, save_path, size_type="default")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig
