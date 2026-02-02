# visualization/timeseries.py
# 时间序列可视化模块（Plotly 版本 - 单栏论文优化）
#
# 提供专业的时间序列可视化：
# - SOC 曲线
# - 功耗曲线
# - 温度曲线
# - 使用状态时间线
# - 复合图表

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
    绘制 SOC（电量）随时间变化曲线
    
    参数：
        result : dict - 仿真结果
        ax : 兼容参数（忽略）
        show : bool - 是否显示图形
        save_path : str - 保存路径
    
    返回：
        plotly.graph_objects.Figure
    """
    time_h = to_hours(result["time"])
    soc_percent = [s * 100 for s in result["SOC"]]
    
    fig = go.Figure()
    
    # 主 SOC 曲线
    fig.add_trace(go.Scatter(
        x=time_h,
        y=soc_percent,
        mode='lines',
        name='电池电量',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        fill='tozeroy',
        fillcolor='rgba(41, 128, 185, 0.15)',
        hovertemplate='时间: %{x:.2f} h<br>SOC: %{y:.1f}%<extra></extra>'
    ))
    
    # 低电量警告线
    fig.add_hline(
        y=20, line_dash="dash", line_color=COLORS["warning"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text="低电量 (20%)",
        annotation_position="top right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    # 极低电量警告线
    fig.add_hline(
        y=5, line_dash="dot", line_color=COLORS["danger"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text="极低电量 (5%)",
        annotation_position="bottom right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    # 布局
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="电池电量变化曲线", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="时间 (小时)",
        yaxis_title="电量 SOC (%)",
        yaxis=dict(range=[0, 105]),
        xaxis=dict(range=[0, max(time_h)]),
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=50, r=20, t=45, b=45),
    )
    
    # 添加续航时间标注
    ttl_hours = result["TTL"] / 3600
    fig.add_annotation(
        x=0.98, y=0.95, xref="paper", yref="paper",
        text=f"<b>续航: {ttl_hours:.2f} h</b>",
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
    绘制功耗随时间变化曲线
    """
    time_h = to_hours(result["time"])
    power = np.array(result["Power"])
    
    # 计算滑动平均
    window_size = min(100, len(power) // 10) if len(power) > 100 else 1
    if window_size > 1:
        power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
        time_smooth = time_h[:len(power_smooth)]
    else:
        power_smooth = power
        time_smooth = time_h
    
    fig = go.Figure()
    
    # 瞬时功耗（半透明）
    fig.add_trace(go.Scatter(
        x=time_h,
        y=power,
        mode='lines',
        name='瞬时功耗',
        line=dict(color=COLORS["neutral"], width=0.5),
        opacity=0.3,
        hoverinfo='skip',
    ))
    
    # 平滑功耗曲线
    fig.add_trace(go.Scatter(
        x=time_smooth,
        y=power_smooth,
        mode='lines',
        name='平滑功耗',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        hovertemplate='时间: %{x:.2f} h<br>功耗: %{y:.3f} W<extra></extra>'
    ))
    
    # 平均功耗线
    avg_power = np.mean(power)
    fig.add_hline(
        y=avg_power, line_dash="dash", line_color=COLORS["accent"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text=f"平均: {avg_power:.2f} W",
        annotation_position="top right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="系统功耗变化曲线", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="时间 (小时)",
        yaxis_title="功耗 (W)",
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
    绘制温度随时间变化曲线
    """
    time_h = to_hours(result["time"])
    temp_c = [tb - 273.15 for tb in result["Tb"]]
    T_amb_c = T_amb - 273.15
    
    fig = go.Figure()
    
    # 温度曲线
    fig.add_trace(go.Scatter(
        x=time_h,
        y=temp_c,
        mode='lines',
        name='电池温度',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        fill='tonexty',
        fillcolor='rgba(192, 57, 43, 0.12)',
        hovertemplate='时间: %{x:.2f} h<br>温度: %{y:.1f}°C<extra></extra>'
    ))
    
    # 环境温度参考线
    fig.add_hline(
        y=T_amb_c, line_dash="dash", line_color=COLORS["neutral"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text=f"环境: {T_amb_c:.1f}°C",
        annotation_position="bottom right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    # 高温警告线
    fig.add_hline(
        y=45, line_dash="dot", line_color=COLORS["danger"],
        line_width=LINE_WIDTHS["secondary"],
        annotation_text="高温警告 (45°C)",
        annotation_position="top right",
        annotation_font_size=FONT_SIZES["annotation"],
    )
    
    width, height = FIGURE_SIZES["wide"]
    fig.update_layout(
        title=dict(text="电池温度变化曲线", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="时间 (小时)",
        yaxis_title="温度 (°C)",
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
    绘制使用状态时间线图
    """
    time_h = to_hours(result["time"])
    states = result["State"]
    
    # 保持状态出现顺序
    unique_states = []
    for s in states:
        if s not in unique_states:
            unique_states.append(s)
    
    fig = go.Figure()
    
    # 绘制每个状态的时间段
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
    
    # 绘制色块
    for state, start_t, end_t in state_spans:
        color = STATE_COLORS.get(state, COLORS["neutral"])
        fig.add_shape(
            type="rect",
            x0=start_t, x1=end_t, y0=0, y1=1,
            fillcolor=color,
            opacity=0.85,
            line_width=0,
        )
    
    # 添加图例
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
        title=dict(text="使用状态时间线", font=dict(size=FONT_SIZES["title"])),
        xaxis_title="时间 (小时)",
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
    绘制单次仿真的基础图表（SOC + 功耗）
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("电池电量 (SOC)", "系统功耗"),
        vertical_spacing=0.12,
    )
    
    time_h = to_hours(result["time"])
    soc_percent = [s * 100 for s in result["SOC"]]
    power = np.array(result["Power"])
    
    # SOC 曲线
    fig.add_trace(go.Scatter(
        x=time_h, y=soc_percent,
        mode='lines', name='SOC',
        line=dict(color=COLORS["accent"], width=LINE_WIDTHS["main"]),
        fill='tozeroy',
        fillcolor='rgba(41, 128, 185, 0.15)',
        showlegend=False,
    ), row=1, col=1)
    
    fig.add_hline(y=20, line_dash="dash", line_color=COLORS["warning"],
                  line_width=1, row=1, col=1)
    
    # 功耗曲线（平滑）
    window_size = min(100, len(power) // 10) if len(power) > 100 else 1
    if window_size > 1:
        power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
        time_smooth = time_h[:len(power_smooth)]
    else:
        power_smooth = power
        time_smooth = time_h
    
    fig.add_trace(go.Scatter(
        x=time_smooth, y=power_smooth,
        mode='lines', name='功耗',
        line=dict(color=COLORS["secondary"], width=LINE_WIDTHS["main"]),
        showlegend=False,
    ), row=2, col=1)
    
    avg_power = np.mean(power)
    fig.add_hline(y=avg_power, line_dash="dash", line_color=COLORS["accent"],
                  line_width=1, row=2, col=1)
    
    # 更新布局
    ttl_hours = result["TTL"] / 3600
    width, height = FIGURE_SIZES["tall"]
    
    fig.update_layout(
        title=dict(
            text=f"电池仿真结果 | 续航时间: {ttl_hours:.2f} 小时",
            font=dict(size=FONT_SIZES["title"]),
        ),
        width=width,
        height=height,
        margin=dict(l=50, r=20, t=60, b=45),
    )
    
    fig.update_xaxes(title_text="时间 (小时)", row=2, col=1)
    fig.update_yaxes(title_text="SOC (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="功耗 (W)", row=2, col=1)
    
    path = filename or save_path
    if path:
        save_plotly_figure(fig, path, subdir, size_type="tall")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig


# =====================================================
# 综合仪表板（保留接口，但简化为多图）
# =====================================================

def plot_comprehensive_dashboard(result, save_path=None, T_amb=298.15, show=None):
    """
    绘制综合图表（多个独立子图）
    保留接口兼容性，实际绘制多个分离图表
    """
    # 直接调用 plot_single_run
    return plot_single_run(result, save_path=save_path, show=show)


# =====================================================
# 复合图表：系统分析图（温度 + 功耗分解 + 状态时间线）
# 设计理念：紧凑、整体化、论文级系统分析导向
# =====================================================

def plot_composite_power_temperature(result, save_path=None, T_amb=298.15, show=None):
    """
    绘制系统分析复合图：温度 + 功耗分解 + 状态时间线
    
    设计特点：
    - 共享时间轴，紧凑纵向布局
    - 温度图：折线+轻填充+阈值线
    - 功耗图：堆叠面积+边界线区分
    - 状态图：纯色块，最低认知负担
    - 图例分布在各自子图区域
    - 深沉低饱和配色
    """
    time_h = np.array(to_hours(result["time"]))
    temp_c = np.array([tb - 273.15 for tb in result["Tb"]])
    T_amb_c = T_amb - 273.15
    states = result["State"]
    ttl_hours = result["TTL"] / 3600
    max_time = max(time_h)
    
    # 创建紧凑子图布局（压缩纵向比例）
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.38, 0.48, 0.14],
        vertical_spacing=0.02,
        shared_xaxes=True,
    )
    
    # ========== 1. 温度曲线（折线+轻填充+阈值线） ==========
    
    # 环境温度参考线填充（极淡）
    fig.add_trace(go.Scatter(
        x=time_h, y=[T_amb_c] * len(time_h),
        mode='lines', name='',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=1)
    
    # 电池温度曲线（带轻填充到环境温度）
    fig.add_trace(go.Scatter(
        x=time_h, y=temp_c,
        mode='lines', name='电池温度',
        line=dict(color=COLORS["secondary"], width=2.0),
        fill='tonexty',
        fillcolor=hex_to_rgba(COLORS["secondary"], 0.08),
        showlegend=False,
    ), row=1, col=1)
    
    # 环境温度参考线（虚线）
    fig.add_hline(
        y=T_amb_c, line_dash="dash", line_color=COLORS["neutral"],
        line_width=1.0, row=1, col=1,
    )
    # 环境温度标注（动态偏移）
    temp_range = max(temp_c) - min(temp_c) if max(temp_c) > min(temp_c) else 5
    annotation_offset = max(temp_range * 0.05, 0.5)
    fig.add_trace(go.Scatter(
        x=[max_time * 0.03], y=[T_amb_c + annotation_offset],
        mode='text',
        text=[f'环境 {T_amb_c:.0f}°C'],
        textfont=dict(size=7, color=COLORS["neutral"]),
        textposition='middle right',
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=1)
    
    # 高温警戒线（45°C）
    TEMP_THRESHOLD = 45
    if max(temp_c) > 40:
        fig.add_hline(
            y=TEMP_THRESHOLD, line_dash="dot", line_color=COLORS["danger"],
            line_width=1.0, row=1, col=1,
        )
        fig.add_trace(go.Scatter(
            x=[max_time * 0.03], y=[TEMP_THRESHOLD + annotation_offset],
            mode='text',
            text=[f'警戒 {TEMP_THRESHOLD}°C'],
            textfont=dict(size=7, color=COLORS["danger"]),
            textposition='middle right',
            showlegend=False,
            hoverinfo='skip',
        ), row=1, col=1)
    
    # ========== 2. 功耗分解图（堆叠面积+边界线区分） ==========
    
    has_breakdown = "Power_screen" in result
    
    if has_breakdown:
        # 堆叠顺序：从底部到顶部
        layers = [
            ("后台", result["Power_background"]),
            ("GPS", result["Power_gps"]),
            ("无线通信", result["Power_radio"]),
            ("CPU", result["Power_cpu"]),
            ("屏幕", result["Power_screen"]),
        ]
        
        # 线型样式（通过边界线区分，而非仅颜色）
        line_styles = {
            "后台": dict(width=0.8, color="#3A3A4A", dash="dot"),
            "GPS": dict(width=0.8, color="#5A4B3A"),
            "无线通信": dict(width=0.8, color="#3A4B5A"),
            "CPU": dict(width=0.8, color="#3A5B4C"),
            "屏幕": dict(width=1.0, color="#4B5B6C"),
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
        
        # 功耗模块图例位置配置
        LEGEND_START_Y = 0.92
        LEGEND_SPACING = 0.12
        LEGEND_X_POS = 0.88
        
        layer_names = ["屏幕", "CPU", "通信", "GPS", "后台"]
        for i, name in enumerate(layer_names):
            full_name = "无线通信" if name == "通信" else name
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
            mode='lines', name='总功耗',
            fill='tozeroy',
            fillcolor=hex_to_rgba(COLORS["secondary"], 0.15),
            line=dict(color=COLORS["secondary"], width=1.5),
            showlegend=False,
        ), row=2, col=1)
        avg_power = np.mean(power)
        max_power = np.max(power)
    
    # 平均功耗参考线（动态偏移）
    power_annotation_offset = max(max_power * 0.05, 0.1)
    fig.add_hline(
        y=avg_power, line_dash="dash", line_color=COLORS["primary"],
        line_width=1.0, row=2, col=1,
    )
    fig.add_trace(go.Scatter(
        x=[max_time * 0.03], y=[avg_power + power_annotation_offset],
        mode='text',
        text=[f'均值 {avg_power:.2f}W'],
        textfont=dict(size=7, color=COLORS["primary"]),
        textposition='middle right',
        showlegend=False,
        hoverinfo='skip',
    ), row=2, col=1)
    
    # ========== 3. 状态时间线（纯色块，最低认知负担） ==========
    
    unique_states = []
    for s in states:
        if s not in unique_states:
            unique_states.append(s)
    
    # 绘制状态色块
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
    
    # 状态图例位置配置
    STATE_LEGEND_START_X = 0.08
    STATE_LEGEND_WIDTH = 0.85
    
    # 状态图例（底部水平排列，使用 paper 坐标）
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
    
    # ========== 布局设置 ==========
    
    # 标题采用小字号右上角标注形式
    fig.add_annotation(
        x=0.95, y=1.01,
        xref="paper", yref="paper",
        text=f"续航 {ttl_hours:.2f} h | 均值功耗 {avg_power:.2f} W",
        showarrow=False,
        font=dict(size=9, color=COLORS["primary"]),
        xanchor="right",
    )
    
    # 紧凑布局（增加右侧边距以容纳图例）
    fig.update_layout(
        width=700,
        height=520,
        margin=dict(l=50, r=45, t=22, b=38),
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    
    # Y轴标签（侧边小字号）
    fig.update_yaxes(
        title_text="温度(°C)", title_font_size=9, title_standoff=5,
        tickfont_size=8, row=1, col=1,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(100,100,100,0.15)",
    )
    fig.update_yaxes(
        title_text="功耗(W)", title_font_size=9, title_standoff=5,
        tickfont_size=8, row=2, col=1,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(100,100,100,0.15)",
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, row=3, col=1,
        range=[0, 1], fixedrange=True,
    )
    
    # X轴（共享时间轴，仅底部显示刻度和标签）
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(
        title_text="时间 (小时)", title_font_size=9, title_standoff=3,
        tickfont_size=8, row=3, col=1,
        showgrid=True, gridwidth=0.5, gridcolor="rgba(100,100,100,0.2)",
    )
    
    # 保存（静态格式）
    if save_path:
        save_plotly_figure(fig, save_path, size_type="composite")
    
    if show is None:
        show = get_show_plots()
    if show:
        fig.show()
    
    return fig


# =====================================================
# SOC 对比图
# =====================================================

def plot_soc_comparison(result, save_path=None, show=None):
    """
    绘制 SOC 对比图：展示各种修正对电池电量预测的影响
    """
    time_h = np.array(to_hours(result["time"]))
    
    fig = go.Figure()
    
    # 线条样式配置
    line_configs = {
        "SOC_uncorrected": dict(color=COLORS["neutral"], dash="solid", width=1.8, name="无修正"),
        "SOC_voltage_only": dict(color=COLORS["accent"], dash="dash", width=1.8, name="仅电压修正"),
        "SOC_temperature_only": dict(color=COLORS["success"], dash="dashdot", width=1.8, name="仅温度修正"),
        "SOC_aging_only": dict(color="#8E44AD", dash="dot", width=2.0, name="仅老化修正"),
        "SOC": dict(color=COLORS["secondary"], dash="solid", width=LINE_WIDTHS["main"], name="全部修正"),
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
                hovertemplate=f'{config["name"]}<br>时间: %{{x:.2f}} h<br>SOC: %{{y:.1f}}%<extra></extra>'
            ))
            
            # 估算续航时间
            idx_zero = np.where(soc_data[:n] <= 0)[0]
            if len(idx_zero) > 0:
                ttl_map[config["name"]] = time_h[idx_zero[0]]
            else:
                ttl_map[config["name"]] = time_h[n-1]
            
            x_max = max(x_max, ttl_map[config["name"]])
    
    # 关键电量线
    fig.add_hline(y=20, line_dash="dot", line_color=COLORS["warning"],
                  line_width=1, opacity=0.7)
    fig.add_hline(y=5, line_dash="dot", line_color=COLORS["danger"],
                  line_width=1, opacity=0.7)
    
    width, height = FIGURE_SIZES["default"]
    fig.update_layout(
        title=dict(
            text="SOC 对比：各修正因子效果分析",
            font=dict(size=FONT_SIZES["title"]),
        ),
        xaxis_title="时间 (小时)",
        yaxis_title="电量 SOC (%)",
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
    
    # 添加统计框
    stats_lines = ["<b>续航对比</b>"]
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
