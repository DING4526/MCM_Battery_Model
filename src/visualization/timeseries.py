# visualization/timeseries.py
# 单次仿真时间序列可视化模块
#
# 提供丰富的时间序列可视化功能：
# - SOC 曲线
# - 功耗曲线
# - 温度曲线
# - 使用状态时间线
# - 综合仪表板

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# 从统一配置模块导入
from .config import (
    setup_style as _setup_style,
    COLORS,
    STATE_COLORS,
    to_hours as _to_hours,
    save_figure,
    get_save_path,
    smart_savefig,
    get_show_plots,
)


# =====================================================
# 单独曲线绘制函数
# =====================================================

def plot_soc_curve(result, ax=None, show=True, save_path=None):
    """
    绘制 SOC（电量）随时间变化曲线
    
    参数：
        result : dict - 仿真结果
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    time_h = _to_hours(result["time"])
    soc_percent = [s * 100 for s in result["SOC"]]
    
    # 主曲线
    ax.plot(time_h, soc_percent, color=COLORS["primary"], linewidth=2, label="SOC")
    
    # 关键电量线
    ax.axhline(y=20, color=COLORS["success"], linestyle='--', alpha=0.7, label="低电量警告 (20%)")
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.7, label="极低电量 (5%)")
    
    # 填充区域
    ax.fill_between(time_h, soc_percent, alpha=0.3, color=COLORS["primary"])
    
    ax.set_xlabel("时间 (小时)", fontsize=11)
    ax.set_ylabel("电量 SOC (%)", fontsize=11)
    ax.set_title("电池电量变化曲线", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(time_h))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_power_curve(result, ax=None, show=True, save_path=None):
    """
    绘制功耗随时间变化曲线
    
    参数：
        result : dict - 仿真结果
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    time_h = _to_hours(result["time"])
    power = result["Power"]
    
    # 计算滑动平均（平滑曲线）
    window_size = min(100, len(power) // 10) if len(power) > 100 else 1
    if window_size > 1:
        power_smooth = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
        time_smooth = time_h[:len(power_smooth)]
    else:
        power_smooth = power
        time_smooth = time_h
    
    # 原始数据（半透明）
    ax.plot(time_h, power, color=COLORS["neutral"], alpha=0.3, linewidth=0.5, label="瞬时功耗")
    
    # 平滑曲线
    ax.plot(time_smooth, power_smooth, color=COLORS["accent"], linewidth=2, label="平滑功耗")
    
    # 平均功耗线
    avg_power = np.mean(power)
    ax.axhline(y=avg_power, color=COLORS["secondary"], linestyle='--', alpha=0.8, 
               label=f"平均功耗: {avg_power:.2f} W")
    
    ax.set_xlabel("时间 (小时)", fontsize=11)
    ax.set_ylabel("功耗 (W)", fontsize=11)
    ax.set_title("系统功耗变化曲线", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(time_h))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_temperature_curve(result, ax=None, show=True, save_path=None, T_amb=298.15):
    """
    绘制温度随时间变化曲线
    
    参数：
        result : dict - 仿真结果
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        T_amb : float - 环境温度（K）
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    time_h = _to_hours(result["time"])
    temp_c = [tb - 273.15 for tb in result["Tb"]]  # 转换为摄氏度
    T_amb_c = T_amb - 273.15
    
    # 温度曲线
    ax.plot(time_h, temp_c, color=COLORS["success"], linewidth=2, label="电池温度")
    
    # 环境温度参考线
    ax.axhline(y=T_amb_c, color=COLORS["neutral"], linestyle='--', alpha=0.7, 
               label=f"环境温度: {T_amb_c:.1f}°C")
    
    # 高温警告线
    ax.axhline(y=45, color='red', linestyle=':', alpha=0.7, label="高温警告 (45°C)")
    
    # 温度区域填充
    ax.fill_between(time_h, T_amb_c, temp_c, alpha=0.2, color=COLORS["success"])
    
    ax.set_xlabel("时间 (小时)", fontsize=11)
    ax.set_ylabel("温度 (°C)", fontsize=11)
    ax.set_title("电池温度变化曲线", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(time_h))
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_state_timeline(result, ax=None, show=True, save_path=None):
    """
    绘制使用状态时间线图
    
    参数：
        result : dict - 仿真结果
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    
    time_h = _to_hours(result["time"])
    states = result["State"]
    
    # 将状态转换为数值用于绘制
    unique_states = list(set(states))
    state_to_num = {s: i for i, s in enumerate(unique_states)}
    
    # 绘制状态色块
    prev_state = states[0]
    start_time = time_h[0]
    labeled_states = set()  # 使用集合跟踪已标注的状态
    
    for i, (t, state) in enumerate(zip(time_h, states)):
        if state != prev_state or i == len(states) - 1:
            color = STATE_COLORS.get(prev_state, COLORS["neutral"])
            # 使用集合进行 O(1) 查找
            label = prev_state if prev_state not in labeled_states else ""
            ax.axvspan(start_time, t, alpha=0.7, color=color, label=label)
            if label:
                labeled_states.add(prev_state)
            start_time = t
            prev_state = state
    
    # 创建图例
    handles = [mpatches.Patch(color=STATE_COLORS.get(s, COLORS["neutral"]), label=s) 
               for s in unique_states]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=len(unique_states), fontsize=9)
    
    ax.set_xlabel("时间 (小时)", fontsize=11)
    ax.set_ylabel("使用状态", fontsize=11)
    ax.set_title("手机使用状态时间线", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(time_h))
    ax.set_yticks([])
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_single_run(result, save_path=None, show=None):
    """
    绘制单次仿真的基础图表（SOC + 功耗）
    
    参数：
        result : dict - 仿真结果
        save_path : str - 保存路径或文件名
        show : bool - 是否显示图形，None 则使用全局设置
    """
    _setup_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # SOC 曲线
    plot_soc_curve(result, ax=axes[0], show=False)
    
    # 功耗曲线
    plot_power_curve(result, ax=axes[1], show=False)
    
    # 添加总标题
    ttl_hours = result["TTL"] / 3600
    fig.suptitle(f"电池仿真结果 | 续航时间 TTL = {ttl_hours:.2f} 小时", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        smart_savefig(save_path)
    
    # 使用参数或全局设置决定是否显示
    if show is None:
        show = get_show_plots()
    if show:
        plt.show()
    
    return fig


def plot_comprehensive_dashboard(result, save_path=None, T_amb=298.15, show=None):
    """
    绘制综合仪表板（比赛级别可视化）
    
    包含：
    - SOC 曲线
    - 功耗曲线
    - 温度曲线
    - 状态时间线
    - 统计信息面板
    
    参数：
        result : dict - 仿真结果
        save_path : str - 保存路径
        T_amb : float - 环境温度（K）
        show : bool - 是否显示图形，None 则使用全局设置
    """
    _setup_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[2, 2, 2, 1], hspace=0.3, wspace=0.3)
    
    # ===== SOC 曲线 (左上) =====
    ax1 = fig.add_subplot(gs[0, :3])
    plot_soc_curve(result, ax=ax1, show=False)
    
    # ===== 统计信息面板 (右上) =====
    ax_stats = fig.add_subplot(gs[0, 3])
    ax_stats.axis('off')
    
    # 计算统计信息
    ttl_hours = result["TTL"] / 3600
    avg_power = np.mean(result["Power"])
    max_power = np.max(result["Power"])
    min_power = np.min(result["Power"])
    avg_temp = np.mean(result["Tb"]) - 273.15
    max_temp = np.max(result["Tb"]) - 273.15
    
    # 统计文本（使用纯ASCII边框，兼容性更好）
    stats_text = f"""
    +-------------------------+
    |   仿 真 统 计 摘 要     |
    +-------------------------+
    |  续航时间:  {ttl_hours:>6.2f} h   |
    |                         |
    |  平均功耗:  {avg_power:>6.2f} W   |
    |  最大功耗:  {max_power:>6.2f} W   |
    |  最小功耗:  {min_power:>6.2f} W   |
    |                         |
    |  平均温度:  {avg_temp:>6.1f} °C  |
    |  最高温度:  {max_temp:>6.1f} °C  |
    +-------------------------+
    """
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=10,
                  verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== 功耗曲线 (中左) =====
    ax2 = fig.add_subplot(gs[1, :3])
    plot_power_curve(result, ax=ax2, show=False)
    
    # ===== 功耗分布饼图 (中右) =====
    ax_pie = fig.add_subplot(gs[1, 3])
    states = result["State"]
    powers = result["Power"]
    
    # 按状态计算总能耗
    state_energy = {}
    for state, power in zip(states, powers):
        if state not in state_energy:
            state_energy[state] = 0
        state_energy[state] += power  # 假设 dt=1s
    
    labels = list(state_energy.keys())
    sizes = list(state_energy.values())
    colors = [STATE_COLORS.get(s, COLORS["neutral"]) for s in labels]
    
    ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax_pie.set_title("各状态能耗占比", fontsize=11, fontweight='bold')
    
    # ===== 温度曲线 (下左) =====
    ax3 = fig.add_subplot(gs[2, :3])
    plot_temperature_curve(result, ax=ax3, show=False, T_amb=T_amb)
    
    # ===== 状态时间占比条形图 (下右) =====
    ax_bar = fig.add_subplot(gs[2, 3])
    
    # 计算状态时间占比
    from collections import Counter
    state_counts = Counter(states)
    total = sum(state_counts.values())
    state_ratios = {k: v/total*100 for k, v in state_counts.items()}
    
    bars = ax_bar.barh(list(state_ratios.keys()), list(state_ratios.values()),
                       color=[STATE_COLORS.get(s, COLORS["neutral"]) for s in state_ratios.keys()])
    ax_bar.set_xlabel("时间占比 (%)", fontsize=10)
    ax_bar.set_title("状态时间分布", fontsize=11, fontweight='bold')
    
    # ===== 状态时间线 (底部) =====
    ax4 = fig.add_subplot(gs[3, :])
    plot_state_timeline(result, ax=ax4, show=False)
    
    # ===== 总标题 =====
    fig.suptitle(f"[Battery] 电池仿真综合仪表板 | 续航时间: {ttl_hours:.2f} 小时", 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        smart_savefig(save_path)
    
    # 使用参数或全局设置决定是否显示
    if show is None:
        show = get_show_plots()
    if show:
        plt.show()
    
    return fig
