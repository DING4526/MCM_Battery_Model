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


def plot_single_run(result, filename=None, subdir="", save_path=None, show=None):
    """
    绘制单次仿真的基础图表（SOC + 功耗）
    
    参数：
        result : dict - 仿真结果
        filename : str - 保存文件名
        subdir : str - 输出子目录
        save_path : str - 完整保存路径（兼容旧接口）
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
    
    # 保存图片
    if filename:
        smart_savefig(filename, subdir)
    elif save_path:
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


# =====================================================
# 复合图表1：温度 + 功耗堆叠图 + 状态时间线（改进版：不拥挤、更协调）
# =====================================================

def plot_composite_power_temperature(result, save_path=None, T_amb=298.15, show=None):
    """
    绘制复合图表：温度变化曲线 + 子模块功耗堆叠图 + 使用状态时间线

    改进点：
    - 去除 emoji（避免 SimHei 缺字形警告）
    - 堆叠图使用低饱和配色 + 白色细分隔线，层次更清晰
    - 堆叠图图例外置，减少图内拥挤
    - 平均功耗线不进入 legend，改用右上角注释
    - 适度弱化网格/边框，整体更协调
    """
    _setup_style()

    # 创建图表布局（温度:功耗:状态 = 1.6:2.8:0.8），更协调
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.6, 2.8, 0.8], hspace=0.22)

    time_h = _to_hours(result["time"])

    # ===== 面板1：温度曲线 =====
    ax1 = fig.add_subplot(gs[0])
    temp_c = [tb - 273.15 for tb in result["Tb"]]  # K -> °C
    T_amb_c = T_amb - 273.15

    ax1.plot(time_h, temp_c, color=COLORS["danger"], linewidth=2.2, label="电池温度")
    ax1.axhline(
        y=T_amb_c,
        color=COLORS["neutral"],
        linestyle="--",
        alpha=0.75,
        linewidth=1.4,
        label=f"环境温度: {T_amb_c:.1f}°C",
    )

    # 高温警告线 + 轻微背景提示
    ax1.axhline(y=45, color="red", linestyle=":", alpha=0.55, linewidth=1.4, label="高温警告 (45°C)")
    ax1.axhspan(45, max(50, max(temp_c) + 2), alpha=0.08, color="red")

    ax1.fill_between(time_h, T_amb_c, temp_c, alpha=0.18, color=COLORS["danger"])

    ax1.set_ylabel("温度 (°C)", fontsize=12, fontweight="bold")
    ax1.set_title("电池温度变化", fontsize=12.5, fontweight="bold", loc="left")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.18, linestyle="-")
    ax1.set_xlim(0, max(time_h))
    ax1.set_xticklabels([])  # 隐藏 x 轴标签

    # 弱化边框
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ===== 面板2：功耗堆叠图 =====
    ax2 = fig.add_subplot(gs[1])

    has_breakdown = "Power_screen" in result

    if has_breakdown:
        p_screen = np.array(result["Power_screen"])
        p_cpu = np.array(result["Power_cpu"])
        p_radio = np.array(result["Power_radio"])
        p_gps = np.array(result["Power_gps"])
        p_bg = np.array(result["Power_background"])

        # 低饱和、论文友好配色
        colors_stack = {
            "屏幕":   "#B7BBBF",  # 深灰蓝
            "GPS":    "#DEB846",  # 深金黄
            "无线通信": "#6DB3DC",  # 深青蓝
            "CPU":    "#3DC287",  # 深墨绿
            "后台":   "#845057",  # 深酒红
        }

        # 固定堆叠顺序：底噪 -> 少量 -> 变化中等 -> 峰值（读图更自然）
        layers = [p_bg, p_gps, p_radio, p_cpu, p_screen]
        labels = ["后台", "GPS", "无线通信", "CPU", "屏幕"]
        colors = [colors_stack[k] for k in labels]

        # 堆叠面积图：加白色细分隔线，层次立刻清晰
        ax2.stackplot(
            time_h,
            *layers,
            labels=labels,
            colors=colors,
            alpha=0.75,
            linewidth=0.6,
            edgecolor="white",
        )

        # 总功耗轮廓线（细一些，避免喧宾夺主）
        total_power_arr = p_screen + p_cpu + p_radio + p_gps + p_bg
        ax2.plot(
            time_h,
            total_power_arr,
            color="black",
            linewidth=1.2,
            linestyle="-",
            alpha=0.65,
            label="总功耗",
        )

        # 平均功耗：不进 legend，用注释显示，减少拥挤
        avg_power = float(np.mean(total_power_arr))
        ax2.axhline(avg_power, color="black", linestyle="--", linewidth=1.6, alpha=0.75)
        ax2.text(
            0.99,
            0.93,
            f"平均功耗: {avg_power:.2f} W",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.5),
        )

        # 图例外置到右侧
        ax2.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
            fontsize=9,
        )

        # 给右侧图例留空间
        fig.subplots_adjust(right=0.82)

    else:
        power = np.array(result["Power"])

        ax2.fill_between(time_h, 0, power, alpha=0.45, color=COLORS["accent"])
        ax2.plot(time_h, power, color=COLORS["accent"], linewidth=1.6)

        avg_power = float(np.mean(power))
        ax2.axhline(avg_power, color="black", linestyle="--", linewidth=1.6, alpha=0.75)
        ax2.text(
            0.99,
            0.93,
            f"平均功耗: {avg_power:.2f} W",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.5),
        )

        ax2.set_xlim(0, max(time_h))

    ax2.set_ylabel("功耗 (W)", fontsize=12, fontweight="bold")
    ax2.set_title("系统功耗分解（堆叠）", fontsize=12.5, fontweight="bold", loc="left")
    ax2.grid(True, alpha=0.18, linestyle="-")
    ax2.set_xlim(0, max(time_h))
    ax2.set_ylim(0, None)
    ax2.set_xticklabels([])

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ===== 面板3：状态时间线 =====
    ax3 = fig.add_subplot(gs[2])

    states = result["State"]

    # 保持状态出现顺序（避免 set() 导致顺序随机）
    unique_states = []
    for s in states:
        if s not in unique_states:
            unique_states.append(s)

    prev_state = states[0]
    start_time = time_h[0]
    labeled_states = set()

    for i, (t, state) in enumerate(zip(time_h, states)):
        is_last = (i == len(states) - 1)
        if state != prev_state or is_last:
            end_t = t if not is_last else time_h[-1]
            color = STATE_COLORS.get(prev_state, COLORS["neutral"])
            label = prev_state if prev_state not in labeled_states else ""
            ax3.axvspan(start_time, end_t, alpha=0.85, color=color, label=label)
            if label:
                labeled_states.add(prev_state)
            start_time = t
            prev_state = state

    handles = [mpatches.Patch(color=STATE_COLORS.get(s, COLORS["neutral"]), label=s) for s in unique_states]
    ax3.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=min(len(unique_states), 6),
        fontsize=9,
        framealpha=0.9,
    )

    ax3.set_xlabel("时间 (小时)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("状态", fontsize=12, fontweight="bold")
    ax3.set_title("手机使用状态时间线", fontsize=12.5, fontweight="bold", loc="left")
    ax3.set_xlim(0, max(time_h))
    ax3.set_yticks([])
    ax3.grid(False)

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # ===== 总标题 =====
    ttl_hours = result["TTL"] / 3600
    fig.suptitle(
        f"电池仿真综合分析 | 续航时间: {ttl_hours:.2f} 小时",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    # tight_layout 对 GridSpec/外置图例可能不完美，但保存时 bbox_inches 会兜底
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    if save_path:
        smart_savefig(save_path)

    if show is None:
        show = get_show_plots()
    if show:
        plt.show()

    return fig



# =====================================================
# 复合图表2：SOC 对比（各种修正效果对比）
# =====================================================

# =====================================================
# 复合图表2：SOC 对比（各种修正效果对比）
# =====================================================

def plot_soc_comparison(result, save_path=None, show=None):
    """
    绘制 SOC 对比图：展示各种修正对电池电量预测的影响

    曲线（若存在）：
    - 无修正 SOC_uncorrected
    - 仅电压 SOC_voltage_only
    - 仅温度 SOC_temperature_only
    - 仅老化 SOC_aging_only
    - 全部修正 SOC（result["SOC"]）
    """
    _setup_style()
    import numpy as np
    import matplotlib.pyplot as plt

    # -----------------------------
    # 基础时间轴：来自仿真记录（通常到“全部修正耗尽”为止）
    # -----------------------------
    t_h = np.asarray(_to_hours(result["time"]), dtype=float)
    if len(t_h) == 0:
        raise ValueError("result['time'] 为空，无法绘图。")

    # -----------------------------
    # 取出各条 SOC（单位：%）
    # -----------------------------
    series = []

    def _get_soc_percent(key):
        if key not in result:
            return None
        arr = np.asarray(result[key], dtype=float) * 100.0
        # 防御：长度对齐到 t_h
        n = min(len(arr), len(t_h))
        return arr[:n]

    soc_full = _get_soc_percent("SOC")
    if soc_full is None:
        raise ValueError("result['SOC'] 不存在，无法绘图。")

    soc_uncorrected = _get_soc_percent("SOC_uncorrected")
    soc_voltage = _get_soc_percent("SOC_voltage_only")
    soc_temp = _get_soc_percent("SOC_temperature_only")
    soc_aging = _get_soc_percent("SOC_aging_only")

    # -----------------------------
    # 样式：先“浅/辅助”，后“深/主线”（视觉更协调）
    # -----------------------------
    line_styles = {
        "uncorrected": dict(color="#6C757D", linestyle="-",  linewidth=2.0, alpha=0.90, label="无修正 (基准)"),
        "voltage":     dict(color="#2E86AB", linestyle="--", linewidth=2.0, alpha=0.90, label="仅电压修正 (OCV)"),
        "temp":        dict(color="#28A745", linestyle="-.", linewidth=2.0, alpha=0.90, label="仅温度修正"),
        "aging":       dict(color="#A23B72", linestyle=":",  linewidth=2.4, alpha=0.90, label="仅老化修正"),
        "full":        dict(color="#C73E1D", linestyle="-",  linewidth=2.8, alpha=0.98, label="全部修正 (OCV+温度+老化)"),
    }

    # -----------------------------
    # 核心：每条曲线估计“耗尽时间”，并把曲线补到 (t_end, 0)
    # -----------------------------
    def _estimate_end_time_linear(t, soc, tail_points=30):
        """
        返回该曲线 SOC 到 0 的时间（小时）。
        - 如果已经到 0：返回首次 <=0 的 t
        - 否则：用末段线性外推到 0
        """
        soc = np.asarray(soc, dtype=float)
        t = np.asarray(t, dtype=float)

        # 已经到 0：取首次到 0 的时刻
        idx0 = np.where(soc <= 0.0)[0]
        if len(idx0) > 0:
            return float(t[idx0[0]])

        # 未到 0：末段线性回归 / 斜率外推
        n = len(soc)
        k = min(tail_points, n)
        if k < 2:
            # 数据太少，退化：无法估计，返回末时刻
            return float(t[-1])

        tt = t[-k:]
        yy = soc[-k:]

        # 用最小二乘拟合 yy ≈ a*tt + b
        A = np.vstack([tt, np.ones_like(tt)]).T
        a, b = np.linalg.lstsq(A, yy, rcond=None)[0]

        # 如果斜率不为负（异常），就不外推
        if a >= -1e-9:
            return float(t[-1])

        # 解 0 = a*t + b -> t = -b/a
        t_end = -b / a

        # 不允许比当前末时刻更小
        t_end = max(float(t[-1]), float(t_end))
        return t_end

    def _extend_curve_to_zero(t, soc, t_end):
        """
        把曲线延伸到 (t_end, 0)
        - 若末点已经是 0：不变
        - 若 t_end == t[-1]：也不补
        """
        t = np.asarray(t, dtype=float)
        soc = np.asarray(soc, dtype=float)

        if len(soc) == 0:
            return t, soc

        if soc[-1] <= 0.0 or t_end <= t[-1] + 1e-9:
            return t, soc

        # 追加一个终点到 0
        t2 = np.append(t, t_end)
        soc2 = np.append(soc, 0.0)
        return t2, soc2

    # 构建要画的序列（按“浅->深”的顺序加入）
    if soc_uncorrected is not None:
        series.append(("uncorrected", soc_uncorrected))
    if soc_voltage is not None:
        series.append(("voltage", soc_voltage))
    if soc_temp is not None:
        series.append(("temp", soc_temp))
    if soc_aging is not None:
        series.append(("aging", soc_aging))
    # 主线最后
    series.append(("full", soc_full))

    # 估计每条曲线的耗尽时间，并扩展
    extended = []
    end_times = []

    for name, soc in series:
        t_end = _estimate_end_time_linear(t_h, soc, tail_points=30)
        t_ext, soc_ext = _extend_curve_to_zero(t_h, soc, t_end)
        extended.append((name, t_ext, soc_ext, t_end))
        end_times.append(t_end)

    # ✅ 横轴上限 = 所有曲线耗尽时间最大值（你要的“最大时间”）
    x_max = max(end_times) if len(end_times) else float(t_h[-1])

    # -----------------------------
    # 绘图
    # -----------------------------
    fig, ax = plt.subplots(figsize=(13, 7))

    for name, t_ext, soc_ext, t_end in extended:
        ax.plot(t_ext, soc_ext, zorder=3 if name == "full" else 2, **line_styles[name])

    # 关键电量线（每种阈值各一条线）
    ax.axhline(y=20, color="#F39C12", linestyle=":", alpha=0.70, linewidth=1.3, label="低电量警告 (20%)")
    ax.axhline(y=5,  color="#E74C3C", linestyle=":", alpha=0.70, linewidth=1.3, label="极低电量 (5%)")

    ax.set_xlabel("时间 (小时)", fontsize=12, fontweight="bold")
    ax.set_ylabel("电量 SOC (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, x_max)  # ✅ 关键修复：最大时间
    ax.grid(True, alpha=0.25, linestyle="-")

    ax.set_title("电池电量变化对比：各修正因子效果分析", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95, ncol=1)

    # -----------------------------
    # 统计信息框（用每条曲线的 t_end）
    # -----------------------------
    # 取出对应耗尽时间（方便展示）
    ttl_map = {name: t_end for (name, _, _, t_end) in extended}

    lines = [
        "对比分析",
        "─" * 16,
        f"全部修正: {ttl_map.get('full', np.nan):.2f} h",
    ]
    if "voltage" in ttl_map:
        lines.append(f"仅电压:   {ttl_map['voltage']:.2f} h")
    if "temp" in ttl_map:
        lines.append(f"仅温度:   {ttl_map['temp']:.2f} h")
    if "aging" in ttl_map:
        lines.append(f"仅老化:   {ttl_map['aging']:.2f} h")
    if "uncorrected" in ttl_map:
        lines.append(f"无修正:   {ttl_map['uncorrected']:.2f} h")

    stats_text = "\n".join(lines)

    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="bottom", horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.95, edgecolor="gray"),
        family="monospace"
    )

    plt.tight_layout()

    if save_path:
        smart_savefig(save_path)

    if show is None:
        show = get_show_plots()
    if show:
        plt.show()

    return fig

