# visualization/sensitivity_plot.py
# 敏感度分析可视化模块
#
# 提供多种敏感度可视化方式：
# - 柱状图
# - 龙卷风图
# - 蜘蛛图
# - 热力图

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 从统一配置模块导入
from .config import (
    setup_style as _setup_style,
    COLORS,
    PARAM_LABELS,
    save_figure,
    get_save_path,
    smart_savefig,
    get_show_plots,
)


def _get_label(param):
    """获取参数的中文标签"""
    return PARAM_LABELS.get(param, param)


# =====================================================
# 敏感度可视化函数
# =====================================================

def plot_sensitivity_bar(sens_results, filename=None, subdir="", ax=None, show=None, save_path=None, 
                         normalized=True, sort=True):
    """
    绘制敏感度柱状图
    
    参数：
        sens_results : dict - 敏感度分析结果
        filename : str - 保存文件名
        subdir : str - 输出子目录
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 兼容旧接口
        normalized : bool - 是否使用归一化敏感度
        sort : bool - 是否按敏感度排序
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 准备数据（排除内部字段）
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
    
    # 绘制柱状图
    bars = ax.barh(range(len(params)), values, color=colors, alpha=0.8, edgecolor='white')
    
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(ylabel, fontsize=11)
    ax.set_title("参数敏感度分析", fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if filename:
        smart_savefig(filename, subdir)
    elif save_path:
        smart_savefig(save_path)
    
    if show is None:
        show = get_show_plots()
    if show:
        plt.show()
    
    return ax


def plot_sensitivity_tornado(sens_results, baseline_ttl, ax=None, show=True, 
                              save_path=None, sort=True):
    """
    绘制敏感度龙卷风图
    
    参数：
        sens_results : dict - 敏感度分析结果
        baseline_ttl : float - 基准 TTL（秒）
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        sort : bool - 是否按影响大小排序
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    baseline_h = baseline_ttl / 3600
    
    # 准备数据
    params = list(sens_results.keys())
    ttl_plus = [sens_results[p]["TTL+"] / 3600 for p in params]
    ttl_minus = [sens_results[p]["TTL-"] / 3600 for p in params]
    labels = [_get_label(p) for p in params]
    
    # 计算影响范围
    ranges = [abs(ttl_plus[i] - ttl_minus[i]) for i in range(len(params))]
    
    # 排序
    if sort:
        sorted_indices = np.argsort(ranges)[::-1]
        params = [params[i] for i in sorted_indices]
        ttl_plus = [ttl_plus[i] for i in sorted_indices]
        ttl_minus = [ttl_minus[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    y_pos = range(len(params))
    
    # 绘制龙卷风图
    for i, (low, high) in enumerate(zip(ttl_minus, ttl_plus)):
        # 左侧（负扰动）
        ax.barh(i, low - baseline_h, left=baseline_h, color=COLORS["primary"], 
                alpha=0.8, height=0.6, label='参数 -20%' if i == 0 else "")
        # 右侧（正扰动）
        ax.barh(i, high - baseline_h, left=baseline_h, color=COLORS["accent"], 
                alpha=0.8, height=0.6, label='参数 +20%' if i == 0 else "")
    
    # 基准线
    ax.axvline(x=baseline_h, color='black', linewidth=2, linestyle='--', 
               label=f'基准值: {baseline_h:.2f} h')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("续航时间 TTL (小时)", fontsize=11)
    ax.set_title("参数敏感度龙卷风图", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_sensitivity_spider(sens_results, ax=None, show=True, save_path=None):
    """
    绘制敏感度蜘蛛图（雷达图）
    
    参数：
        sens_results : dict - 敏感度分析结果
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    # 准备数据
    params = list(sens_results.keys())
    values = [abs(sens_results[p]["S_norm"]) for p in params]
    labels = [_get_label(p) for p in params]
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
    
    # 闭合图形
    values = values + [values[0]]
    angles = angles + [angles[0]]
    labels = labels + [labels[0]]
    
    # 创建极坐标图
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 绘制蜘蛛图
    ax.fill(angles, values, color=COLORS["primary"], alpha=0.25)
    ax.plot(angles, values, color=COLORS["primary"], linewidth=2, marker='o', markersize=8)
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)
    
    ax.set_title("参数敏感度蜘蛛图\n（归一化敏感度绝对值）", fontsize=13, fontweight='bold', y=1.1)
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_sensitivity_heatmap(sens_results, ax=None, show=True, save_path=None):
    """
    绘制敏感度热力图
    
    参数：
        sens_results : dict - 敏感度分析结果
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # 准备数据
    params = list(sens_results.keys())
    labels = [_get_label(p) for p in params]
    
    # 创建数据矩阵
    data = np.zeros((len(params), 3))
    for i, p in enumerate(params):
        data[i, 0] = sens_results[p]["TTL-"] / 3600
        data[i, 1] = (sens_results[p]["TTL+"] + sens_results[p]["TTL-"]) / 2 / 3600  # 基准近似
        data[i, 2] = sens_results[p]["TTL+"] / 3600
    
    # 绘制热力图
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')
    
    # 设置标签
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(labels)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['参数 -20%', '基准', '参数 +20%'])
    
    # 添加数值标注
    for i in range(len(params)):
        for j in range(3):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('续航时间 TTL (小时)', fontsize=10)
    
    ax.set_title("参数敏感度热力图", fontsize=13, fontweight='bold')
    
    if save_path:
        smart_savefig(save_path)
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_sensitivity_comprehensive(sens_results, baseline_ttl, filename=None, subdir="", save_path=None, show=None):
    """
    绘制敏感度分析综合图表
    
    参数：
        sens_results : dict - 敏感度分析结果
        baseline_ttl : float - 基准 TTL（秒）
        filename : str - 保存文件名
        subdir : str - 输出子目录
        save_path : str - 兼容旧接口
        show : bool - 是否显示图形
    """
    _setup_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # ===== 左上：归一化敏感度柱状图 =====
    ax1 = fig.add_subplot(2, 2, 1)
    plot_sensitivity_bar(sens_results, ax=ax1, show=False, normalized=True)
    
    # ===== 右上：龙卷风图 =====
    ax2 = fig.add_subplot(2, 2, 2)
    plot_sensitivity_tornado(sens_results, baseline_ttl, ax=ax2, show=False)
    
    # ===== 左下：蜘蛛图 =====
    ax3 = fig.add_subplot(2, 2, 3, polar=True)
    plot_sensitivity_spider(sens_results, ax=ax3, show=False)
    
    # ===== 右下：统计面板 =====
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # 计算关键洞察
    params = list(sens_results.keys())
    s_norms = {p: sens_results[p]["S_norm"] for p in params}
    
    # 找出最敏感的参数
    most_sensitive = max(params, key=lambda p: abs(s_norms[p]))
    least_sensitive = min(params, key=lambda p: abs(s_norms[p]))
    
    # 正负敏感参数
    positive_sens = [p for p in params if s_norms[p] > 0]
    negative_sens = [p for p in params if s_norms[p] < 0]
    
    # 使用纯ASCII边框，兼容性更好
    insights_text_parts = [f"""
    +==================================================+
    |         [Sens] 敏 感 度 分 析 洞 察              |
    +==================================================+
    |                                                  |
    |  基准续航时间:  {baseline_ttl/3600:>8.2f} 小时                 |
    |                                                  |
    |  ----------- 关键发现 -----------                |
    |                                                  |
    |  [!] 最敏感参数:  {_get_label(most_sensitive):<15}             |
    |      敏感度: {s_norms[most_sensitive]:>8.4f}                       |
    |                                                  |
    |  [o] 最不敏感参数: {_get_label(least_sensitive):<15}            |
    |      敏感度: {s_norms[least_sensitive]:>8.4f}                       |
    |                                                  |
    |  ----------- 参数分类 -----------                |
    |                                                  |
    |  负敏感度（增加->减少TTL）:                       |
    """]
    
    # 使用列表收集字符串，避免循环中字符串拼接
    for p in negative_sens:
        insights_text_parts.append(f"|    - {_get_label(p):<20} ({s_norms[p]:.4f})      |\n")
    
    insights_text_parts.append("""|                                                  |
|  正敏感度（增加->增加TTL）:                       |
""")
    
    for p in positive_sens:
        insights_text_parts.append(f"|    - {_get_label(p):<20} ({s_norms[p]:.4f})      |\n")
    
    insights_text_parts.append("""|                                                  |
+==================================================+
""")
    
    insights_text = ''.join(insights_text_parts)
    
    ax4.text(0.05, 0.5, insights_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 总标题
    fig.suptitle("[Sensitivity] 参数敏感度分析综合报告", fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if filename:
        smart_savefig(filename, subdir)
    elif save_path:
        smart_savefig(save_path)
    
    if show is None:
        show = get_show_plots()
    if show:
        plt.show()
    
    return fig
