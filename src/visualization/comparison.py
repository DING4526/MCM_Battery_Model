# visualization/comparison.py
# 场景对比可视化模块
#
# 提供多种场景对比可视化方式：
# - 场景对比柱状图
# - 多场景箱线图
# - 雷达图
# - 多场景时间线对比

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 从统一配置模块导入
from .config import (
    setup_style as _setup_style,
    COLORS,
    SCENARIO_COLORS,
    DEFAULT_COLORS,
    to_hours as _to_hours,
    save_figure,
    get_save_path,
)


def _get_color(scenario_name, index):
    """获取场景颜色"""
    if scenario_name in SCENARIO_COLORS:
        return SCENARIO_COLORS[scenario_name]
    return DEFAULT_COLORS[index % len(DEFAULT_COLORS)]


# =====================================================
# 场景对比可视化函数
# =====================================================

def plot_scenario_comparison(comparison_results, ax=None, show=True, save_path=None, 
                              metric="mean", error_bars=True):
    """
    绘制场景对比柱状图
    
    参数：
        comparison_results : dict - 对比结果
            {scenario_name: {"ttl_list": [...], "mean": float, "std": float, ...}}
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        metric : str - 使用的指标 ("mean", "median")
        error_bars : bool - 是否显示误差棒
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    scenarios = list(comparison_results.keys())
    
    # 获取指标值
    if metric == "mean":
        values = [comparison_results[s]["mean"] / 3600 for s in scenarios]
        errors = [comparison_results[s]["std"] / 3600 for s in scenarios] if error_bars else None
        ylabel = "平均续航时间 (小时)"
    else:
        values = [comparison_results[s]["median"] / 3600 for s in scenarios]
        q1 = [comparison_results[s]["q1"] / 3600 for s in scenarios]
        q3 = [comparison_results[s]["q3"] / 3600 for s in scenarios]
        errors = [[v - q1[i] for i, v in enumerate(values)],
                  [q3[i] - v for i, v in enumerate(values)]] if error_bars else None
        ylabel = "中位数续航时间 (小时)"
    
    # 颜色
    colors = [_get_color(s, i) for i, s in enumerate(scenarios)]
    
    # 绘制柱状图
    x = range(len(scenarios))
    bars = ax.bar(x, values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # 误差棒
    if error_bars and errors:
        ax.errorbar(x, values, yerr=errors, fmt='none', color='black', capsize=5, linewidth=1.5)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.2f}h', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title("不同使用场景续航时间对比", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加参考线（最佳和最差）
    max_val = max(values)
    min_val = min(values)
    ax.axhline(y=max_val, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=min_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_scenario_boxplot(comparison_results, ax=None, show=True, save_path=None):
    """
    绘制多场景箱线图对比
    
    参数：
        comparison_results : dict - 对比结果
            {scenario_name: {"ttl_list": [...], ...}}
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    scenarios = list(comparison_results.keys())
    data = [_to_hours(comparison_results[s]["ttl_list"]) for s in scenarios]
    colors = [_get_color(s, i) for i, s in enumerate(scenarios)]
    
    # 箱线图
    bp = ax.boxplot(data, patch_artist=True, labels=scenarios)
    
    # 设置颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    # 添加散点（小样本时）
    for i, (d, color) in enumerate(zip(data, colors)):
        if len(d) <= 100:
            jitter = np.random.normal(0, 0.04, size=len(d))
            ax.scatter(i + 1 + jitter, d, alpha=0.3, color=color, s=15)
    
    # 添加均值标记
    means = [np.mean(d) for d in data]
    ax.scatter(range(1, len(scenarios) + 1), means, color='red', marker='D', s=80, 
               zorder=5, label='均值')
    
    ax.set_xticklabels(scenarios, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel("续航时间 TTL (小时)", fontsize=11)
    ax.set_title("不同使用场景 TTL 分布对比", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_scenario_radar(comparison_results, metrics=None, ax=None, show=True, save_path=None):
    """
    绘制场景对比雷达图
    
    参数：
        comparison_results : dict - 对比结果
        metrics : list - 用于雷达图的指标列表
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    scenarios = list(comparison_results.keys())
    
    # 默认指标
    if metrics is None:
        metrics = ["mean", "std", "min", "max", "median"]
    
    metric_labels = {
        "mean": "平均值",
        "std": "标准差",
        "min": "最小值",
        "max": "最大值",
        "median": "中位数"
    }
    
    # 准备数据
    data = {}
    for s in scenarios:
        ttl_h = _to_hours(comparison_results[s]["ttl_list"])
        data[s] = {
            "mean": np.mean(ttl_h),
            "std": np.std(ttl_h),
            "min": np.min(ttl_h),
            "max": np.max(ttl_h),
            "median": np.median(ttl_h)
        }
    
    # 归一化数据（每个指标 0-1）
    normalized = {}
    for m in metrics:
        vals = [data[s][m] for s in scenarios]
        min_v, max_v = min(vals), max(vals)
        range_v = max_v - min_v if max_v != min_v else 1
        for s in scenarios:
            if s not in normalized:
                normalized[s] = []
            normalized[s].append((data[s][m] - min_v) / range_v)
    
    # 角度
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # 标签
    labels = [metric_labels.get(m, m) for m in metrics]
    
    # 创建极坐标图
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # 绘制每个场景
    for i, s in enumerate(scenarios):
        values = normalized[s] + [normalized[s][0]]
        color = _get_color(s, i)
        ax.fill(angles, values, color=color, alpha=0.1)
        ax.plot(angles, values, color=color, linewidth=2, marker='o', markersize=6, label=s)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("场景多维度对比雷达图\n（归一化指标）", fontsize=13, fontweight='bold', y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_multi_scenario_timeline(results_dict, ax=None, show=True, save_path=None):
    """
    绘制多场景时间线对比
    
    参数：
        results_dict : dict - 多场景仿真结果
            {scenario_name: result_dict}
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    scenarios = list(results_dict.keys())
    
    for i, (scenario, result) in enumerate(results_dict.items()):
        time_h = _to_hours(result["time"])
        soc_percent = [s * 100 for s in result["SOC"]]
        color = _get_color(scenario, i)
        
        ax.plot(time_h, soc_percent, color=color, linewidth=2, label=scenario)
    
    # 关键电量线
    ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label="低电量警告 (20%)")
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label="极低电量 (5%)")
    
    ax.set_xlabel("时间 (小时)", fontsize=11)
    ax.set_ylabel("电量 SOC (%)", fontsize=11)
    ax.set_title("多场景 SOC 变化曲线对比", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_scenario_comprehensive_comparison(comparison_results, results_dict=None, save_path=None):
    """
    绘制场景对比综合图表（比赛级别可视化）
    
    包含：
    - 柱状图对比
    - 箱线图分布
    - 时间线对比（如有）
    - 统计表格
    
    参数：
        comparison_results : dict - Monte Carlo 对比结果
        results_dict : dict - 单次仿真结果（可选）
        save_path : str - 保存路径
    """
    _setup_style()
    
    has_timeline = results_dict is not None
    
    if has_timeline:
        fig = plt.figure(figsize=(18, 14))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
    else:
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, (3, 4))
        ax4 = None
    
    # ===== 柱状图对比 =====
    plot_scenario_comparison(comparison_results, ax=ax1, show=False)
    
    # ===== 箱线图分布 =====
    plot_scenario_boxplot(comparison_results, ax=ax2, show=False)
    
    if has_timeline:
        # ===== 时间线对比 =====
        plot_multi_scenario_timeline(results_dict, ax=ax3, show=False)
        
        # ===== 统计表格 =====
        ax4.axis('off')
    else:
        ax3.axis('off')
    
    # 构建统计表格（使用纯ASCII边框，兼容性更好）
    scenarios = list(comparison_results.keys())
    
    table_text = """
    +===============================================================================+
    |                    [Compare] 场 景 对 比 统 计 表                            |
    +==================+===========+==========+==========+==========+==============+
    |      场景        |  均值(h)  | 标准差(h)|  最小(h) |  最大(h) | 相对基准(%)  |
    +==================+===========+==========+==========+==========+==============+
"""
    
    # 基准（第一个场景）
    baseline_mean = comparison_results[scenarios[0]]["mean"]
    
    for s in scenarios:
        mean_h = comparison_results[s]["mean"] / 3600
        std_h = comparison_results[s]["std"] / 3600
        min_h = comparison_results[s]["min"] / 3600
        max_h = comparison_results[s]["max"] / 3600
        relative = (comparison_results[s]["mean"] / baseline_mean - 1) * 100
        
        # 根据相对变化添加符号
        rel_str = f"+{relative:.1f}%" if relative > 0 else f"{relative:.1f}%"
        
        table_text += f"    | {s:<16} | {mean_h:>9.2f} | {std_h:>8.2f} | {min_h:>8.2f} | {max_h:>8.2f} | {rel_str:>12} |\n"
    
    table_text += """    +==================+===========+==========+==========+==========+==============+
"""
    
    # 添加洞察
    best_scenario = max(scenarios, key=lambda s: comparison_results[s]["mean"])
    worst_scenario = min(scenarios, key=lambda s: comparison_results[s]["mean"])
    
    insights = f"""
    
    [Key] 关键洞察:
    
    * 最佳续航场景: {best_scenario} ({comparison_results[best_scenario]["mean"]/3600:.2f} 小时)
    * 最差续航场景: {worst_scenario} ({comparison_results[worst_scenario]["mean"]/3600:.2f} 小时)
    * 最大续航差异: {(comparison_results[best_scenario]["mean"] - comparison_results[worst_scenario]["mean"])/3600:.2f} 小时
    """
    
    table_text += insights
    
    target_ax = ax4 if has_timeline else ax3
    target_ax.text(0.05, 0.5, table_text, transform=target_ax.transAxes, fontsize=9,
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 总标题
    fig.suptitle("[Scenario] 使用场景对比分析报告", fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
