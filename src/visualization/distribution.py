# visualization/distribution.py
# TTL 分布可视化模块
#
# 提供多种分布可视化方式：
# - 直方图
# - 箱线图
# - 小提琴图
# - 核密度估计
# - 综合统计摘要图

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# =====================================================
# 配色方案
# =====================================================

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "neutral": "#6C757D",
}


def _setup_style():
    """设置全局绘图样式"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['figure.dpi'] = 100


def _to_hours(ttl_list):
    """将秒转换为小时"""
    return [t / 3600 for t in ttl_list]


# =====================================================
# 基础分布可视化
# =====================================================

def plot_ttl_distribution(ttl_list, ax=None, show=True, save_path=None, bins=20):
    """
    绘制 TTL 分布直方图
    
    参数：
        ttl_list : list - TTL 列表（秒）
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        bins : int - 直方图区间数
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ttl_h = _to_hours(ttl_list)
    
    # 绘制直方图
    n, bins_edges, patches = ax.hist(ttl_h, bins=bins, edgecolor='white', 
                                      color=COLORS["primary"], alpha=0.7)
    
    # 添加核密度估计曲线
    kde = stats.gaussian_kde(ttl_h)
    x_range = np.linspace(min(ttl_h), max(ttl_h), 100)
    ax.plot(x_range, kde(x_range) * len(ttl_h) * (bins_edges[1] - bins_edges[0]), 
            color=COLORS["secondary"], linewidth=2, label="核密度估计")
    
    # 添加统计线
    mean_ttl = np.mean(ttl_h)
    median_ttl = np.median(ttl_h)
    std_ttl = np.std(ttl_h)
    
    ax.axvline(x=mean_ttl, color=COLORS["accent"], linestyle='--', linewidth=2, 
               label=f"均值: {mean_ttl:.2f} h")
    ax.axvline(x=median_ttl, color=COLORS["success"], linestyle=':', linewidth=2, 
               label=f"中位数: {median_ttl:.2f} h")
    
    # 标注置信区间
    ci_low = mean_ttl - 1.96 * std_ttl / np.sqrt(len(ttl_h))
    ci_high = mean_ttl + 1.96 * std_ttl / np.sqrt(len(ttl_h))
    ax.axvspan(ci_low, ci_high, alpha=0.2, color=COLORS["accent"], 
               label=f"95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    
    ax.set_xlabel("续航时间 TTL (小时)", fontsize=11)
    ax.set_ylabel("频数", fontsize=11)
    ax.set_title("Monte Carlo 仿真 TTL 分布", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_ttl_boxplot(ttl_list, ax=None, show=True, save_path=None, label="TTL"):
    """
    绘制 TTL 箱线图
    
    参数：
        ttl_list : list - TTL 列表（秒）
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        label : str - 标签名称
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ttl_h = _to_hours(ttl_list)
    
    # 箱线图
    bp = ax.boxplot(ttl_h, patch_artist=True, labels=[label])
    
    # 设置颜色
    bp['boxes'][0].set_facecolor(COLORS["primary"])
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color(COLORS["accent"])
    bp['medians'][0].set_linewidth(2)
    
    # 添加散点（抖动）
    jitter = np.random.normal(0, 0.04, size=len(ttl_h))
    ax.scatter(1 + jitter, ttl_h, alpha=0.3, color=COLORS["secondary"], s=20)
    
    # 添加统计标注
    mean_val = np.mean(ttl_h)
    ax.scatter([1], [mean_val], color=COLORS["success"], marker='D', s=100, 
               zorder=5, label=f"均值: {mean_val:.2f} h")
    
    ax.set_ylabel("续航时间 TTL (小时)", fontsize=11)
    ax.set_title("TTL 分布箱线图", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_ttl_violin(ttl_list, ax=None, show=True, save_path=None, label="TTL"):
    """
    绘制 TTL 小提琴图
    
    参数：
        ttl_list : list - TTL 列表（秒）
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        label : str - 标签名称
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ttl_h = _to_hours(ttl_list)
    
    # 小提琴图
    parts = ax.violinplot(ttl_h, positions=[1], showmeans=True, showmedians=True)
    
    # 设置颜色
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS["primary"])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color(COLORS["accent"])
    parts['cmedians'].set_color(COLORS["secondary"])
    
    # 添加散点
    jitter = np.random.normal(0, 0.04, size=len(ttl_h))
    ax.scatter(1 + jitter, ttl_h, alpha=0.3, color=COLORS["neutral"], s=15)
    
    ax.set_xticks([1])
    ax.set_xticklabels([label])
    ax.set_ylabel("续航时间 TTL (小时)", fontsize=11)
    ax.set_title("TTL 分布小提琴图", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_ttl_kde(ttl_list, ax=None, show=True, save_path=None, fill=True):
    """
    绘制 TTL 核密度估计图
    
    参数：
        ttl_list : list - TTL 列表（秒）
        ax : matplotlib.axes.Axes - 可选的绑定轴
        show : bool - 是否显示图形
        save_path : str - 保存路径
        fill : bool - 是否填充曲线下区域
    """
    _setup_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    ttl_h = _to_hours(ttl_list)
    
    # 核密度估计
    kde = stats.gaussian_kde(ttl_h)
    x_range = np.linspace(min(ttl_h) - 0.5, max(ttl_h) + 0.5, 200)
    density = kde(x_range)
    
    # 绘制 KDE 曲线
    ax.plot(x_range, density, color=COLORS["primary"], linewidth=2.5, label="核密度估计")
    
    if fill:
        ax.fill_between(x_range, density, alpha=0.3, color=COLORS["primary"])
    
    # 添加数据点（rug plot）
    ax.scatter(ttl_h, np.zeros_like(ttl_h) - 0.01 * max(density), 
               alpha=0.5, color=COLORS["secondary"], s=10, marker='|')
    
    # 添加统计信息
    mean_ttl = np.mean(ttl_h)
    std_ttl = np.std(ttl_h)
    
    ax.axvline(x=mean_ttl, color=COLORS["accent"], linestyle='--', linewidth=2, 
               label=f"均值: {mean_ttl:.2f} h")
    ax.axvline(x=mean_ttl - std_ttl, color=COLORS["neutral"], linestyle=':', linewidth=1.5, 
               label=f"±1σ: [{mean_ttl-std_ttl:.2f}, {mean_ttl+std_ttl:.2f}]")
    ax.axvline(x=mean_ttl + std_ttl, color=COLORS["neutral"], linestyle=':', linewidth=1.5)
    
    ax.set_xlabel("续航时间 TTL (小时)", fontsize=11)
    ax.set_ylabel("概率密度", fontsize=11)
    ax.set_title("TTL 核密度估计分布", fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.tight_layout()
        plt.show()
    
    return ax


def plot_ttl_statistical_summary(ttl_list, save_path=None):
    """
    绘制 TTL 综合统计摘要图（比赛级别可视化）
    
    包含：
    - 直方图 + KDE
    - 箱线图
    - 统计信息面板
    - QQ 图（正态性检验）
    
    参数：
        ttl_list : list - TTL 列表（秒）
        save_path : str - 保存路径
    """
    _setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ttl_h = _to_hours(ttl_list)
    
    # ===== 左上：直方图 + KDE =====
    ax1 = axes[0, 0]
    n, bins_edges, patches = ax1.hist(ttl_h, bins=20, edgecolor='white', 
                                       color=COLORS["primary"], alpha=0.7, density=True)
    kde = stats.gaussian_kde(ttl_h)
    x_range = np.linspace(min(ttl_h), max(ttl_h), 100)
    ax1.plot(x_range, kde(x_range), color=COLORS["secondary"], linewidth=2)
    ax1.set_xlabel("续航时间 TTL (小时)")
    ax1.set_ylabel("概率密度")
    ax1.set_title("TTL 分布直方图", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ===== 右上：箱线图 + 散点 =====
    ax2 = axes[0, 1]
    bp = ax2.boxplot(ttl_h, patch_artist=True, vert=True, positions=[1])
    bp['boxes'][0].set_facecolor(COLORS["primary"])
    bp['boxes'][0].set_alpha(0.7)
    bp['medians'][0].set_color(COLORS["accent"])
    bp['medians'][0].set_linewidth(2)
    
    jitter = np.random.normal(0, 0.04, size=len(ttl_h))
    ax2.scatter(1 + jitter, ttl_h, alpha=0.3, color=COLORS["secondary"], s=20)
    ax2.set_xticks([1])
    ax2.set_xticklabels(["TTL"])
    ax2.set_ylabel("续航时间 (小时)")
    ax2.set_title("TTL 箱线图", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ===== 左下：QQ 图 =====
    ax3 = axes[1, 0]
    stats.probplot(ttl_h, dist="norm", plot=ax3)
    ax3.set_title("Q-Q 图（正态性检验）", fontweight='bold')
    ax3.get_lines()[0].set_markerfacecolor(COLORS["primary"])
    ax3.get_lines()[0].set_alpha(0.6)
    ax3.get_lines()[1].set_color(COLORS["accent"])
    ax3.grid(True, alpha=0.3)
    
    # ===== 右下：统计信息面板 =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计量
    n_samples = len(ttl_h)
    mean_val = np.mean(ttl_h)
    std_val = np.std(ttl_h)
    min_val = np.min(ttl_h)
    max_val = np.max(ttl_h)
    median_val = np.median(ttl_h)
    q1 = np.percentile(ttl_h, 25)
    q3 = np.percentile(ttl_h, 75)
    iqr = q3 - q1
    skewness = stats.skew(ttl_h)
    kurtosis = stats.kurtosis(ttl_h)
    
    # Shapiro-Wilk 正态性检验
    if n_samples <= 5000:
        _, p_value = stats.shapiro(ttl_h)
    else:
        _, p_value = stats.normaltest(ttl_h)
    
    # 95% 置信区间
    ci_low = mean_val - 1.96 * std_val / np.sqrt(n_samples)
    ci_high = mean_val + 1.96 * std_val / np.sqrt(n_samples)
    
    stats_text = f"""
    ╔══════════════════════════════════════════╗
    ║         📊 统 计 分 析 报 告           ║
    ╠══════════════════════════════════════════╣
    ║  样本数量:          {n_samples:>8}              ║
    ║                                          ║
    ║  ─────── 中心趋势 ───────                ║
    ║  均值 (Mean):        {mean_val:>8.3f} h          ║
    ║  中位数 (Median):    {median_val:>8.3f} h          ║
    ║  95% 置信区间:  [{ci_low:.3f}, {ci_high:.3f}] h   ║
    ║                                          ║
    ║  ─────── 离散程度 ───────                ║
    ║  标准差 (Std):       {std_val:>8.3f} h          ║
    ║  变异系数 (CV):      {std_val/mean_val*100:>8.2f} %          ║
    ║  四分位距 (IQR):     {iqr:>8.3f} h          ║
    ║                                          ║
    ║  ─────── 范围 ───────                    ║
    ║  最小值 (Min):       {min_val:>8.3f} h          ║
    ║  最大值 (Max):       {max_val:>8.3f} h          ║
    ║  Q1 (25%):           {q1:>8.3f} h          ║
    ║  Q3 (75%):           {q3:>8.3f} h          ║
    ║                                          ║
    ║  ─────── 分布形态 ───────                ║
    ║  偏度 (Skewness):    {skewness:>8.3f}            ║
    ║  峰度 (Kurtosis):    {kurtosis:>8.3f}            ║
    ║  正态性检验 p值:     {p_value:>8.4f}            ║
    ╚══════════════════════════════════════════╝
    """
    
    ax4.text(0.05, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 总标题
    fig.suptitle(f"🔋 Monte Carlo 仿真 TTL 统计分析 (n={n_samples})", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig
