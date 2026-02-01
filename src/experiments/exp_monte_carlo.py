# experiments/exp_monte_carlo.py
# Monte Carlo 仿真实验模块
#
# 提供 Monte Carlo 仿真实验功能：
# - 批量随机仿真
# - 统计分析
# - 分布可视化

import sys
import os
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_monte_carlo
from visualization import (
    plot_ttl_distribution,
    plot_ttl_boxplot,
    plot_ttl_violin,
    plot_ttl_kde,
    plot_ttl_statistical_summary,
)
from usage.scenario import *


def run_monte_carlo_experiment(
    scenario=None,
    scenario_name="默认场景",
    n_samples=100,
    base_seed=0,
    dt=1.0,
    T_amb=298.15,
    verbose=True,
    visualize=True,
    summary_plot=False,
    save_prefix=None,
):
    """
    运行 Monte Carlo 仿真实验
    
    参数：
        scenario : dict - 使用场景配置
        scenario_name : str - 场景名称
        n_samples : int - 仿真次数
        base_seed : int - 基础随机种子
        dt : float - 时间步长（秒）
        T_amb : float - 环境温度（K）
        verbose : bool - 是否输出详细信息
        visualize : bool - 是否可视化结果
        summary_plot : bool - 是否显示综合统计图
        save_prefix : str - 图片保存路径前缀
    
    返回：
        results : dict - 仿真结果
            {
                "ttl_list": [...],
                "mean": float,
                "std": float,
                "min": float,
                "max": float,
                "median": float,
                "q1": float,
                "q3": float,
            }
    """
    
    # 默认场景
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常混合场景"
    
    if verbose:
        print("=" * 60)
        print("🎲 Monte Carlo 仿真实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"仿真次数: {n_samples}")
        print(f"基础种子: {base_seed}")
        print(f"时间步长: {dt} 秒")
        print(f"环境温度: {T_amb - 273.15:.1f} °C")
        print("-" * 60)
        print("正在运行 Monte Carlo 仿真...")
    
    # 运行 Monte Carlo 仿真
    ttl_list = run_monte_carlo(
        scenario=scenario,
        n_samples=n_samples,
        base_seed=base_seed,
        dt=dt,
        T_amb=T_amb,
    )
    
    # 转换为小时用于统计
    ttl_hours = [t / 3600 for t in ttl_list]
    
    # 计算统计量
    results = {
        "ttl_list": ttl_list,
        "mean": np.mean(ttl_list),
        "std": np.std(ttl_list),
        "min": np.min(ttl_list),
        "max": np.max(ttl_list),
        "median": np.median(ttl_list),
        "q1": np.percentile(ttl_list, 25),
        "q3": np.percentile(ttl_list, 75),
    }
    
    # 输出结果摘要
    if verbose:
        print("-" * 60)
        print("✅ Monte Carlo 仿真完成！")
        print("-" * 60)
        print(f"📊 统计摘要:")
        print(f"   样本数: {n_samples}")
        print(f"   平均 TTL: {results['mean']/3600:.2f} 小时")
        print(f"   标准差: {results['std']/3600:.3f} 小时")
        print(f"   最小 TTL: {results['min']/3600:.2f} 小时")
        print(f"   最大 TTL: {results['max']/3600:.2f} 小时")
        print(f"   中位数: {results['median']/3600:.2f} 小时")
        print(f"   25% 分位: {results['q1']/3600:.2f} 小时")
        print(f"   75% 分位: {results['q3']/3600:.2f} 小时")
        
        # 95% 置信区间
        ci_low = results['mean'] - 1.96 * results['std'] / np.sqrt(n_samples)
        ci_high = results['mean'] + 1.96 * results['std'] / np.sqrt(n_samples)
        print(f"   95% 置信区间: [{ci_low/3600:.3f}, {ci_high/3600:.3f}] 小时")
        print("=" * 60)
    
    # 可视化
    if visualize:
        if summary_plot:
            # 综合统计图
            save_path = f"{save_prefix}_mc_summary.png" if save_prefix else None
            plot_ttl_statistical_summary(ttl_list, save_path=save_path)
        else:
            # 简单直方图
            save_path = f"{save_prefix}_mc_hist.png" if save_prefix else None
            plot_ttl_distribution(ttl_list, save_path=save_path)
    
    return results


def run_convergence_analysis(
    scenario=None,
    scenario_name="默认场景",
    max_samples=500,
    step=50,
    base_seed=0,
    verbose=True,
    visualize=True,
):
    """
    运行收敛性分析（分析不同样本量对结果的影响）
    
    参数：
        scenario : dict - 使用场景配置
        scenario_name : str - 场景名称
        max_samples : int - 最大样本数
        step : int - 样本数步长
        base_seed : int - 基础随机种子
        verbose : bool - 是否输出详细信息
        visualize : bool - 是否可视化结果
    
    返回：
        convergence_results : dict - 收敛分析结果
    """
    import matplotlib.pyplot as plt
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常混合场景"
    
    if verbose:
        print("=" * 60)
        print("📈 Monte Carlo 收敛性分析")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"最大样本数: {max_samples}")
        print("-" * 60)
    
    sample_sizes = list(range(step, max_samples + 1, step))
    means = []
    stds = []
    ci_lows = []
    ci_highs = []
    
    # 一次性运行最大样本
    all_ttl = run_monte_carlo(
        scenario=scenario,
        n_samples=max_samples,
        base_seed=base_seed,
    )
    
    for n in sample_sizes:
        ttl_subset = all_ttl[:n]
        mean = np.mean(ttl_subset)
        std = np.std(ttl_subset)
        
        means.append(mean / 3600)
        stds.append(std / 3600)
        ci_lows.append((mean - 1.96 * std / np.sqrt(n)) / 3600)
        ci_highs.append((mean + 1.96 * std / np.sqrt(n)) / 3600)
        
        if verbose:
            print(f"  n={n:4d}: 均值={mean/3600:.3f}h, 标准差={std/3600:.4f}h")
    
    convergence_results = {
        "sample_sizes": sample_sizes,
        "means": means,
        "stds": stds,
        "ci_lows": ci_lows,
        "ci_highs": ci_highs,
    }
    
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 均值收敛图
        ax1 = axes[0]
        ax1.plot(sample_sizes, means, 'b-', linewidth=2, label='均值')
        ax1.fill_between(sample_sizes, ci_lows, ci_highs, alpha=0.3, color='blue', label='95% CI')
        ax1.axhline(y=means[-1], color='red', linestyle='--', alpha=0.7, label=f'最终均值: {means[-1]:.3f}h')
        ax1.set_xlabel("样本数")
        ax1.set_ylabel("TTL 均值 (小时)")
        ax1.set_title("均值收敛性")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 标准差收敛图
        ax2 = axes[1]
        ax2.plot(sample_sizes, stds, 'g-', linewidth=2, label='标准差')
        ax2.set_xlabel("样本数")
        ax2.set_ylabel("TTL 标准差 (小时)")
        ax2.set_title("标准差稳定性")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f"Monte Carlo 收敛性分析 - {scenario_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    if verbose:
        print("=" * 60)
    
    return convergence_results


def run_quick_demo():
    """
    快速演示 Monte Carlo 仿真
    """
    print("\n" + "🚀 快速演示：Monte Carlo 仿真\n")
    
    # 基础 Monte Carlo
    results = run_monte_carlo_experiment(
        scenario=SCENARIO_STUDENT_DAILY_MIXED,
        scenario_name="学生日常",
        n_samples=200,
        verbose=True,
        visualize=True,
        summary_plot=True,
    )
    
    print("\n进行收敛性分析...")
    convergence_analysis = run_convergence_analysis(
        scenario=SCENARIO_STUDENT_DAILY_MIXED,
        scenario_name="学生日常",
        max_samples=300,
        step=30,
        verbose=True,
        visualize=True,
    )


if __name__ == "__main__":
    run_quick_demo()
