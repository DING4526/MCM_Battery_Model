# experiments/exp_monte_carlo.py
# Monte Carlo 仿真实验模块

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_monte_carlo
from visualization import plot_ttl_distribution, plot_ttl_statistical_summary
from visualization.config import smart_savefig
from usage.scenario import *


def run_monte_carlo_experiment(
    scenario=None,
    scenario_name="默认场景",
    n_samples=100,
    base_seed=0,
    dt=1.0,
    T_amb=298.15,
    verbose=True,
    output_dir="monte_carlo",
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
        output_dir : str - 输出子目录名
    
    返回：
        results : dict - 仿真结果
    """
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常"
    
    if verbose:
        print("=" * 60)
        print("🎲 Monte Carlo 仿真实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"仿真次数: {n_samples}")
        print(f"环境温度: {T_amb - 273.15:.1f} °C")
        print("-" * 60)
        print("正在运行仿真...")
    
    # 运行 Monte Carlo 仿真
    ttl_list = run_monte_carlo(
        scenario=scenario,
        n_samples=n_samples,
        base_seed=base_seed,
        dt=dt,
        T_amb=T_amb,
    )
    
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
    
    if verbose:
        print("-" * 60)
        print("✅ 仿真完成！")
        print(f"   平均 TTL: {results['mean']/3600:.2f} 小时")
        print(f"   标准差: {results['std']/3600:.3f} 小时")
        print(f"   范围: [{results['min']/3600:.2f}, {results['max']/3600:.2f}] 小时")
        print("=" * 60)
    
    # 保存图片
    plot_ttl_distribution(ttl_list, filename="ttl_distribution.png", subdir=output_dir, show=False)
    plot_ttl_statistical_summary(ttl_list, filename="ttl_summary.png", subdir=output_dir, show=False)
    
    return results


def run_convergence_analysis(
    scenario=None,
    scenario_name="默认场景",
    max_samples=500,
    step=50,
    base_seed=0,
    verbose=True,
    output_dir="monte_carlo",
):
    """运行收敛性分析"""
    import matplotlib.pyplot as plt
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
    
    if verbose:
        print("📈 Monte Carlo 收敛性分析")
        print(f"最大样本数: {max_samples}")
    
    sample_sizes = list(range(step, max_samples + 1, step))
    means = []
    stds = []
    
    all_ttl = run_monte_carlo(scenario=scenario, n_samples=max_samples, base_seed=base_seed)
    
    for n in sample_sizes:
        ttl_subset = all_ttl[:n]
        means.append(np.mean(ttl_subset) / 3600)
        stds.append(np.std(ttl_subset) / 3600)
    
    # 绘制收敛图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(sample_sizes, means, 'b-', linewidth=2)
    axes[0].set_xlabel("样本数")
    axes[0].set_ylabel("TTL 均值 (小时)")
    axes[0].set_title("均值收敛性")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(sample_sizes, stds, 'g-', linewidth=2)
    axes[1].set_xlabel("样本数")
    axes[1].set_ylabel("TTL 标准差 (小时)")
    axes[1].set_title("标准差稳定性")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    smart_savefig("convergence.png", output_dir)
    
    return {"sample_sizes": sample_sizes, "means": means, "stds": stds}


if __name__ == "__main__":
    run_monte_carlo_experiment()
