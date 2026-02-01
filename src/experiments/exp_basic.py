# experiments/exp_basic.py
# 基础单次仿真实验模块
#
# 提供单次仿真实验功能：
# - 运行单次仿真
# - 输出结果摘要
# - 可视化仿真过程

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_simulation
from visualization import (
    plot_single_run,
    plot_comprehensive_dashboard,
    plot_soc_curve,
    plot_power_curve,
    plot_temperature_curve,
    plot_state_timeline,
)
from usage.scenario import *


def run_basic_experiment(
    scenario=None,
    scenario_name="默认场景",
    seed=42,
    dt=1.0,
    T_amb=298.15,
    verbose=True,
    visualize=True,
    dashboard=False,
    save_prefix=None,
):
    """
    运行基础单次仿真实验
    
    参数：
        scenario : dict - 使用场景配置
        scenario_name : str - 场景名称
        seed : int - 随机种子
        dt : float - 时间步长（秒）
        T_amb : float - 环境温度（K）
        verbose : bool - 是否输出详细信息
        visualize : bool - 是否可视化结果
        dashboard : bool - 是否显示综合仪表板
        save_prefix : str - 图片保存路径前缀
    
    返回：
        result : dict - 仿真结果
    """
    
    # 默认场景
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常混合场景"
    
    if verbose:
        print("=" * 60)
        print("🔋 基础仿真实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"随机种子: {seed}")
        print(f"时间步长: {dt} 秒")
        print(f"环境温度: {T_amb - 273.15:.1f} °C")
        print("-" * 60)
        print("正在运行仿真...")
    
    # 运行仿真
    result = run_simulation(
        scenario=scenario,
        dt=dt,
        T_amb=T_amb,
        seed=seed,
        record=True,
    )
    
    # 输出结果摘要
    ttl_hours = result["TTL"] / 3600
    
    if verbose:
        print("-" * 60)
        print("✅ 仿真完成！")
        print("-" * 60)
        print(f"📊 结果摘要:")
        print(f"   续航时间 (TTL): {ttl_hours:.2f} 小时")
        
        if "Power" in result:
            import numpy as np
            avg_power = np.mean(result["Power"])
            max_power = np.max(result["Power"])
            print(f"   平均功耗: {avg_power:.3f} W")
            print(f"   最大功耗: {max_power:.3f} W")
        
        if "Tb" in result:
            max_temp = max(result["Tb"]) - 273.15
            print(f"   最高温度: {max_temp:.1f} °C")
        
        if "State" in result:
            from collections import Counter
            state_counts = Counter(result["State"])
            total = sum(state_counts.values())
            print(f"   状态分布:")
            for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
                print(f"     - {state}: {count/total*100:.1f}%")
        
        print("=" * 60)
    
    # 可视化
    if visualize:
        if dashboard:
            # 综合仪表板
            save_path = f"{save_prefix}_dashboard.png" if save_prefix else None
            plot_comprehensive_dashboard(result, save_path=save_path, T_amb=T_amb)
        else:
            # 简单图表
            save_path = f"{save_prefix}_basic.png" if save_prefix else None
            plot_single_run(result, save_path=save_path)
    
    return result


def run_quick_demo():
    """
    快速演示基础仿真
    """
    print("\n" + "🚀 快速演示：基础仿真\n")
    
    # 测试几个不同场景
    scenarios = [
        (SCENARIO_STUDENT_DAILY_MIXED, "学生日常 (Mixed)"),
        (PURE_GAMING, "纯游戏"),
        (PURE_VIDEO, "纯视频"),
    ]
    
    for scenario, name in scenarios:
        result = run_basic_experiment(
            scenario=scenario,
            scenario_name=name,
            seed=42,
            visualize=False,
            verbose=True,
        )
    
    # 最后一个显示可视化
    print("\n显示最后一个场景的可视化...")
    run_basic_experiment(
        scenario=SCENARIO_STUDENT_DAILY_MIXED,
        scenario_name="学生日常",
        seed=42,
        visualize=True,
        dashboard=True,
        verbose=False,
    )


if __name__ == "__main__":
    run_quick_demo()
