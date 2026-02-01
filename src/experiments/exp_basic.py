# experiments/exp_basic.py
# 基础单次仿真实验模块

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_simulation
from visualization import plot_single_run
from visualization.config import smart_savefig
from usage.scenario import *


def run_basic_experiment(
    scenario=None,
    scenario_name="默认场景",
    seed=42,
    dt=1.0,
    T_amb=298.15,
    verbose=True,
    output_dir="basic",
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
        output_dir : str - 输出子目录名
    
    返回：
        result : dict - 仿真结果
    """
    
    if scenario is None:
        scenario = SCENARIO_STUDENT_DAILY_MIXED
        scenario_name = "学生日常"
    
    if verbose:
        print("=" * 60)
        print("🔋 基础仿真实验")
        print("=" * 60)
        print(f"场景: {scenario_name}")
        print(f"随机种子: {seed}")
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
    
    ttl_hours = result["TTL"] / 3600
    
    if verbose:
        print("-" * 60)
        print("✅ 仿真完成！")
        print(f"   续航时间: {ttl_hours:.2f} 小时")
        
        if "Power" in result:
            import numpy as np
            print(f"   平均功耗: {np.mean(result['Power']):.3f} W")
        
        if "Tb" in result:
            print(f"   最高温度: {max(result['Tb']) - 273.15:.1f} °C")
        
        print("=" * 60)
    
    # 保存图片
    plot_single_run(result, filename="soc_power.png", subdir=output_dir, show=False)
    
    return result


if __name__ == "__main__":
    run_basic_experiment()
