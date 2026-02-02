# experiments/exp_basic.py
# 基础单次仿真实验模块（Plotly 版本）

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import run_simulation
from visualization.timeseries import (
    plot_soc_curve,
    plot_power_curve,
    plot_temperature_curve,
    plot_state_timeline,
    plot_composite_power_temperature,
    plot_soc_comparison,
)
from visualization.config import save_plotly_figure, get_output_dir
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
    
    # 运行仿真（启用子模块功耗分解和无修正 SOC 记录）
    result = run_simulation(
        scenario=scenario,
        dt=dt,
        T_amb=T_amb,
        seed=seed,
        record=True,
        record_breakdown=True,
        record_uncorrected=True,
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
    
    # 保存图表（Plotly 版本）
    if verbose:
        print("保存图表...")
    
    # 1. SOC 曲线
    fig = plot_soc_curve(result, show=False)
    save_plotly_figure(fig, "soc_curve", output_dir, size_type="wide")
    
    # 2. 功耗曲线
    fig = plot_power_curve(result, show=False)
    save_plotly_figure(fig, "power_curve", output_dir, size_type="wide")
    
    # 3. 温度曲线
    fig = plot_temperature_curve(result, T_amb=T_amb, show=False)
    save_plotly_figure(fig, "temperature_curve", output_dir, size_type="wide")
    
    # 4. 状态时间线
    fig = plot_state_timeline(result, show=False)
    save_plotly_figure(fig, "state_timeline", output_dir, size_type="timeline")
    
    # 5. 复合图表：温度 + 功耗堆叠图 + 状态时间线
    fig = plot_composite_power_temperature(result, T_amb=T_amb, show=False)
    save_plotly_figure(fig, "composite_power_temperature", output_dir, size_type="composite")
    
    # 6. SOC 对比（各修正因子效果）
    fig = plot_soc_comparison(result, show=False)
    save_plotly_figure(fig, "soc_comparison", output_dir, size_type="default")
    
    if verbose:
        out_path = get_output_dir(output_dir)
        print(f"图表已保存到 {out_path}/ 目录")
        print("  - soc_curve.{png,pdf}")
        print("  - power_curve.{png,pdf}")
        print("  - temperature_curve.{png,pdf}")
        print("  - state_timeline.{png,pdf}")
        print("  - composite_power_temperature.{png,pdf}")
        print("  - soc_comparison.{png,pdf}")
    
    return result


if __name__ == "__main__":
    run_basic_experiment()
