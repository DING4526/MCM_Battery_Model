#!/usr/bin/env python3
# main.py
# 实验主入口
#
# 提供统一的命令行接口，支持运行不同类型的实验：
# - 基础仿真
# - Monte Carlo 仿真
# - 敏感度分析
# - 场景对比

import sys
import os
import argparse

# 确保 src 目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments import (
    run_basic_experiment,
    run_monte_carlo_experiment,
    run_sensitivity_experiment,
    run_comparison_experiment,
)
from experiments.exp_monte_carlo import run_convergence_analysis
from experiments.exp_sensitivity import run_multi_eps_sensitivity
from experiments.exp_compare import run_sensitivity_to_temperature, run_all_group_comparisons

from usage.scenario import *


# =====================================================
# 场景映射
# =====================================================

SCENARIO_MAP = {
    "student_daily": (SCENARIO_STUDENT_DAILY_MIXED, "学生日常 (Mixed)"),
    "student_markov": (SCENARIO_STUDENT_DAILY_MARKOV, "学生日常 (Markov)"),
    "commute": (SCENARIO_COMMUTE_MIXED, "通勤"),
    "weekend": (SCENARIO_WEEKEND_MIXED, "周末娱乐"),
    "travel": (SCENARIO_TRAVEL_MIXED, "旅行"),
    "deepidle": (PURE_DEEPIDLE, "纯待机"),
    "social": (PURE_SOCIAL, "纯社交"),
    "video": (PURE_VIDEO, "纯视频"),
    "gaming": (PURE_GAMING, "纯游戏"),
    "navigation": (PURE_NAVIGATION, "纯导航"),
}


def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🔋  手机电池仿真系统 - Battery Simulation Framework  🔋   ║
    ║                                                              ║
    ║   支持功能:                                                  ║
    ║   • 基础单次仿真 (basic)                                     ║
    ║   • Monte Carlo 随机仿真 (monte_carlo)                       ║
    ║   • 参数敏感度分析 (sensitivity)                             ║
    ║   • 多场景对比分析 (compare)                                 ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_basic(args):
    """运行基础仿真"""
    scenario, scenario_name = SCENARIO_MAP.get(args.scenario, (SCENARIO_STUDENT_DAILY_MIXED, "学生日常"))
    
    run_basic_experiment(
        scenario=scenario,
        scenario_name=scenario_name,
        seed=args.seed,
        dt=args.dt,
        T_amb=args.temperature + 273.15,
        verbose=True,
        visualize=not args.no_plot,
        dashboard=args.dashboard,
        save_prefix=args.save,
    )


def run_monte_carlo(args):
    """运行 Monte Carlo 仿真"""
    scenario, scenario_name = SCENARIO_MAP.get(args.scenario, (SCENARIO_STUDENT_DAILY_MIXED, "学生日常"))
    
    run_monte_carlo_experiment(
        scenario=scenario,
        scenario_name=scenario_name,
        n_samples=args.samples,
        base_seed=args.seed,
        dt=args.dt,
        T_amb=args.temperature + 273.15,
        verbose=True,
        visualize=not args.no_plot,
        summary_plot=args.summary,
        save_prefix=args.save,
    )
    
    if args.convergence:
        print("\n进行收敛性分析...")
        run_convergence_analysis(
            scenario=scenario,
            scenario_name=scenario_name,
            max_samples=args.samples,
            base_seed=args.seed,
            verbose=True,
            visualize=not args.no_plot,
        )


def run_sensitivity(args):
    """运行敏感度分析"""
    scenario, scenario_name = SCENARIO_MAP.get(args.scenario, (SCENARIO_STUDENT_DAILY_MARKOV, "学生日常 Markov"))
    
    run_sensitivity_experiment(
        scenario=scenario,
        scenario_name=scenario_name,
        eps=args.eps,
        n_mc=args.samples,
        verbose=True,
        visualize=not args.no_plot,
        comprehensive_plot=args.comprehensive,
        save_prefix=args.save,
    )
    
    if args.multi_eps:
        print("\n进行多扰动幅度分析...")
        run_multi_eps_sensitivity(
            scenario=scenario,
            scenario_name=scenario_name,
            n_mc=args.samples // 2,
            verbose=True,
            visualize=not args.no_plot,
        )


def run_compare(args):
    """运行场景对比"""
    run_comparison_experiment(
        group_name=args.group,
        n_mc=args.samples,
        base_seed=args.seed,
        dt=args.dt,
        T_amb=args.temperature + 273.15,
        verbose=True,
        visualize=not args.no_plot,
        comprehensive_plot=args.comprehensive,
        include_timeline=args.timeline,
        save_prefix=args.save,
    )
    
    if args.temperature_analysis:
        print("\n进行温度敏感性分析...")
        scenario, scenario_name = SCENARIO_MAP.get("student_daily", (SCENARIO_STUDENT_DAILY_MIXED, "学生日常"))
        run_sensitivity_to_temperature(
            scenario=scenario,
            scenario_name=scenario_name,
            n_mc=args.samples // 2,
            verbose=True,
            visualize=not args.no_plot,
        )


def run_demo(args):
    """运行快速演示"""
    print_banner()
    
    if args.demo_type == "all" or args.demo_type == "basic":
        print("\n" + "=" * 60)
        print("📱 基础仿真演示")
        print("=" * 60)
        run_basic_experiment(
            scenario=SCENARIO_STUDENT_DAILY_MIXED,
            scenario_name="学生日常",
            seed=42,
            visualize=True,
            dashboard=True,
        )
    
    if args.demo_type == "all" or args.demo_type == "monte_carlo":
        print("\n" + "=" * 60)
        print("🎲 Monte Carlo 仿真演示")
        print("=" * 60)
        run_monte_carlo_experiment(
            scenario=SCENARIO_STUDENT_DAILY_MIXED,
            scenario_name="学生日常",
            n_samples=100,
            visualize=True,
            summary_plot=True,
        )
    
    if args.demo_type == "all" or args.demo_type == "sensitivity":
        print("\n" + "=" * 60)
        print("📊 敏感度分析演示")
        print("=" * 60)
        run_sensitivity_experiment(
            scenario=SCENARIO_STUDENT_DAILY_MARKOV,
            scenario_name="学生日常 Markov",
            n_mc=30,
            visualize=True,
            comprehensive_plot=True,
        )
    
    if args.demo_type == "all" or args.demo_type == "compare":
        print("\n" + "=" * 60)
        print("🔬 场景对比演示")
        print("=" * 60)
        run_comparison_experiment(
            group_name="日常场景",
            n_mc=30,
            visualize=True,
            comprehensive_plot=True,
            include_timeline=True,
        )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="手机电池仿真系统 - Battery Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行基础仿真
  python main.py basic --scenario student_daily --seed 42 --dashboard
  
  # 运行 Monte Carlo 仿真
  python main.py monte_carlo --scenario gaming --samples 200 --summary
  
  # 运行敏感度分析
  python main.py sensitivity --scenario student_markov --eps 0.2 --comprehensive
  
  # 运行场景对比
  python main.py compare --group 日常场景 --timeline --comprehensive
  
  # 运行快速演示
  python main.py demo --type all
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="实验类型")
    
    # ===== 基础仿真 =====
    parser_basic = subparsers.add_parser("basic", help="基础单次仿真")
    parser_basic.add_argument("--scenario", type=str, default="student_daily",
                              choices=list(SCENARIO_MAP.keys()),
                              help="使用场景")
    parser_basic.add_argument("--seed", type=int, default=42, help="随机种子")
    parser_basic.add_argument("--dt", type=float, default=1.0, help="时间步长（秒）")
    parser_basic.add_argument("--temperature", type=float, default=25.0, help="环境温度（°C）")
    parser_basic.add_argument("--dashboard", action="store_true", help="显示综合仪表板")
    parser_basic.add_argument("--no-plot", action="store_true", help="不显示图形")
    parser_basic.add_argument("--save", type=str, default=None, help="保存图片路径前缀")
    parser_basic.set_defaults(func=run_basic)
    
    # ===== Monte Carlo 仿真 =====
    parser_mc = subparsers.add_parser("monte_carlo", help="Monte Carlo 随机仿真")
    parser_mc.add_argument("--scenario", type=str, default="student_daily",
                           choices=list(SCENARIO_MAP.keys()),
                           help="使用场景")
    parser_mc.add_argument("--samples", type=int, default=100, help="仿真次数")
    parser_mc.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser_mc.add_argument("--dt", type=float, default=1.0, help="时间步长（秒）")
    parser_mc.add_argument("--temperature", type=float, default=25.0, help="环境温度（°C）")
    parser_mc.add_argument("--summary", action="store_true", help="显示综合统计图")
    parser_mc.add_argument("--convergence", action="store_true", help="进行收敛性分析")
    parser_mc.add_argument("--no-plot", action="store_true", help="不显示图形")
    parser_mc.add_argument("--save", type=str, default=None, help="保存图片路径前缀")
    parser_mc.set_defaults(func=run_monte_carlo)
    
    # ===== 敏感度分析 =====
    parser_sens = subparsers.add_parser("sensitivity", help="参数敏感度分析")
    parser_sens.add_argument("--scenario", type=str, default="student_markov",
                             choices=list(SCENARIO_MAP.keys()),
                             help="使用场景")
    parser_sens.add_argument("--eps", type=float, default=0.2, help="扰动幅度")
    parser_sens.add_argument("--samples", type=int, default=50, help="Monte Carlo 样本数")
    parser_sens.add_argument("--comprehensive", action="store_true", help="显示综合分析图")
    parser_sens.add_argument("--multi-eps", action="store_true", help="进行多扰动幅度分析")
    parser_sens.add_argument("--no-plot", action="store_true", help="不显示图形")
    parser_sens.add_argument("--save", type=str, default=None, help="保存图片路径前缀")
    parser_sens.set_defaults(func=run_sensitivity)
    
    # ===== 场景对比 =====
    parser_compare = subparsers.add_parser("compare", help="多场景对比分析")
    parser_compare.add_argument("--group", type=str, default="日常场景",
                                choices=["日常场景", "极端场景", "混合 vs Markov"],
                                help="场景组")
    parser_compare.add_argument("--samples", type=int, default=50, help="Monte Carlo 样本数")
    parser_compare.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser_compare.add_argument("--dt", type=float, default=1.0, help="时间步长（秒）")
    parser_compare.add_argument("--temperature", type=float, default=25.0, help="环境温度（°C）")
    parser_compare.add_argument("--comprehensive", action="store_true", help="显示综合分析图")
    parser_compare.add_argument("--timeline", action="store_true", help="包含时间线对比")
    parser_compare.add_argument("--temperature-analysis", action="store_true", help="进行温度敏感性分析")
    parser_compare.add_argument("--no-plot", action="store_true", help="不显示图形")
    parser_compare.add_argument("--save", type=str, default=None, help="保存图片路径前缀")
    parser_compare.set_defaults(func=run_compare)
    
    # ===== 快速演示 =====
    parser_demo = subparsers.add_parser("demo", help="快速演示")
    parser_demo.add_argument("--type", type=str, dest="demo_type", default="all",
                             choices=["all", "basic", "monte_carlo", "sensitivity", "compare"],
                             help="演示类型")
    parser_demo.set_defaults(func=run_demo)
    
    args = parser.parse_args()
    
    if args.command is None:
        print_banner()
        parser.print_help()
        return
    
    print_banner()
    args.func(args)


if __name__ == "__main__":
    main()
