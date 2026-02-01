#!/usr/bin/env python3
# main.py
# 实验主入口（简化版）
#
# 默认行为：
# - 不弹出图形窗口（保存到 output/ 目录）
# - 每个实验类型保存到独立子目录
# - 支持 python main.py / python main.py all 一键跑全套

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
from usage.scenario import *


# =====================================================
# 场景映射
# =====================================================

SCENARIO_MAP = {
    "student_daily": (SCENARIO_STUDENT_DAILY_MIXED, "学生日常"),
    "student_markov": (SCENARIO_STUDENT_DAILY_MARKOV, "学生日常 Markov"),
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
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║   🔋  手机电池仿真系统 - Battery Simulation Framework  🔋   ║
╠══════════════════════════════════════════════════════════════╣
║   all          - 一键跑完全部实验                            ║
║   basic        - 基础单次仿真                                ║
║   monte_carlo  - Monte Carlo 随机仿真                        ║
║   sensitivity  - 参数敏感度分析                              ║
║   compare      - 多场景对比分析                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    )


def run_basic(args):
    """运行基础仿真"""
    scenario, scenario_name = SCENARIO_MAP.get(
        args.scenario, (SCENARIO_STUDENT_DAILY_MIXED, "学生日常")
    )

    run_basic_experiment(
        scenario=scenario,
        scenario_name=scenario_name,
        seed=args.seed,
        T_amb=args.temperature + 273.15,
        output_dir="basic",
    )


def run_monte_carlo(args):
    """运行 Monte Carlo 仿真"""
    scenario, scenario_name = SCENARIO_MAP.get(
        args.scenario, (SCENARIO_STUDENT_DAILY_MIXED, "学生日常")
    )

    run_monte_carlo_experiment(
        scenario=scenario,
        scenario_name=scenario_name,
        n_samples=args.samples,
        base_seed=args.seed,
        T_amb=args.temperature + 273.15,
        output_dir="monte_carlo",
    )


def run_sensitivity(args):
    """运行敏感度分析"""
    scenario, scenario_name = SCENARIO_MAP.get(
        args.scenario, (SCENARIO_STUDENT_DAILY_MARKOV, "学生日常 Markov")
    )

    run_sensitivity_experiment(
        scenario=scenario,
        scenario_name=scenario_name,
        eps=args.eps,
        n_mc=args.samples,
        output_dir="sensitivity",
    )


def run_compare(args):
    """运行场景对比"""
    run_comparison_experiment(
        group_name=args.group,
        n_mc=args.samples,
        T_amb=args.temperature + 273.15,
        output_dir="compare",
    )


# ===== 新增：统一跑全部实验 =====
def run_all(args):
    """一键跑完所有实验（默认不弹窗，保存到 output/<type>/）"""

    # 1) basic
    run_basic(
        argparse.Namespace(
            scenario=args.scenario,
            seed=args.seed,
            temperature=args.temperature,
        )
    )

    # 2) monte_carlo
    run_monte_carlo(
        argparse.Namespace(
            scenario=args.scenario,
            samples=args.samples,
            seed=args.seed,
            temperature=args.temperature,
        )
    )

    # 3) sensitivity（通常用 Markov 更合适）
    run_sensitivity(
        argparse.Namespace(
            scenario=args.sens_scenario,
            eps=args.eps,
            samples=args.sens_samples,
        )
    )

    # 4) compare
    run_compare(
        argparse.Namespace(
            group=args.group,
            samples=args.compare_samples,
            temperature=args.temperature,
        )
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="手机电池仿真系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py
  python main.py all
  python main.py all --scenario gaming --samples 300 --group 极端场景
  python main.py basic --scenario gaming
  python main.py monte_carlo --samples 200
  python main.py sensitivity --eps 0.2
  python main.py compare --group 日常场景

图片默认保存到 output/<实验类型>/ 目录
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="实验类型")

    # ===== all：一键跑完全部 =====
    parser_all = subparsers.add_parser("all", help="一键跑完所有实验")

    # basic + monte_carlo 共用默认场景
    parser_all.add_argument(
        "--scenario",
        type=str,
        default="student_daily",
        choices=list(SCENARIO_MAP.keys()),
        help="basic/MC 默认场景",
    )
    parser_all.add_argument("--seed", type=int, default=42, help="随机种子（basic/MC）")
    parser_all.add_argument("--samples", type=int, default=200, help="MC 仿真次数")
    parser_all.add_argument(
        "--temperature", type=float, default=25.0, help="环境温度（°C）"
    )

    # sensitivity 专用参数（重要：你 run_all 用到了这些，必须定义）
    parser_all.add_argument(
        "--sens-scenario",
        dest="sens_scenario",
        type=str,
        default="student_markov",
        choices=list(SCENARIO_MAP.keys()),
        help="敏感度分析场景",
    )
    parser_all.add_argument("--eps", type=float, default=0.2, help="敏感度扰动幅度")
    parser_all.add_argument("--sens-samples", dest="sens_samples", type=int, default=80,
                            help="敏感度分析 MC 样本数")

    # compare 专用参数（重要：你 run_all 用到了这些，必须定义）
    parser_all.add_argument(
        "--group",
        type=str,
        default="日常场景",
        choices=["日常场景", "极端场景", "混合 vs Markov"],
        help="对比场景组",
    )
    parser_all.add_argument("--compare-samples", dest="compare_samples", type=int, default=80,
                            help="场景对比 MC 样本数")

    # 关键：绑定 all 的执行函数（你之前缺这个会直接失败）
    parser_all.set_defaults(func=run_all)

    # ===== basic =====
    parser_basic = subparsers.add_parser("basic", help="基础单次仿真")
    parser_basic.add_argument(
        "--scenario",
        type=str,
        default="student_daily",
        choices=list(SCENARIO_MAP.keys()),
        help="使用场景",
    )
    parser_basic.add_argument("--seed", type=int, default=42, help="随机种子")
    parser_basic.add_argument(
        "--temperature", type=float, default=25.0, help="环境温度（°C）"
    )
    parser_basic.set_defaults(func=run_basic)

    # ===== monte_carlo =====
    parser_mc = subparsers.add_parser("monte_carlo", help="Monte Carlo 随机仿真")
    parser_mc.add_argument(
        "--scenario",
        type=str,
        default="student_daily",
        choices=list(SCENARIO_MAP.keys()),
        help="使用场景",
    )
    parser_mc.add_argument("--samples", type=int, default=100, help="仿真次数")
    parser_mc.add_argument("--seed", type=int, default=0, help="基础随机种子")
    parser_mc.add_argument(
        "--temperature", type=float, default=25.0, help="环境温度（°C）"
    )
    parser_mc.set_defaults(func=run_monte_carlo)

    # ===== sensitivity =====
    parser_sens = subparsers.add_parser("sensitivity", help="参数敏感度分析")
    parser_sens.add_argument(
        "--scenario",
        type=str,
        default="student_markov",
        choices=list(SCENARIO_MAP.keys()),
        help="使用场景",
    )
    parser_sens.add_argument("--eps", type=float, default=0.2, help="扰动幅度")
    parser_sens.add_argument("--samples", type=int, default=50, help="Monte Carlo 样本数")
    parser_sens.set_defaults(func=run_sensitivity)

    # ===== compare =====
    parser_compare = subparsers.add_parser("compare", help="多场景对比分析")
    parser_compare.add_argument(
        "--group",
        type=str,
        default="日常场景",
        choices=["日常场景", "极端场景", "混合 vs Markov"],
        help="场景组",
    )
    parser_compare.add_argument("--samples", type=int, default=50, help="Monte Carlo 样本数")
    parser_compare.add_argument(
        "--temperature", type=float, default=25.0, help="环境温度（°C）"
    )
    parser_compare.set_defaults(func=run_compare)

    args = parser.parse_args()

    # 不传 command：默认跑 all
    if args.command is None:
        print_banner()
        default_all = parser.parse_args(["all"])
        default_all.func(default_all)
        return

    print_banner()
    args.func(args)


if __name__ == "__main__":
    main()
