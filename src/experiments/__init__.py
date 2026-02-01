# experiments/__init__.py
# 实验模块初始化
#
# 提供统一的实验入口接口，包含：
# - 基础仿真实验
# - Monte Carlo 仿真实验
# - 敏感度分析实验
# - 场景对比实验

from .exp_basic import run_basic_experiment
from .exp_monte_carlo import run_monte_carlo_experiment
from .exp_sensitivity import run_sensitivity_experiment
from .exp_compare import run_comparison_experiment

__all__ = [
    "run_basic_experiment",
    "run_monte_carlo_experiment",
    "run_sensitivity_experiment",
    "run_comparison_experiment",
]
