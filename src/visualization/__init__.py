# visualization/__init__.py
# 可视化模块
#
# 提供统一的可视化接口

# 配置函数
from .config import (
    setup_style,
    set_output_dir,
    get_output_dir,
    set_show_plots,
    get_show_plots,
    smart_savefig,
    COLORS,
    STATE_COLORS,
    PARAM_LABELS,
)

# 主要可视化函数（供实验模块使用）
from .timeseries import (
    plot_single_run,
    plot_soc_curve,
    plot_power_curve,
    plot_temperature_curve,
    plot_state_timeline,
    plot_comprehensive_dashboard,
)

from .distribution import (
    plot_ttl_distribution,
    plot_ttl_boxplot,
    plot_ttl_violin,
    plot_ttl_kde,
    plot_ttl_statistical_summary,
)

from .sensitivity_plot import (
    plot_sensitivity_bar,
    plot_sensitivity_tornado,
    plot_sensitivity_spider,
    plot_sensitivity_heatmap,
    plot_sensitivity_comprehensive,
)

from .comparison import (
    plot_scenario_comparison,
    plot_scenario_boxplot,
    plot_scenario_radar,
    plot_multi_scenario_timeline,
    plot_scenario_comprehensive_comparison,
)

__all__ = [
    # 配置
    "setup_style",
    "set_output_dir",
    "get_output_dir",
    "set_show_plots",
    "get_show_plots",
    "COLORS",
    "STATE_COLORS",
    "PARAM_LABELS",
    # 时间序列
    "plot_single_run",
    "plot_comprehensive_dashboard",
    # 分布
    "plot_ttl_distribution",
    "plot_ttl_statistical_summary",
    # 敏感度
    "plot_sensitivity_bar",
    "plot_sensitivity_comprehensive",
    # 对比
    "plot_scenario_comparison",
    "plot_scenario_comprehensive_comparison",
]
