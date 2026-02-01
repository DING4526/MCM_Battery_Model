# visualization/__init__.py
# 可视化模块初始化
#
# 提供统一的可视化接口，包含：
# - 时间序列可视化
# - 分布可视化
# - 敏感度分析可视化
# - 场景对比可视化
# - 配置管理（字体、输出目录）

# 导出配置函数
from .config import (
    setup_style,
    set_output_dir,
    get_output_dir,
    get_save_path,
    save_figure,
    ensure_output_dir,
    COLORS,
    STATE_COLORS,
    SCENARIO_COLORS,
    PARAM_LABELS,
)

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
)

from .comparison import (
    plot_scenario_comparison,
    plot_scenario_boxplot,
    plot_scenario_radar,
    plot_multi_scenario_timeline,
)

__all__ = [
    # 配置
    "setup_style",
    "set_output_dir",
    "get_output_dir",
    "get_save_path",
    "save_figure",
    "ensure_output_dir",
    "COLORS",
    "STATE_COLORS",
    "SCENARIO_COLORS",
    "PARAM_LABELS",
    # 时间序列
    "plot_single_run",
    "plot_soc_curve",
    "plot_power_curve",
    "plot_temperature_curve",
    "plot_state_timeline",
    "plot_comprehensive_dashboard",
    # 分布
    "plot_ttl_distribution",
    "plot_ttl_boxplot",
    "plot_ttl_violin",
    "plot_ttl_kde",
    "plot_ttl_statistical_summary",
    # 敏感度
    "plot_sensitivity_bar",
    "plot_sensitivity_tornado",
    "plot_sensitivity_spider",
    "plot_sensitivity_heatmap",
    # 对比
    "plot_scenario_comparison",
    "plot_scenario_boxplot",
    "plot_scenario_radar",
    "plot_multi_scenario_timeline",
]
