# visualization/__init__.py
# 可视化模块（Plotly 版本 - 单栏论文优化）
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
    save_plotly_figure,
    COLORS,
    STATE_COLORS,
    PARAM_LABELS,
    FIGURE_SIZES,
    FONT_SIZES,
)

# 时间序列可视化
from .timeseries import (
    plot_single_run,
    plot_soc_curve,
    plot_power_curve,
    plot_temperature_curve,
    plot_state_timeline,
    plot_comprehensive_dashboard,
    plot_composite_power_temperature,
    plot_soc_comparison,
)

# 分布可视化
from .distribution import (
    plot_ttl_distribution,
    plot_ttl_boxplot,
    plot_ttl_violin,
    plot_ttl_kde,
    plot_ttl_statistical_summary,
)

# 敏感度分析可视化
from .sensitivity_plot import (
    plot_sensitivity_bar,
    plot_sensitivity_tornado,
    plot_sensitivity_spider,
    plot_sensitivity_heatmap,
    plot_sensitivity_comprehensive,
)

# 场景对比可视化
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
    "smart_savefig",
    "save_plotly_figure",
    "COLORS",
    "STATE_COLORS",
    "PARAM_LABELS",
    "FIGURE_SIZES",
    "FONT_SIZES",
    # 时间序列
    "plot_single_run",
    "plot_soc_curve",
    "plot_power_curve",
    "plot_temperature_curve",
    "plot_state_timeline",
    "plot_comprehensive_dashboard",
    "plot_composite_power_temperature",
    "plot_soc_comparison",
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
    "plot_sensitivity_comprehensive",
    # 对比
    "plot_scenario_comparison",
    "plot_scenario_boxplot",
    "plot_scenario_radar",
    "plot_multi_scenario_timeline",
    "plot_scenario_comprehensive_comparison",
]
