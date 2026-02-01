# visualization/config.py
# 可视化配置模块
#
# 提供统一的配置：
# - 中文字体支持
# - 默认输出目录
# - 全局样式设置

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# =====================================================
# 默认输出目录配置
# =====================================================

# 默认输出目录（相对于项目根目录）
DEFAULT_OUTPUT_DIR = "output/figures"

# 全局输出目录（可通过 set_output_dir 修改）
_output_dir = None


def get_output_dir():
    """
    获取当前输出目录
    
    返回：
        str - 输出目录路径
    """
    global _output_dir
    if _output_dir is None:
        # 尝试找到项目根目录
        current_dir = Path(__file__).resolve().parent
        # 向上查找直到找到 src 目录的父目录
        while current_dir.name != 'src' and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        project_root = current_dir.parent
        _output_dir = str(project_root / DEFAULT_OUTPUT_DIR)
    
    return _output_dir


def set_output_dir(path):
    """
    设置输出目录
    
    参数：
        path : str - 输出目录路径
    """
    global _output_dir
    _output_dir = path
    # 确保目录存在
    ensure_output_dir()


def ensure_output_dir():
    """确保输出目录存在"""
    output_dir = get_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_save_path(filename):
    """
    获取完整的保存路径
    
    参数：
        filename : str - 文件名
    
    返回：
        str - 完整路径
    """
    output_dir = ensure_output_dir()
    return os.path.join(output_dir, filename)


# =====================================================
# 中文字体配置
# =====================================================

# 标记是否已初始化字体
_font_initialized = False

def setup_chinese_font():
    """
    配置 matplotlib 支持中文显示
    
    自动检测系统可用的中文字体，优先级：
    1. WenQuanYi Micro Hei (文泉驿微米黑) - Linux
    2. WenQuanYi Zen Hei (文泉驿正黑) - Linux
    3. Noto Sans CJK SC (思源黑体) - Linux
    4. SimHei (黑体) - Windows
    5. Microsoft YaHei (微软雅黑) - Windows
    6. PingFang SC (苹方) - macOS
    7. Heiti SC (黑体-简) - macOS
    8. DejaVu Sans (兜底方案)
    """
    global _font_initialized
    import warnings
    import matplotlib.font_manager as fm
    
    # 刷新字体缓存（确保新安装的字体被识别）
    if not _font_initialized:
        try:
            # 尝试重新加载字体管理器
            fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        _font_initialized = True
    
    # 候选中文字体列表（优先 Linux 字体）
    chinese_fonts = [
        'WenQuanYi Micro Hei',  # Linux 文泉驿微米黑
        'WenQuanYi Zen Hei',    # Linux 文泉驿正黑
        'Noto Sans CJK SC',     # Linux 思源黑体
        'Droid Sans Fallback',  # Android/Linux
        'SimHei',               # Windows 黑体
        'Microsoft YaHei',      # Windows 微软雅黑
        'PingFang SC',          # macOS 苹方
        'Heiti SC',             # macOS 黑体
        'STHeiti',              # macOS 华文黑体
        'DejaVu Sans',          # 兜底方案
    ]
    
    # 获取系统可用字体
    available_fonts = set()
    try:
        from matplotlib.font_manager import fontManager
        for font in fontManager.ttflist:
            available_fonts.add(font.name)
    except Exception:
        pass
    
    # 找到第一个可用的中文字体
    selected_font = None
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 如果没有找到，使用 sans-serif 作为默认
    if selected_font is None:
        selected_font = 'sans-serif'
        warnings.warn(
            "未找到支持中文的字体，中文可能显示为方块。"
            "建议安装以下字体之一：SimHei, Microsoft YaHei, PingFang SC, "
            "WenQuanYi Micro Hei, Noto Sans CJK SC",
            UserWarning
        )
    
    return selected_font


def setup_style():
    """
    设置全局绘图样式（包含中文支持）
    """
    # 使用 seaborn 风格
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # 兼容旧版本
        try:
            plt.style.use('seaborn-whitegrid')
        except OSError:
            pass
    
    # 配置中文字体
    chinese_font = setup_chinese_font()
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['figure.dpi'] = 100
    
    # 图例字体
    plt.rcParams['legend.fontsize'] = 9
    
    # 刻度字体
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9


# =====================================================
# 配色方案
# =====================================================

COLORS = {
    "primary": "#2E86AB",      # 主色调 - 深蓝
    "secondary": "#A23B72",    # 次色调 - 紫红
    "accent": "#F18F01",       # 强调色 - 橙色
    "success": "#28A745",      # 成功色 - 绿色
    "danger": "#C73E1D",       # 警告色 - 红色
    "neutral": "#6C757D",      # 中性色 - 灰色
}

# 使用状态配色
STATE_COLORS = {
    "DeepIdle": "#4ECDC4",     # 青绿色
    "Social": "#45B7D1",       # 天蓝色
    "Video": "#96CEB4",        # 浅绿色
    "Gaming": "#FF6B6B",       # 珊瑚红
    "Navigation": "#FFE66D",   # 明黄色
    "Camera": "#DDA0DD",       # 梅红色
}

# 场景配色
SCENARIO_COLORS = {
    "Student Daily": "#2E86AB",
    "Commute": "#A23B72",
    "Weekend": "#F18F01",
    "Travel": "#28A745",
    "DeepIdle Only": "#6C757D",
    "Gaming Only": "#DC3545",
    "Video Only": "#17A2B8",
    "Navigation Only": "#FFC107",
}

# 默认颜色列表
DEFAULT_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#28A745",
    "#DC3545", "#17A2B8", "#6C757D", "#FFC107",
    "#6610F2", "#E83E8C", "#20C997", "#FD7E14"
]

# 敏感度参数中文标签
PARAM_LABELS = {
    "u": "屏幕亮度",
    "r": "刷新率",
    "u_cpu": "CPU 利用率",
    "lambda_cell": "蜂窝网络比例",
    "delta_signal": "信号质量修正",
    "r_on": "GPS 开启比例",
    "r_bg": "后台活跃比例",
}


# =====================================================
# 工具函数
# =====================================================

def save_figure(fig, filename, dpi=300, close_after=False):
    """
    保存图表到输出目录
    
    参数：
        fig : matplotlib.figure.Figure - 图表对象
        filename : str - 文件名
        dpi : int - 分辨率
        close_after : bool - 保存后是否关闭图表
    
    返回：
        str - 保存的完整路径
    """
    save_path = get_save_path(filename)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"📊 图表已保存: {save_path}")
    
    if close_after:
        plt.close(fig)
    
    return save_path


def to_hours(time_list):
    """将秒转换为小时"""
    if isinstance(time_list, list):
        return [t / 3600 for t in time_list]
    return time_list / 3600
