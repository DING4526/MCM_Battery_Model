# visualization/config.py
# 可视化配置模块（Plotly 版本 - 单栏 LaTeX 论文优化）
#
# 提供统一的配置：
# - 专业配色方案
# - 单栏论文尺寸适配（3.5 英寸宽度）
# - 全局样式设置
# - Plotly 模板

import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

# =====================================================
# 单栏 LaTeX 论文尺寸配置
# =====================================================

# 单栏论文标准宽度：3.5 英寸 ≈ 8.89 cm
# 300 DPI 下：3.5 * 300 = 1050 像素
# 使用 scale=2 导出以获得高清效果

LATEX_SINGLE_COLUMN_WIDTH_INCH = 3.5
LATEX_DPI = 300

# 图表尺寸（像素）- 导出时 scale=2
FIGURE_WIDTH = 700
FIGURE_HEIGHT = 500

# 不同图表类型的推荐尺寸
FIGURE_SIZES = {
    "default": (700, 500),
    "wide": (700, 400),
    "square": (600, 550),
    "tall": (700, 600),
    "timeline": (700, 250),
    "composite": (700, 520),   # 紧凑复合图
}

# 字体大小（论文适配）
FONT_SIZES = {
    "title": 14,
    "axis_title": 11,
    "axis_tick": 10,
    "legend": 9,
    "annotation": 9,
}

# 线条宽度
LINE_WIDTHS = {
    "main": 2.0,
    "secondary": 1.5,
    "grid": 0.5,
    "axis": 1.0,
}


# =====================================================
# 默认输出目录配置
# =====================================================

_project_root = None
_show_plots = False


def _get_project_root():
    """获取项目根目录"""
    global _project_root
    if _project_root is None:
        current_dir = Path(__file__).resolve().parent
        while current_dir.name != 'src' and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        _project_root = current_dir.parent
    return _project_root


def set_show_plots(show: bool):
    """设置是否显示图形"""
    global _show_plots
    _show_plots = show


def get_show_plots() -> bool:
    """获取当前是否显示图形窗口"""
    return _show_plots


def get_output_dir(subdir=""):
    """获取输出目录"""
    project_root = _get_project_root()
    if subdir:
        output_dir = project_root / "output" / subdir
    else:
        output_dir = project_root / "output"
    os.makedirs(output_dir, exist_ok=True)
    return str(output_dir)


def set_output_dir(path):
    """设置输出目录（兼容旧接口）"""
    pass


def ensure_output_dir():
    """确保输出目录存在（兼容旧接口）"""
    return get_output_dir()


def get_save_path(filename, subdir=""):
    """获取完整的保存路径"""
    if filename is None:
        return None
    
    if os.path.isabs(filename):
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return filename
    
    output_dir = get_output_dir(subdir)
    return os.path.join(output_dir, filename)


# =====================================================
# 配色方案（论文级专业配色 - 深沉低饱和）
# =====================================================

COLORS = {
    "primary": "#34495E",      # 深蓝灰
    "secondary": "#8B4513",    # 深棕色（温度）
    "accent": "#2C5F7C",       # 深青蓝
    "success": "#3D6B4F",      # 深绿
    "danger": "#A0522D",       # 褐红色
    "neutral": "#5D6D7E",      # 中灰蓝
    "warning": "#8B6914",      # 深金色
}

# 使用状态配色（低饱和、高区分度）
STATE_COLORS = {
    "DeepIdle": "#7B8A8B",     # 灰色
    "Social": "#4A6F8A",       # 暗蓝
    "Video": "#6B4F7A",        # 暗紫
    "Gaming": "#8B4D4D",       # 暗红
    "Navigation": "#7A6332",   # 暗黄棕
    "Camera": "#3D7A6B",       # 暗青绿
}

# 场景配色
SCENARIO_COLORS = {
    "Student Daily": "#4A6F8A",
    "学生日常": "#4A6F8A",
    "Commute": "#8B4D4D",
    "通勤": "#8B4D4D",
    "Weekend": "#6B4F7A",
    "周末娱乐": "#6B4F7A",
    "Travel": "#3D7A6B",
    "旅行": "#3D7A6B",
    "DeepIdle Only": "#7B8A8B",
    "纯待机": "#7B8A8B",
    "Gaming Only": "#8B4D4D",
    "纯游戏": "#8B4D4D",
    "Video Only": "#6B4F7A",
    "纯视频": "#6B4F7A",
    "Navigation Only": "#7A6332",
    "纯导航": "#7A6332",
}

# 默认颜色序列（低饱和专业色）
DEFAULT_COLORS = [
    "#4A6F8A", "#8B4D4D", "#3D7A6B", "#6B4F7A",
    "#7A6332", "#5D6D7E", "#34495E", "#7B8A8B",
]

# 功耗分解配色（深沉低饱和，通过明度区分层级）
POWER_BREAKDOWN_COLORS = {
    "屏幕": "#6B7B8C",       # 浅灰蓝（最上层，明度最高）
    "CPU": "#4A6B5C",        # 深绿灰
    "无线通信": "#5A6B7C",   # 中灰蓝
    "GPS": "#7A6B4A",        # 棕灰
    "后台": "#5A5A6A",       # 暗灰紫（最底层，明度最低）
}

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
# Plotly 模板配置（论文优化）
# =====================================================

def setup_chinese_font():
    """返回中文字体族（优先使用已安装的中文字体）"""
    # 优先使用 Linux 系统安装的中文字体
    return "WenQuanYi Micro Hei, Noto Sans CJK SC, SimHei, Microsoft YaHei, PingFang SC, Arial, sans-serif"


def setup_style():
    """设置全局绘图样式（针对单栏论文优化）"""
    font_family = setup_chinese_font()
    
    professional_template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family=font_family,
                size=FONT_SIZES["axis_tick"],
                color="#2C3E50"
            ),
            title=dict(
                font=dict(size=FONT_SIZES["title"], color="#1A252F"),
                x=0.5,
                xanchor="center",
                y=0.95,
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=LINE_WIDTHS["grid"],
                gridcolor="rgba(127, 140, 141, 0.3)",
                showline=True,
                linewidth=LINE_WIDTHS["axis"],
                linecolor="#2C3E50",
                ticks="outside",
                tickfont=dict(size=FONT_SIZES["axis_tick"]),
                title_font=dict(size=FONT_SIZES["axis_title"], color="#2C3E50"),
                zeroline=False,
                mirror=True,
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=LINE_WIDTHS["grid"],
                gridcolor="rgba(127, 140, 141, 0.3)",
                showline=True,
                linewidth=LINE_WIDTHS["axis"],
                linecolor="#2C3E50",
                ticks="outside",
                tickfont=dict(size=FONT_SIZES["axis_tick"]),
                title_font=dict(size=FONT_SIZES["axis_title"], color="#2C3E50"),
                zeroline=False,
                mirror=True,
            ),
            legend=dict(
                font=dict(size=FONT_SIZES["legend"]),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(44, 62, 80, 0.5)",
                borderwidth=1,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            colorway=DEFAULT_COLORS,
            margin=dict(l=50, r=20, t=50, b=45),
            hoverlabel=dict(
                bgcolor="white",
                font_size=FONT_SIZES["annotation"],
                font_family=font_family,
            ),
        )
    )
    
    pio.templates["professional"] = professional_template
    pio.templates.default = "professional"


def get_monospace_font():
    """获取等宽字体"""
    return "Consolas, Monaco, 'Courier New', monospace"


# =====================================================
# 图表保存函数（论文优化）
# =====================================================

def save_figure(fig, filename, subdir="", dpi=300, close_after=False):
    """保存 Plotly 图表（论文格式优化）"""
    save_path = get_save_path(filename, subdir)
    ext = os.path.splitext(filename)[1].lower()
    
    if ext == '.html':
        fig.write_html(
            save_path,
            include_plotlyjs="cdn",
            full_html=True,
            config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False,
            }
        )
    else:
        scale = dpi / 100
        fig.write_image(save_path, scale=scale)
    
    print(f"[Save] 图表已保存: {save_path}")
    return save_path


def smart_savefig(filename, subdir="", dpi=300):
    """兼容旧接口"""
    if filename is None:
        return None
    return get_save_path(filename, subdir)


def save_plotly_figure(fig, filename, subdir="", size_type="default"):
    """
    保存 Plotly 图表（静态格式：PNG 和 PDF）
    
    参数：
        fig : Figure - 图表对象
        filename : str - 文件名（不含扩展名）
        subdir : str - 子目录名
        size_type : str - 尺寸类型
    """
    if filename is None:
        return None
    
    base_name = os.path.splitext(filename)[0]
    width, height = FIGURE_SIZES.get(size_type, FIGURE_SIZES["default"])
    
    saved_paths = {}
    
    # PNG（高清静态图）
    png_path = get_save_path(f"{base_name}.png", subdir)
    try:
        fig.write_image(png_path, width=width, height=height, scale=2)
        saved_paths["png"] = png_path
        print(f"[Save] PNG: {png_path}")
    except Exception as e:
        print(f"[Warning] PNG 导出失败: {e}")
        print("  提示: 请确保已安装 kaleido: pip install kaleido")
    
    # PDF（矢量格式，适合论文）
    pdf_path = get_save_path(f"{base_name}.pdf", subdir)
    try:
        fig.write_image(pdf_path, width=width, height=height, scale=2)
        saved_paths["pdf"] = pdf_path
        print(f"[Save] PDF: {pdf_path}")
    except Exception as e:
        print(f"[Warning] PDF 导出失败: {e}")
        print("  提示: 请确保已安装 kaleido: pip install kaleido")
    
    return saved_paths


# =====================================================
# 工具函数
# =====================================================

def to_hours(time_list):
    """将秒转换为小时"""
    if isinstance(time_list, list):
        return [t / 3600 for t in time_list]
    return time_list / 3600


def get_figure_size(size_type="default"):
    """获取图表尺寸"""
    return FIGURE_SIZES.get(size_type, FIGURE_SIZES["default"])


def hex_to_rgba(hex_color, alpha=1.0):
    """
    将十六进制颜色转换为 RGBA 字符串
    
    参数：
        hex_color : str - 十六进制颜色，如 "#2980B9"
        alpha : float - 透明度 (0.0-1.0)
    
    返回：
        str - RGBA 字符串，如 "rgba(41, 128, 185, 0.5)"
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def get_color_with_alpha(color_name, alpha=1.0):
    """
    获取带透明度的颜色
    
    参数：
        color_name : str - 颜色名称（COLORS 字典的键）
        alpha : float - 透明度 (0.0-1.0)
    
    返回：
        str - RGBA 字符串
    """
    if color_name in COLORS:
        return hex_to_rgba(COLORS[color_name], alpha)
    return hex_to_rgba(color_name, alpha)


# 初始化样式
setup_style()
