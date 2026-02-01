from typing import Dict, Any, List

# 预定义的日常场景（各状态占比，按一天 1440 分钟归一）
SCENARIOS: Dict[str, List[tuple[str, int]]] = {
    "Average": [
        ("S0", 400),
        ("S1", 300),
        ("S2", 200),
        ("S3", 150),
        ("S4", 120),
        ("S5", 120),
        ("S6", 100),
        ("S7", 50),
    ],
    "Heavy": [
        ("S0", 200),
        ("S1", 200),
        ("S2", 250),
        ("S3", 250),
        ("S4", 200),
        ("S5", 180),
        ("S6", 140),
        ("S7", 20),
    ],
    "Light": [
        ("S0", 600),
        ("S1", 400),
        ("S2", 150),
        ("S3", 100),
        ("S4", 80),
        ("S5", 60),
        ("S6", 30),
        ("S7", 20),
    ],
}


def get_daily_profile(name: str) -> List[tuple[str, int]]:
    """返回指定名称的日常场景（状态序列与持续分钟数）。"""
    if name not in SCENARIOS:
        raise KeyError(f"未知场景：{name}")
    return SCENARIOS[name]

