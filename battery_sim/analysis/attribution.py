from typing import Dict, List


def energy_breakdown(results: List[Dict]):
    """对蒙特卡洛结果进行能量分解统计并打印。"""
    if not results:
        print("没有结果可供分解。")
        return
    modules = ["screen", "cpu", "radio", "gps", "bg"]
    sums = {m: 0.0 for m in modules}
    for r in results:
        for m in modules:
            sums[m] += r["energy_by_module"].get(m, 0.0)
    n = len(results)
    print("==== 子模块平均能耗（Wh）====")
    for m in modules:
        print(f"{m:10s}: {sums[m]/n:.3f}")


def state_ablation(results: List[Dict]):
    """简单状态能耗统计。"""
    if not results:
        print("没有结果可供分解。")
        return
    agg: Dict[str, float] = {}
    for r in results:
        for k, v in r["energy_by_state"].items():
            agg[k] = agg.get(k, 0.0) + v
    n = len(results)
    print("==== 状态平均能耗（Wh）====")
    for k, v in agg.items():
        print(f"{k:5s}: {v/n:.3f}")
