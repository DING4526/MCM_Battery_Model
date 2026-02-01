from typing import List, Dict


def oat_sensitivity(results: List[Dict], key: str = "E0", span: float = 0.1):
    """
    单因素灵敏度（极简版）：假设 results 中的设备参数略扰动，比较 TTE 线性近似。
    """
    if not results:
        print("没有结果可用于灵敏度分析。")
        return
    base_TTE = sum([r["TTE_h"] for r in results]) / len(results)
    perturb = []
    for r in results:
        dev = r["device"]
        if key not in dev:
            continue
        dev_plus = dev.copy()
        dev_plus[key] = dev[key] * (1 + span)
        # 线性假设：TTE 与 key 成正比
        if dev[key] != 0:
            perturb.append(base_TTE * (dev_plus[key] / dev[key]))
    if not perturb:
        print("缺少可用参数进行灵敏度分析。")
        return
    print(f"==== 单因素灵敏度（{key}，±{int(span*100)}% 线性近似）====")
    print(f"基线 TTE: {base_TTE:.2f} h")
    print(f"{key} 上调后近似 TTE: {sum(perturb)/len(perturb):.2f} h")
