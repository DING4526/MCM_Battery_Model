from __future__ import annotations

import json
from pathlib import Path

from src.simulate import run_simulation
from src.visualization.soc_counterfactual import plot_soc_counterfactual, SOCCounterfactualConfig


def main():
    # 你可直接改成：output/population/timeseries/<Device_ID>.json
    base_json = Path(r"output/population/timeseries/1daf68e5-9789-4474-bc80-a6c5652e802d.json")

    out_dir = Path("output/00_all/soc_counterfactual_0C_from_timeseries")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(base_json.read_text(encoding="utf-8"))
    meta = payload.get("meta", {})

    # ---- 从 timeseries meta 恢复仿真输入 ----
    scenario = meta.get("scenario")
    usage_states = meta.get("usage_states")
    if scenario is None or usage_states is None:
        raise ValueError("timeseries json missing meta.scenario or meta.usage_states")

    dt = float(meta.get("dt", 1.0))
    fixed_dwell_sec = int(meta.get("fixed_dwell_sec", 15 * 60))
    seed = int(meta.get("seed", 42))

    cap_mah = float(meta.get("battery_capacity_mah", 5000.0))
    aging_loss = float((meta.get("aging", {}) or {}).get("aging_loss", meta.get("aging_loss", 0.15)))

    # ---- 只改环境温度：0°C ----
    T_amb_c = 0.0
    T_amb_K = 273.15 + T_amb_c

    # 注意：scenario 里 type 必须是 hsmm 才会触发 HSMM 逻辑
    # 你的 population/data_driven.py 已经是 type="hsmm" 了，所以一般这里不用改。
    # 如果你担心老文件还是 mixed，可以强行改：
    scenario = dict(scenario)
    scenario["type"] = "hsmm"

    # ---- 重跑仿真（关键：record_uncorrected=True）----
    result = run_simulation(
        scenario=scenario,
        dt=dt,
        T_amb=T_amb_K,
        seed=seed,
        record=True,
        record_breakdown=True,
        record_uncorrected=True,
        usage_states_override=usage_states,
        battery_capacity_mah=cap_mah,
        battery_aging_loss=aging_loss,
        fixed_dwell_sec=fixed_dwell_sec,
        # 这里其实无所谓，只要 scenario["type"]="hsmm" 就会走 HSMM 分支
        dwell_mode="fixed",
    )

    # ---- 导出新的 timeseries json（同结构，便于复用现有可视化）----
    device_id = meta.get("Device_ID", base_json.stem)
    out_json = out_dir / f"{device_id}_0C.json"

    out_payload = {
        "meta": {
            **meta,
            "T_amb_c": T_amb_c,     # 覆盖环境温度
            "note": "counterfactual: only T_amb changed to 0C; other device-driven inputs preserved",
        },
        "time": result.get("time", []),
        "SOC": result.get("SOC", []),  # fully corrected
        "Tb": result.get("Tb", []),
        "Power": result.get("Power", []),
        "State": result.get("State", []),

        "Power_screen": result.get("Power_screen", []),
        "Power_cpu": result.get("Power_cpu", []),
        "Power_radio": result.get("Power_radio", []),
        "Power_background": result.get("Power_background", []),

        "SOC_uncorrected": result.get("SOC_uncorrected", []),
        "SOC_voltage_only": result.get("SOC_voltage_only", []),
        "SOC_temperature_only": result.get("SOC_temperature_only", []),
        "SOC_aging_only": result.get("SOC_aging_only", []),
    }

    out_json.write_text(json.dumps(out_payload, ensure_ascii=False), encoding="utf-8")

    # ---- 画图 ----
    out_fig = out_dir / f"{device_id}_0C_soc_counterfactual.pdf"

    cfg = SOCCounterfactualConfig(
        figsize=(9.2, 4.2),          # 还想更宽就再加
        title=f"Counterfactual SOC @ {T_amb_c:.0f}°C (device-driven)",
        baseline="uncorrected",
        show_error_fill=True,
        show_error_axis=True,
        legend_outside=True,
    )

    plot_soc_counterfactual(
        str(out_json),
        cfg=cfg,
        save_path=str(out_fig),
        show=False,
    )

    print(f"[OK] base : {base_json}")
    print(f"[OK] json : {out_json}")
    print(f"[OK] fig  : {out_fig}")


if __name__ == "__main__":
    main()
