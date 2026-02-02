# src/population/runner.py
from __future__ import annotations

import os
import csv
import json
import hashlib
from typing import Optional, List, Dict, Any, Set

from src.aging_model import estimate_aging_from_row
from src.simulate import run_simulation
from src.population.data_driven import build_scenario_from_row, build_usage_states_from_row

# 运行方式（项目根目录）：
#   python -m src.population.runner


# =========================================================
# Utils
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seed_from_device(device_id: str, seed_base: int = 42) -> int:
    """
    稳定 seed：只依赖 Device_ID（不会因抽样/顺序变化导致同一设备结果变）
    """
    h = hashlib.md5(device_id.encode("utf-8")).hexdigest()
    return int(seed_base) + (int(h[:8], 16) % 2_000_000_000)


def _read_existing_ids(summary_csv_path: str) -> Set[str]:
    """resume：读取 summary.csv 中已存在的 Device_ID"""
    if not os.path.exists(summary_csv_path):
        return set()
    done: Set[str] = set()
    with open(summary_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            did = (row.get("Device_ID") or "").strip()
            if did:
                done.add(did)
    return done


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


# =========================================================
# Runner
# =========================================================

def run_population(
    csv_path: str,
    out_dir: str = "output/population",
    dt: float = 1.0,
    T_amb_c: float = 25.0,
    seed_base: int = 42,
    fixed_dwell_sec: int = 900,

    # ===== 只保留一个“取 N 个”参数 =====
    n_devices: Optional[int] = None,     # None: 跑全量；否则跑前 N 个

    # ===== timeseries 输出策略：每隔 K 台存一次（比 stride 降采样更省）=====
    save_timeseries: bool = True,
    save_every: int = 50,                # 例如 50：只保存第 1/51/101/... 台设备的过程数据
    timeseries_dirname: str = "timeseries",

    # ===== 断点续跑 =====
    resume: bool = False,
):
    """
    人群仿真：读取 CSV，每个 Device_ID 跑一次 run_simulation

    输出：
    - output/population/summary.csv：设备级结果宽表（每台设备 1 行）
    - output/population/timeseries/<Device_ID>.json：仅对“每隔 save_every 台设备”保存一次（可选）
    """
    ensure_dir(out_dir)
    ts_dir = os.path.join(out_dir, timeseries_dirname)
    if save_timeseries:
        ensure_dir(ts_dir)

    summary_path = os.path.join(out_dir, "summary.csv")
    done_ids: Set[str] = _read_existing_ids(summary_path) if resume else set()

    # ---------- 读取 CSV ----------
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # ---------- 只取前 N 个（如果设置了） ----------
    if n_devices is not None:
        if n_devices <= 0:
            print("n_devices <= 0, nothing to run.")
            return
        all_rows = all_rows[:n_devices]

    if not all_rows:
        print("CSV is empty or no rows selected.")
        return

    # ---------- summary 输出：resume 则 append，否则覆盖 ----------
    append_mode = resume and os.path.exists(summary_path)
    mode = "a" if append_mode else "w"

    # 统一字段（稳定、后续分析方便）
    fieldnames = [
        "Device_ID",
        "TTL_hours",
        "battery_capacity_mah",
        "avg_power_W",
        "max_Tb_C",
        "ratio_idle",
        "ratio_social",
        "ratio_video",
        "ratio_game",
        "signal_strength_avg",
        "background_app_usage_level",
        # 原始特征（强烈建议写入 summary，后续做回归/分组分析很有用）
        "avg_screen_on_hours_per_day",
        "gaming_hours_per_week",
        "video_streaming_hours_per_week",
        "device_age_months",
        "avg_charging_cycles_per_week",
        "avg_battery_temp_celsius",
        "fast_charging_usage_percent",
        "overnight_charging_freq_per_week",
        # 老化估计（解释性）
        "SOH_est",
        "aging_loss_est",
        "n_cycles_est",
        "lambda_eff_est",
    ]

    # 写表头：仅当非追加模式
    with open(summary_path, mode, newline="", encoding="utf-8") as wf:
        writer = csv.DictWriter(wf, fieldnames=fieldnames)
        if not append_mode:
            writer.writeheader()

        # ---------- 主循环 ----------
        saved_ts_count = 0
        ran_count = 0
        skipped_count = 0

        # save_every 的健壮性
        save_every = int(save_every)
        if save_every <= 0:
            save_every = 10**9  # 等价于“基本不保存”

        for idx, row in enumerate(all_rows):
            device_id = (row.get("Device_ID") or "").strip()
            if not device_id:
                continue

            if resume and device_id in done_ids:
                skipped_count += 1
                continue

            # 稳定 seed
            seed = seed_from_device(device_id, seed_base)

            scenario = build_scenario_from_row(row)
            usage_states = build_usage_states_from_row(row)
            cap_mah = _safe_float(row.get("battery_capacity_mah"), default=5000.0)

            aging = estimate_aging_from_row(row)
            aging_loss = float(aging.get("aging_loss", 0.10))

            result = run_simulation(
                scenario=scenario,
                dt=dt,
                T_amb=T_amb_c + 273.15,
                seed=seed,
                record=True,
                record_breakdown=True,
                record_uncorrected=False,
                usage_states_override=usage_states,
                device_params_override=None,
                battery_capacity_mah=cap_mah,
                battery_aging_loss=aging_loss,
                fixed_dwell_sec=fixed_dwell_sec,
            )

            ran_count += 1

            ttl_h = result["TTL"] / 3600.0
            power_avg = (sum(result["Power"]) / len(result["Power"])) if result.get("Power") else 0.0
            tb_max_c = (max(result["Tb"]) - 273.15) if result.get("Tb") else None
            sr = scenario.get("state_ratio", {})

            # 原始特征（写入 summary，后面你做统计/回归/分组都靠这些）
            f_screen = _safe_float(row.get("avg_screen_on_hours_per_day"))
            f_game_w = _safe_float(row.get("gaming_hours_per_week"))
            f_video_w = _safe_float(row.get("video_streaming_hours_per_week"))
            f_age_m = _safe_float(row.get("device_age_months"))
            f_cycles_w = _safe_float(row.get("avg_charging_cycles_per_week"))
            f_avgTb = _safe_float(row.get("avg_battery_temp_celsius"))
            f_fast = _safe_float(row.get("fast_charging_usage_percent"))
            f_overnight = _safe_float(row.get("overnight_charging_freq_per_week"))

            out_row = {
                "Device_ID": device_id,
                "TTL_hours": round(ttl_h, 6),
                "battery_capacity_mah": round(cap_mah, 3),
                "avg_power_W": round(power_avg, 6),
                "max_Tb_C": round(tb_max_c, 6) if tb_max_c is not None else "",

                "ratio_idle": round(float(sr.get("IDLE", 0.0)), 8),
                "ratio_social": round(float(sr.get("SOCIAL", 0.0)), 8),
                "ratio_video": round(float(sr.get("VIDEO", 0.0)), 8),
                "ratio_game": round(float(sr.get("GAME", 0.0)), 8),

                "signal_strength_avg": (row.get("signal_strength_avg") or ""),
                "background_app_usage_level": (row.get("background_app_usage_level") or ""),

                "avg_screen_on_hours_per_day": f_screen,
                "gaming_hours_per_week": f_game_w,
                "video_streaming_hours_per_week": f_video_w,
                "device_age_months": f_age_m,
                "avg_charging_cycles_per_week": f_cycles_w,
                "avg_battery_temp_celsius": f_avgTb,
                "fast_charging_usage_percent": f_fast,
                "overnight_charging_freq_per_week": f_overnight,

                "SOH_est": round(float(aging.get("SOH", 1.0)), 8),
                "aging_loss_est": round(float(aging_loss), 8),
                "n_cycles_est": round(float(aging.get("n_cycles", 0.0)), 4),
                "lambda_eff_est": round(float(aging.get("lambda_eff", 0.0)), 10),
            }

            writer.writerow(out_row)

            # ---------- 仅对每隔 save_every 台设备保存 timeseries ----------
            # 用 idx（原始顺序）还是 ran_count（实际跑过的序号）？
            # 推荐用 ran_count：resume 跳过时也能保持“每隔 K 个有效样本存一次”
            if save_timeseries and (ran_count - 1) % save_every == 0:
                out_path = os.path.join(ts_dir, f"{device_id}.json")

                payload = {
                    "meta": {
                        "Device_ID": device_id,
                        "seed": seed,
                        "battery_capacity_mah": cap_mah,
                        "T_amb_c": T_amb_c,
                        "dt": dt,
                        "fixed_dwell_sec": fixed_dwell_sec,
                        "scenario": scenario,
                        "usage_states": usage_states,
                        "features": {
                            "avg_screen_on_hours_per_day": f_screen,
                            "gaming_hours_per_week": f_game_w,
                            "video_streaming_hours_per_week": f_video_w,
                            "device_age_months": f_age_m,
                            "avg_charging_cycles_per_week": f_cycles_w,
                            "avg_battery_temp_celsius": f_avgTb,
                            "fast_charging_usage_percent": f_fast,
                            "overnight_charging_freq_per_week": f_overnight,
                            "signal_strength_avg": (row.get("signal_strength_avg") or ""),
                            "background_app_usage_level": (row.get("background_app_usage_level") or ""),
                        },
                        "aging": {
                            "SOH": aging.get("SOH", 1.0),
                            "aging_loss": aging_loss,
                            "n_cycles": aging.get("n_cycles", 0.0),
                            "lambda_eff": aging.get("lambda_eff", 0.0),
                        },
                    },
                    "time": result.get("time", []),
                    "SOC": result.get("SOC", []),
                    "Tb": result.get("Tb", []),
                    "Power": result.get("Power", []),
                    "State": result.get("State", []),
                    "Power_screen": result.get("Power_screen", []),
                    "Power_cpu": result.get("Power_cpu", []),
                    "Power_radio": result.get("Power_radio", []),
                    "Power_background": result.get("Power_background", []),
                }

                with open(out_path, "w", encoding="utf-8") as jf:
                    json.dump(payload, jf, ensure_ascii=False)

                saved_ts_count += 1

            print(f"[{idx+1}/{len(all_rows)}] Device {device_id}: TTL={ttl_h:.2f}h")

    print(f"\nSaved summary -> {summary_path}")
    if save_timeseries:
        print(f"Saved timeseries (sampled) -> {ts_dir}")
        print(f"Timeseries saved count = {saved_ts_count}")
    if resume:
        print(f"Skipped (resume) = {skipped_count}, Ran = {ran_count}")


if __name__ == "__main__":
    run_population(
        csv_path="smartphone_battery_features.csv",
        out_dir="output/population",
        dt=1.0,
        T_amb_c=25.0,
        seed_base=42,
        fixed_dwell_sec=900,

        # ===== 只跑前 N 个（想全跑就设 None）=====
        n_devices=None,

        # ===== 每隔多少台设备保存一次过程性 JSON =====
        save_timeseries=True,
        save_every=500,

        # ===== resume =====
        resume=False,
    )
