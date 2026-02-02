# src/experiments/sensitivity/theme_physics.py
from __future__ import annotations

import json
import copy
from pathlib import Path

from src.visualization.tornado import plot_tornado, TornadoConfig
from .common import run_from_meta

def main(timeseries_json: str, out_dir: str = "output/sensitivity/theme_physics"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(timeseries_json).read_text(encoding="utf-8"))
    meta = payload["meta"]

    base = run_from_meta(meta)

    effects_ttl = {}
    effects_tb = {}

    # 1) T_amb: low=0, high=35 (相对 baseline 25)
    low = run_from_meta(meta, T_amb_c=0.0)
    high = run_from_meta(meta, T_amb_c=35.0)
    effects_ttl["T_amb_c (0 / 35)"] = (low["TTL_hours"] - base["TTL_hours"], high["TTL_hours"] - base["TTL_hours"])
    effects_tb["T_amb_c (0 / 35)"] = (low["max_Tb_C"] - base["max_Tb_C"], high["max_Tb_C"] - base["max_Tb_C"])

    def factor_param(name: str, key: str, lo_f: float, hi_f: float):
        # 需要你给 run_simulation 加 battery_params_override 支持
        lo = run_from_meta(meta, battery_params_override={key: lo_f})
        hi = run_from_meta(meta, battery_params_override={key: hi_f})
        effects_ttl[name] = (lo["TTL_hours"] - base["TTL_hours"], hi["TTL_hours"] - base["TTL_hours"])
        effects_tb[name] = (lo["max_Tb_C"] - base["max_Tb_C"], hi["max_Tb_C"] - base["max_Tb_C"])

    factor_param("alpha (×0.8 / ×1.2)", "alpha", 0.03 * 0.8, 0.03 * 1.2)
    factor_param("h (×0.8 / ×1.2)", "h", 0.15 * 0.8, 0.15 * 1.2)
    factor_param("C_th (×0.8 / ×1.2)", "C_th", 60.0 * 0.8, 60.0 * 1.2)
    factor_param("eta_heat (×0.8 / ×1.2)", "eta_heat", 0.65 * 0.8, 0.65 * 1.2)

    # aging_loss: 直接覆盖 aging_loss（注意 clamp）
    base_aging = float(meta.get("aging_loss", meta.get("aging", {}).get("aging_loss", 0.15)))
    lo_aging = max(0.0, min(0.95, base_aging * 0.7))
    hi_aging = max(0.0, min(0.95, base_aging * 1.3))
    lo = run_from_meta(meta, battery_aging_loss=lo_aging)
    hi = run_from_meta(meta, battery_aging_loss=hi_aging)
    effects_ttl["aging_loss (×0.7 / ×1.3)"] = (lo["TTL_hours"] - base["TTL_hours"], hi["TTL_hours"] - base["TTL_hours"])
    effects_tb["aging_loss (×0.7 / ×1.3)"] = (lo["max_Tb_C"] - base["max_Tb_C"], hi["max_Tb_C"] - base["max_Tb_C"])

    # plot
    plot_tornado(
        effects_ttl,
        cfg=TornadoConfig(title="Theme A: Physics - Tornado (ΔTTL hours)", xlabel="ΔTTL (hours)", figsize=(10.8, 6.6)),
        save_path=str(out / "tornado_physics_dTTL.pdf"),
        show=False,
    )
    plot_tornado(
        effects_tb,
        cfg=TornadoConfig(title="Theme A: Physics - Tornado (Δmax Tb °C)", xlabel="Δmax Tb (°C)", figsize=(10.8, 6.6)),
        save_path=str(out / "tornado_physics_dTb.pdf"),
        show=False,
    )

if __name__ == "__main__":
    main("output/population/timeseries/some_device.json")
