# src/experiments/sensitivity/run_all.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.sensitivity.theme_physics import main as run_physics
from src.sensitivity.theme_usage_mapping import main as run_usage
from src.sensitivity.theme_hsmm import main as run_hsmm


def parse_args():
    ap = argparse.ArgumentParser("Run all sensitivity themes and output tornado charts.")
    ap.add_argument("--json", required=True, help="timeseries json saved by population runner")
    ap.add_argument("--out", default="output/sensitivity/run_all", help="output directory")
    ap.add_argument("--K", type=int, default=30, help="MC samples for HSMM theme (theme_hsmm)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 统一把三套输出放到子目录，避免文件名冲突
    out_physics = str(out / "theme_physics")
    out_usage = str(out / "theme_usage_mapping")
    out_hsmm = str(out / "theme_hsmm")

    # 1) 物理/电池主题（Physics）
    run_physics(args.json, out_dir=out_physics)

    # 2) usage mapping 主题（Usage）
    run_usage(args.json, out_dir=out_usage)

    # 3) HSMM dwell 主题（HSMM）
    run_hsmm(args.json, out_dir=out_hsmm, K=args.K)

    print("\n=== DONE ===")
    print(f"- Physics tornado: {out_physics}")
    print(f"- Usage tornado:   {out_usage}")
    print(f"- HSMM tornado:    {out_hsmm}")


if __name__ == "__main__":
    main()
