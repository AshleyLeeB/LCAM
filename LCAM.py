#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paths import AMLWORLD


def _patch_dataset_configs() -> None:
    import dataset_configs as dc

    dc.DATASETS = {
        k: replace(v, data_dir=str(AMLWORLD)) if k == "amlworld" else v
        for k, v in dc.DATASETS.items()
    }


def main() -> None:
    p = argparse.ArgumentParser(description="LCAM on AMLWorld")
    p.add_argument("--start_seed", type=int, default=10)
    p.add_argument("--end_seed", type=int, default=18)
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--min_epochs", type=int, default=None)
    p.add_argument("--output_root", type=str, default=str(ROOT / "results"))
    args = p.parse_args()

    _patch_dataset_configs()

    argv = [
        "run_lcam.py",
        "--dataset",
        "amlworld",
        "--only",
        "lcam",
        "--start_seed",
        str(args.start_seed),
        "--end_seed",
        str(args.end_seed),
        "--output_root",
        str(Path(args.output_root).resolve()),
    ]
    if args.min_epochs is not None:
        argv.extend(["--min_epochs", str(args.min_epochs)])
    if args.no_early_stop:
        argv.append("--no_early_stop")

    sys.argv = argv
    import run_lcam as rs

    rs.main()


if __name__ == "__main__":
    main()
