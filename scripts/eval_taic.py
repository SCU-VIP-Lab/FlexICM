#!/usr/bin/env python3
"""Eval / test entry for TAIC (placeholder).

Full rate-accuracy evaluation (mAP / mIoU / PQ / OKS) will be added later.
This script only validates that the requested checkpoint exists and is not a PLACEHOLDER.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from flexicm.utils.train_utils import load_yaml_config


def resolve_ckpt(path: str) -> str:
    if not path:
        raise FileNotFoundError("checkpoint path is empty")
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    placeholder = os.path.join(os.path.dirname(path), "PLACEHOLDER")
    if os.path.isfile(placeholder) and not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not ready (PLACEHOLDER still present):\n  {placeholder}\n"
            f"Expected real weights at:\n  {path}\n"
            "See checkpoints/README.md"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    if os.path.basename(path) == "PLACEHOLDER" or path.endswith(".txt"):
        raise FileNotFoundError(f"Refusing to load placeholder file: {path}")
    return path


def main(argv):
    parser = argparse.ArgumentParser("Eval FlexICM TAIC (stub)")
    parser.add_argument("-c", "--config", required=True, help="configs/eval/taic_*.yaml")
    args = parser.parse_args(argv)
    cfg = load_yaml_config(args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config))

    ckpt = resolve_ckpt(cfg["checkpoint"])
    print(f"[eval_taic stub] checkpoint ok: {ckpt}")
    print(f"[eval_taic stub] task={cfg.get('task')} quality={cfg.get('quality_level')}")
    print("[eval_taic stub] Full metric evaluation is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
