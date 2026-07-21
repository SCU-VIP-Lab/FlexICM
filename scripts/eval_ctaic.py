#!/usr/bin/env python3
"""Eval / test entry for C-TAIC (placeholder).

Full multi-task rate-accuracy evaluation will be added later.
This script only validates that required checkpoints exist and are not PLACEHOLDERs.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from flexicm.utils.train_utils import load_yaml_config


def resolve_ckpt(path: str, label: str) -> str:
    if not path:
        raise FileNotFoundError(f"{label}: checkpoint path is empty")
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    placeholder = os.path.join(os.path.dirname(path), "PLACEHOLDER")
    if os.path.isfile(placeholder) and not os.path.isfile(path):
        raise FileNotFoundError(
            f"{label}: checkpoint not ready (PLACEHOLDER still present):\n  {placeholder}\n"
            f"Expected real weights at:\n  {path}\n"
            "See checkpoints/README.md"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label}: missing checkpoint: {path}")
    return path


def main(argv):
    parser = argparse.ArgumentParser("Eval FlexICM C-TAIC (stub)")
    parser.add_argument("-c", "--config", required=True, help="configs/eval/ctaic_*.yaml")
    args = parser.parse_args(argv)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    cfg = load_yaml_config(cfg_path)

    ext = resolve_ckpt(cfg["checkpoint"], "extension C-TAIC")
    base = resolve_ckpt(cfg["base_taic_checkpoint"], "base TAIC")
    print(f"[eval_ctaic stub] extension ckpt ok: {ext}")
    print(f"[eval_ctaic stub] base ckpt ok: {base}")
    print(f"[eval_ctaic stub] scenario={cfg.get('scenario')} quality={cfg.get('quality_level')}")
    print("[eval_ctaic stub] Full metric evaluation is not implemented yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
