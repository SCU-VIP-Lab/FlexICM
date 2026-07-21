"""Shared training helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from datetime import datetime

import torch
import torch.nn as nn
import yaml


def setup_logger(log_path: str):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(log_formatter)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(log_formatter)
    root.addHandler(sh)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_checkpoint(state, is_best, out_dir, filename="checkpoint.pth.tar"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    torch.save(state, path)
    if is_best:
        best = os.path.join(out_dir, "checkpoint_best_loss.pth.tar")
        torch.save(state, best)
    logging.info(f"Saved checkpoint to {path} (best={is_best})")


def load_checkpoint_dict(path: str, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    # strip module.
    out = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out, ckpt if isinstance(ckpt, dict) else {"state_dict": state}


def adamw_trainable(model: nn.Module, lr: float, weight_decay: float = 0.01):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def exp_dir(root: str, exp_name: str, quality_level) -> str:
    path = os.path.join(root, exp_name, str(quality_level))
    os.makedirs(path, exist_ok=True)
    return path
