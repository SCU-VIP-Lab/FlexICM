#!/usr/bin/env python3
"""Quick sanity check: build TAIC/C-TAIC and run one forward pass."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch
from flexicm.models import TAIC, CTAIC


def main():
    device = "cpu"
    x = torch.rand(1, 3, 256, 256, device=device)
    taic = TAIC(out_channels=128).to(device)
    taic.freeze_base_codec()
    out = taic(x)
    assert out["h"].shape == (1, 128, 64, 64), out["h"].shape
    assert out["y_hat"].shape == (1, 192, 16, 16), out["y_hat"].shape

    ctaic = CTAIC(out_channels=128).to(device)
    ctaic.freeze_for_stage2()
    out2 = ctaic(x, y_b_hat=out["y_hat"], use_condition=True)
    assert out2["h"].shape == (1, 128, 64, 64)
    print("sanity check passed")


if __name__ == "__main__":
    main()
