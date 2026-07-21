"""Lightweight Task Connector (Fig. 1 / Fig. 2).

After frozen TIC decoder stages STB(g_s0)+Deconv(g_s1) at H/8 x W/8 x N,
the Task Connector produces h at H/4 x W/4 x out_channels via:
  residual(Linear -> DW-Conv -> Linear) -> Deconv
"""

import torch
import torch.nn as nn
from compressai.models.utils import deconv


class TaskConnector(nn.Module):
    def __init__(self, in_channels=128, mid_channels=128, out_channels=128):
        super().__init__()
        self.linear1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.dw = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels
        )
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(mid_channels, in_channels, kernel_size=1)
        # H/8 -> H/4
        self.upsample = deconv(in_channels, out_channels, kernel_size=3, stride=2)

    def forward(self, x, condition=None):
        """
        Args:
            x: (B, C, H/8, W/8) feature after frozen STB+Deconv
            condition: optional (B, C, H/8, W/8) decoder-side condition Cd
        """
        if condition is not None:
            x = x + condition
        residual = x
        y = self.linear2(self.act(self.dw(self.linear1(x))))
        y = residual + y
        return self.upsample(y)
