"""Conditional Prompt Generator and Condition Generator for C-TAIC (Fig. 2)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models.utils import conv, deconv


def window_partition_features(x, window_size):
    """Partition (B,C,H,W) into (B*nW, window_size*window_size, C)."""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, C)


class MaskGenerator(nn.Module):
    """Lightweight soft mask over fused base-latent / image features."""

    def __init__(self, img_channels=3, latent_channels=192, out_channels=192, mid=64):
        super().__init__()
        self.img_stem = nn.Sequential(
            conv(img_channels, mid, kernel_size=5, stride=2),  # H/2
            nn.GELU(),
            conv(mid, mid, kernel_size=3, stride=2),  # H/4
            nn.GELU(),
            conv(mid, mid, kernel_size=3, stride=2),  # H/8
            nn.GELU(),
            conv(mid, out_channels, kernel_size=3, stride=2),  # H/16
        )
        self.latent_proj = nn.Conv2d(latent_channels, out_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, mid, 1),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, y_b_hat):
        fx = self.img_stem(x)
        fy = self.latent_proj(y_b_hat)
        if fx.shape[-2:] != fy.shape[-2:]:
            fx = F.interpolate(fx, size=fy.shape[-2:], mode="bilinear", align_corners=False)
        m = self.fuse(torch.cat([fx, fy], dim=1))
        return m, fx


class ConditionalPromptGenerator(nn.Module):
    """Generate multi-scale prompts C2 / C4 from (x, y_b_hat).

    Prompts are returned as lists of per-window token tensors compatible with
    TIC window_size=8 attention:
      C2 -> first encoder STB at H/2  (16 prompt tokens / window)
      C4 -> second encoder STB at H/4 (16 prompt tokens / window)
    """

    def __init__(self, latent_channels=192, prompt_dim=128, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.prompt_dim = prompt_dim
        self.mask_gen = MaskGenerator(
            img_channels=3, latent_channels=latent_channels, out_channels=latent_channels
        )
        self.img_to_latent = nn.Sequential(
            conv(3, 64, kernel_size=5, stride=2),
            nn.GELU(),
            conv(64, 128, kernel_size=3, stride=2),
            nn.GELU(),
            conv(128, latent_channels, kernel_size=3, stride=2),
            nn.GELU(),
            conv(latent_channels, latent_channels, kernel_size=3, stride=2),
        )
        # H/16 -> H/8 -> H/4
        self.up1 = deconv(latent_channels, latent_channels, kernel_size=3, stride=2)
        self.up2 = deconv(latent_channels, latent_channels, kernel_size=3, stride=2)
        self.proj_c4 = nn.Conv2d(latent_channels, prompt_dim, 1)
        self.proj_c2 = nn.Conv2d(latent_channels, prompt_dim, 1)

    def _to_window_prompts(self, feat, target_hw):
        """Map spatial prompt feature to windows of the target STB resolution.

        Each TIC window (window_size x window_size) at target_hw covers a
        (window_size/scale) block on ``feat``, yielding 16 tokens when
        feat is 2x down relative to target (paper ratio 1/4 of 64).
        """
        B, C, Hf, Wf = feat.shape
        Ht, Wt = target_hw
        # Align spatial size: target windows expect feat at Ht/2 x Wt/2
        expect_h, expect_w = Ht // 2, Wt // 2
        if (Hf, Wf) != (expect_h, expect_w):
            feat = F.interpolate(feat, size=(expect_h, expect_w), mode="bilinear", align_corners=False)
        # Partition with window_size//2 so each target window gets 16 tokens
        ws = self.window_size // 2
        return window_partition_features(feat, ws)

    def forward(self, x, y_b_hat, sizes_h2, sizes_h4):
        """
        Args:
            x: input image (B,3,H,W)
            y_b_hat: base-layer latent after AD (B,192,H/16,W/16)
            sizes_h2: (H/2, W/2) of first STB
            sizes_h4: (H/4, W/4) of second STB
        Returns:
            prompt_c2, prompt_c4: (B*nW, 16, prompt_dim)
        """
        m, _ = self.mask_gen(x, y_b_hat)
        fx = self.img_to_latent(x)
        if fx.shape[-2:] != y_b_hat.shape[-2:]:
            fx = F.interpolate(fx, size=y_b_hat.shape[-2:], mode="bilinear", align_corners=False)
        if m.shape[-2:] != y_b_hat.shape[-2:]:
            m = F.interpolate(m, size=y_b_hat.shape[-2:], mode="bilinear", align_corners=False)
        f_sum = m * fx + (1.0 - m) * y_b_hat

        f_h8 = self.up1(f_sum)   # H/8
        f_h4 = self.up2(f_h8)    # H/4

        c4_map = self.proj_c4(f_h8)
        c2_map = self.proj_c2(f_h4)

        prompt_c2 = self._to_window_prompts(c2_map, sizes_h2)
        prompt_c4 = self._to_window_prompts(c4_map, sizes_h4)
        return prompt_c2, prompt_c4


class ConditionGenerator(nn.Module):
    """Decoder-side condition Cd from base latent y_b_hat.

    Lightweight: Deconv (H/16->H/8) + 1x1 Linear/Conv to match Task Connector channels.
    """

    def __init__(self, latent_channels=192, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            deconv(latent_channels, out_channels, kernel_size=3, stride=2),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, y_b_hat):
        return self.net(y_b_hat)
