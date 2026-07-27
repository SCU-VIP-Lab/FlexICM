"""TAIC: Task-Adaptive Image Coding (FlexICM base layer, Fig. 1).

Encoder: frozen TIC + trainable SFMA (same placement as AdaptiveICMH).
Decoder: frozen TIC g_s0 (STB) + g_s1 (Deconv) + trainable Task Connector
         -> h at H/4 x W/4 x out_channels (no full image reconstruction).
"""

import math

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import conv, deconv, update_registered_buffers
from timm.models.layers import trunc_normal_

from flexicm.layers.layers import RSTB
from flexicm.models.sfma import SFMA
from flexicm.models.task_connector import TaskConnector

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def ste_round(x):
    return torch.round(x) - x.detach() + x


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class TAIC(nn.Module):
    def __init__(
        self,
        N=128,
        M=192,
        input_resolution=(256, 256),
        out_channels=128,
        in_channel=3,
    ):
        super().__init__()
        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.0
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.0
        attn_drop_rate = 0.0
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        use_checkpoint = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.N = N
        self.M = M
        self.out_channels = out_channels

        # Encoder-side SFMA (trainable)
        self.encoder_sfmas = nn.Sequential(SFMA(N), SFMA(N), SFMA(N))

        # ---- encoder (frozen TIC) ----
        self.g_a0 = conv(in_channel, N, kernel_size=5, stride=2)
        self.g_a1 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]) : sum(depths[:1])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 4, input_resolution[1] // 4),
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]) : sum(depths[:2])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 8, input_resolution[1] // 8),
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]) : sum(depths[:3])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = RSTB(
            dim=M,
            input_resolution=(input_resolution[0] // 16, input_resolution[1] // 16),
            depth=depths[3],
            num_heads=num_heads[3],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:3]) : sum(depths[:4])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        # ---- hyperprior (frozen TIC) ----
        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 32, input_resolution[1] // 32),
            depth=depths[4],
            num_heads=num_heads[4],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:4]) : sum(depths[:5])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 64, input_resolution[1] // 64),
            depth=depths[5],
            num_heads=num_heads[5],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:5]) : sum(depths[:6])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        depths_rev = depths[::-1]
        num_heads_rev = num_heads[::-1]
        self.h_s0 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 64, input_resolution[1] // 64),
            depth=depths_rev[0],
            num_heads=num_heads_rev[0],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_rev[:0]) : sum(depths_rev[:1])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.h_s1 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s2 = RSTB(
            dim=N,
            input_resolution=(input_resolution[0] // 32, input_resolution[1] // 32),
            depth=depths_rev[1],
            num_heads=num_heads_rev[1],
            window_size=window_size // 2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_rev[:1]) : sum(depths_rev[:2])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.h_s3 = deconv(N, M * 2, kernel_size=3, stride=2)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

        # ---- partial decoder (frozen TIC g_s0 + g_s1) ----
        self.g_s0 = RSTB(
            dim=M,
            input_resolution=(input_resolution[0] // 16, input_resolution[1] // 16),
            depth=depths_rev[2],
            num_heads=num_heads_rev[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths_rev[:2]) : sum(depths_rev[:3])],
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )
        self.g_s1 = deconv(M, N, kernel_size=3, stride=2)  # H/16 -> H/8

        # ---- trainable Task Connector ----
        self.task_connector = TaskConnector(
            in_channels=N, mid_channels=N, out_channels=out_channels
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def g_a(self, x, x_size=None, prompts=None):
        """
        Args:
            prompts: optional dict {"c2": tensor, "c4": tensor} for C-TAIC
        """
        if x_size is None:
            x_size = x.shape[2:4]
        x = self.g_a0(x)
        if prompts is not None and prompts.get("c2") is not None:
            from flexicm.models.cross_attention import rstb_forward_with_prompt

            x = rstb_forward_with_prompt(
                self.g_a1, x, (x_size[0] // 2, x_size[1] // 2), prompts["c2"]
            )
        else:
            x, _ = self.g_a1(x, (x_size[0] // 2, x_size[1] // 2))
        x = self.encoder_sfmas[0](x)
        x = self.g_a2(x)

        if prompts is not None and prompts.get("c4") is not None:
            from flexicm.models.cross_attention import rstb_forward_with_prompt

            x = rstb_forward_with_prompt(
                self.g_a3, x, (x_size[0] // 4, x_size[1] // 4), prompts["c4"]
            )
        else:
            x, _ = self.g_a3(x, (x_size[0] // 4, x_size[1] // 4))
        x = self.encoder_sfmas[1](x)
        x = self.g_a4(x)

        x, _ = self.g_a5(x, (x_size[0] // 8, x_size[1] // 8))
        x = self.encoder_sfmas[2](x)
        x = self.g_a6(x)
        x, _ = self.g_a7(x, (x_size[0] // 16, x_size[1] // 16))
        return x

    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 16, x.shape[3] * 16)
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0] // 64, x_size[1] // 64))
        return x

    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = (x.shape[2] * 64, x.shape[3] * 64)
        x, _ = self.h_s0(x, (x_size[0] // 64, x_size[1] // 64))
        x = self.h_s1(x)
        x, _ = self.h_s2(x, (x_size[0] // 32, x_size[1] // 32))
        x = self.h_s3(x)
        return x

    def decode_feature(self, y_hat, x_size=None, condition=None):
        """Map quantized latent to task feature h."""
        if x_size is None:
            x_size = (y_hat.shape[2] * 16, y_hat.shape[3] * 16)
        x, _ = self.g_s0(y_hat, (x_size[0] // 16, x_size[1] // 16))
        x = self.g_s1(x)  # H/8 x W/8 x N
        h = self.task_connector(x, condition=condition)
        return h

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def forward(self, x, prompts=None, condition=None):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size, prompts=prompts)
        z = self.h_a(y, x_size)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = ste_round(z - z_offset) + z_offset
        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        y_hat = ste_round(y - means_hat) + means_hat
        h = self.decode_feature(y_hat, x_size, condition=condition)
        return {
            "h": h,
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, prompts=None):
        x_size = (x.shape[2], x.shape[3])
        y = self.g_a(x, x_size, prompts=prompts)
        z = self.h_a(y, x_size)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "x_size": x_size}

    def decompress(self, strings, shape, x_size=None, condition=None):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        if x_size is None:
            x_size = (shape[0] * 64, shape[1] * 64)
        gaussian_params = self.h_s(z_hat, x_size)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, means=means_hat)
        h = self.decode_feature(y_hat, x_size, condition=condition)
        return {"h": h, "y_hat": y_hat}

    def freeze_base_codec(self):
        """Freeze TIC weights; keep SFMA + Task Connector trainable."""
        for name, p in self.named_parameters():
            if ("sfma" in name.lower()) or ("task_connector" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def trainable_parameter_names(self):
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated = False
        for m in self.children():
            if isinstance(m, EntropyBottleneck):
                updated |= m.update(force=force)
        return updated

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, strict=strict)

    def load_base_codec(self, state_dict, strict=False):
        """Load pretrained TIC / TIC-SFMA weights into matching modules."""
        own = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            # skip decoder stages after g_s1 and any decoder SFMAs
            if nk.startswith("g_s2") or nk.startswith("g_s3") or nk.startswith("g_s4"):
                continue
            if nk.startswith("g_s5") or nk.startswith("g_s6") or nk.startswith("g_s7"):
                continue
            if "decoder_sfmas" in nk:
                continue
            if nk in own and own[nk].shape == v.shape:
                filtered[nk] = v
        missing_unexpected = self.load_state_dict(filtered, strict=False)
        return missing_unexpected
