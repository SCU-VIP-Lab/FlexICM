"""C-TAIC: Conditional Task-Adaptive Image Coding (FlexICM extension layer, Fig. 2).

Stage-1 (TAIC mode): train SFMA + Task Connector (no base-layer condition).
Stage-2 (C-TAIC mode): freeze all except Conditional Prompt Generator + Condition Generator;
                        use base-layer y_b_hat for cross-attention prompts and decoder Cd.
"""

import torch
import torch.nn as nn

from flexicm.models.taic import TAIC
from flexicm.models.conditional import ConditionalPromptGenerator, ConditionGenerator


class CTAIC(TAIC):
    def __init__(
        self,
        N=128,
        M=192,
        input_resolution=(256, 256),
        out_channels=128,
        in_channel=3,
    ):
        super().__init__(
            N=N,
            M=M,
            input_resolution=input_resolution,
            out_channels=out_channels,
            in_channel=in_channel,
        )
        self.prompt_generator = ConditionalPromptGenerator(
            latent_channels=M, prompt_dim=N, window_size=8
        )
        self.condition_generator = ConditionGenerator(
            latent_channels=M, out_channels=N
        )

    def forward(self, x, y_b_hat=None, use_condition=True):
        """
        Args:
            x: input image
            y_b_hat: base-layer latent after AD (H/16 x W/16 x M). Required when use_condition.
            use_condition: if False, operate as TAIC (stage-1 / graceful degradation)
        """
        prompts = None
        condition = None
        if use_condition and y_b_hat is not None:
            H, W = x.shape[2], x.shape[3]
            c2, c4 = self.prompt_generator(
                x, y_b_hat, sizes_h2=(H // 2, W // 2), sizes_h4=(H // 4, W // 4)
            )
            prompts = {"c2": c2, "c4": c4}
            condition = self.condition_generator(y_b_hat)
        return super().forward(x, prompts=prompts, condition=condition)

    def compress(self, x, y_b_hat=None, use_condition=True):
        prompts = None
        if use_condition and y_b_hat is not None:
            H, W = x.shape[2], x.shape[3]
            c2, c4 = self.prompt_generator(
                x, y_b_hat, sizes_h2=(H // 2, W // 2), sizes_h4=(H // 4, W // 4)
            )
            prompts = {"c2": c2, "c4": c4}
        return super().compress(x, prompts=prompts)

    def decompress(self, strings, shape, x_size=None, y_b_hat=None, use_condition=True):
        condition = None
        if use_condition and y_b_hat is not None:
            condition = self.condition_generator(y_b_hat)
        return super().decompress(strings, shape, x_size=x_size, condition=condition)

    def freeze_for_stage1(self):
        """Train SFMA + Task Connector only (TAIC mode)."""
        for name, p in self.named_parameters():
            train = ("encoder_sfmas" in name) or ("task_connector" in name)
            p.requires_grad = train
        # keep prompt/condition gens frozen in stage-1
        for p in self.prompt_generator.parameters():
            p.requires_grad = False
        for p in self.condition_generator.parameters():
            p.requires_grad = False

    def freeze_for_stage2(self):
        """Train Conditional Prompt Generator + Condition Generator only."""
        for p in self.parameters():
            p.requires_grad = False
        for p in self.prompt_generator.parameters():
            p.requires_grad = True
        for p in self.condition_generator.parameters():
            p.requires_grad = True

    def load_taic_checkpoint(self, state_dict, strict=False):
        """Initialize from a trained TAIC (extension-task) checkpoint."""
        own = self.state_dict()
        filtered = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            if nk in own and own[nk].shape == v.shape:
                filtered[nk] = v
        return self.load_state_dict(filtered, strict=False)
