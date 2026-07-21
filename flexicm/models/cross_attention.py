"""Cross-attention with conditional prompts inside TIC window attention.

Q is computed from feature tokens only; K,V from [features; prompts] (paper Eq.5).
Uses the frozen TIC qkv projection weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexicm.layers.layers import window_partition, window_reverse


def window_attention_with_prompt(attn_module, x_windows, prompts, mask=None):
    """
    Args:
        attn_module: WindowAttention with attributes qkv, proj, scale, num_heads, ...
        x_windows: (B*nW, N, C) feature tokens in each window
        prompts: (B*nW, Np, C) prompt tokens aligned with windows
        mask: optional attention mask for shifted windows (N x N); prompt cols get 0
    Returns:
        out: (B*nW, N, C)
    """
    B_, N, C = x_windows.shape
    Np = prompts.shape[1]
    num_heads = attn_module.num_heads
    head_dim = C // num_heads
    scale = attn_module.scale

    # Q from features only
    qkv_f = attn_module.qkv(x_windows).reshape(B_, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k_f, v_f = qkv_f[0], qkv_f[1], qkv_f[2]

    # K,V from prompts
    qkv_p = attn_module.qkv(prompts).reshape(B_, Np, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    k_p, v_p = qkv_p[1], qkv_p[2]

    k = torch.cat([k_f, k_p], dim=2)  # (B_, heads, N+Np, head_dim)
    v = torch.cat([v_f, v_p], dim=2)

    q = q * scale
    attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N+Np)

    # Relative position bias only on feature-feature block
    relative_position_bias = attn_module.relative_position_bias_table[
        attn_module.relative_position_index.view(-1)
    ].view(
        attn_module.window_size[0] * attn_module.window_size[1],
        attn_module.window_size[0] * attn_module.window_size[1],
        -1,
    )
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
    attn[:, :, :, :N] = attn[:, :, :, :N] + relative_position_bias.unsqueeze(0)

    if mask is not None:
        # mask: (nW, N, N) -> pad prompt columns with 0
        nW = mask.shape[0]
        mask_pad = F.pad(mask, (0, Np), value=0.0)  # (nW, N, N+Np)
        attn = attn.view(-1, nW, num_heads, N, N + Np) + mask_pad.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, N, N + Np)

    attn = attn_module.softmax(attn)
    attn = attn_module.attn_drop(attn)
    out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    out = attn_module.proj(out)
    out = attn_module.proj_drop(out)
    return out


def swin_block_forward_with_prompt(block, x, x_size, prompts):
    """Run one SwinTransformerBlock with optional prompt cross-attention.

    Args:
        block: SwinTransformerBlock
        x: (B, H*W, C)
        x_size: (H, W)
        prompts: (B*nW, Np, C) or None
    """
    H, W = x_size
    B, L, C = x.shape
    shortcut = x
    x = block.norm1(x)
    x = x.view(B, H, W, C)

    if block.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-block.shift_size, -block.shift_size), dims=(1, 2))
        attn_mask = block.calculate_mask(x_size).to(x.device)
    else:
        shifted_x = x
        attn_mask = None

    x_windows = window_partition(shifted_x, block.window_size)
    x_windows = x_windows.view(-1, block.window_size * block.window_size, C)

    if prompts is not None:
        attn_windows = window_attention_with_prompt(block.attn, x_windows, prompts, mask=attn_mask)
    else:
        attn_windows, _ = block.attn(x_windows, mask=attn_mask)

    attn_windows = attn_windows.view(-1, block.window_size, block.window_size, C)
    shifted_x = window_reverse(attn_windows, block.window_size, H, W)

    if block.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(block.shift_size, block.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    x = shortcut + block.drop_path(x)
    x = x + block.drop_path(block.mlp(block.norm2(x)))
    return x


def rstb_forward_with_prompt(rstb, x, x_size, prompts=None):
    """RSTB forward; if prompts is set, inject into every block of the RSTB."""
    out = rstb.patch_embed(x)
    for blk in rstb.residual_group.blocks:
        out = swin_block_forward_with_prompt(blk, out, x_size, prompts)
    return rstb.patch_unembed(out, x_size) + x
