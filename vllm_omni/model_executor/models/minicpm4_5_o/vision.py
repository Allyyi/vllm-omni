# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenBMB Team.
"""MiniCPM-O 4.5 vision components: SigLip encoder, Resampler, and audio projector."""

from functools import partial

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# 2-D sincos positional embedding utilities
# ---------------------------------------------------------------------------


def get_1d_sincos_pos_embed_from_grid_new(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    embed_dim: output dimension for each position
    pos: positions of shape (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim: int, image_size) -> np.ndarray:
    """
    image_size: int or (height, width)
    return: pos_embed of shape [H, W, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


# ---------------------------------------------------------------------------
# Perceiver Resampler
# ---------------------------------------------------------------------------


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross-attention layer using
    learnable queries and 2d sincos positional embeddings.

    Outputs: (batch_size, num_queries, embed_dim)
    """

    def __init__(
        self,
        num_queries: int,
        embed_dim: int,
        num_heads: int,
        kv_dim: int | None = None,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        adaptive: bool = False,
        max_size: tuple[int, int] = (70, 70),
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim**-0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)

    def _set_2d_pos_cache(self, max_size, device="cpu"):
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes: torch.Tensor, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)

    def _repeat(self, query, n: int):
        return query.unsqueeze(1).repeat(1, n, 1)

    def forward(self, x: torch.Tensor, tgt_sizes: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))
            key_padding_mask[i, patch_len[i] :] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(
            1, 0, 2
        )  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            x + pos_embed,  # L * B * D
            x,
            key_padding_mask=key_padding_mask,
        )[0]
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


# ---------------------------------------------------------------------------
# Audio projection / pooling helpers (used by thinker)
# ---------------------------------------------------------------------------


class MultiModalProjector(nn.Module):
    """2-layer MLP that projects audio encoder features to LLM hidden size."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.relu(self.linear1(audio_features))
        hidden_states = self.linear2(hidden_states)
        return hidden_states
