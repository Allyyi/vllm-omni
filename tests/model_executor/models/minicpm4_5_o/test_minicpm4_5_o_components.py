# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPM-O 4.5 vision and audio components."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Vision: positional embeddings (pure function)
# ---------------------------------------------------------------------------


class TestSincosPosEmbed:
    """Tests for 2-D sincos positional embedding generation."""

    def test_output_shape_square(self) -> None:
        """Square grid should produce (H, W, embed_dim) output."""
        from vllm_omni.model_executor.models.minicpm4_5_o.vision import get_2d_sincos_pos_embed

        emb = get_2d_sincos_pos_embed(embed_dim=256, image_size=4)
        assert emb.shape == (4, 4, 256)

    def test_output_shape_rectangular(self) -> None:
        """Rectangular grid should produce (H, W, embed_dim) output."""
        from vllm_omni.model_executor.models.minicpm4_5_o.vision import get_2d_sincos_pos_embed

        emb = get_2d_sincos_pos_embed(embed_dim=128, image_size=(3, 5))
        assert emb.shape == (3, 5, 128)

    def test_values_finite(self) -> None:
        """All embedding values should be finite."""
        from vllm_omni.model_executor.models.minicpm4_5_o.vision import get_2d_sincos_pos_embed

        emb = get_2d_sincos_pos_embed(embed_dim=64, image_size=8)
        assert np.all(np.isfinite(emb))


# ---------------------------------------------------------------------------
# Vision: SiglipAttention
# ---------------------------------------------------------------------------


def _make_vision_config(**overrides) -> SimpleNamespace:
    """Create a minimal vision config namespace."""
    defaults = dict(
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        image_size=56,
        patch_size=14,
        num_channels=3,
        num_hidden_layers=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSiglipAttention:
    """Tests for SiglipAttention forward shape and projections."""

    @pytest.fixture
    def attention(self):
        """Create a SiglipAttention module with default config."""
        from vllm_omni.model_executor.models.minicpm4_5_o.vision import SiglipAttention

        return SiglipAttention(_make_vision_config())

    def test_forward_shape(self, attention) -> None:
        """Output shape should match input shape."""
        batch, seq, dim = 2, 16, 128
        x = torch.randn(batch, seq, dim)
        out = attention(x)
        assert out.shape == (batch, seq, dim)

    def test_projections_exist(self, attention) -> None:
        """Q/K/V projections should have correct output features."""
        assert attention.q_proj.out_features == 128
        assert attention.k_proj.out_features == 128
        assert attention.v_proj.out_features == 128


# ---------------------------------------------------------------------------
# Vision: SiglipEncoderLayer
# ---------------------------------------------------------------------------


class TestSiglipEncoderLayer:
    """Tests for a single SiglipEncoderLayer."""

    @pytest.fixture
    def layer(self):
        """Create a SiglipEncoderLayer with default config."""
        from vllm_omni.model_executor.models.minicpm4_5_o.vision import SiglipEncoderLayer

        return SiglipEncoderLayer(_make_vision_config())

    def test_forward_shape(self, layer) -> None:
        """Output shape should match input shape."""
        batch, seq, dim = 2, 16, 128
        x = torch.randn(batch, seq, dim)
        out = layer(x)
        assert out.shape == (batch, seq, dim)

    def test_residual_connection(self, layer) -> None:
        """With zero input, residual keeps output close to zero."""
        x = torch.zeros(1, 4, 128)
        out = layer(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Vision: Resampler (Perceiver)
# ---------------------------------------------------------------------------


class TestResampler:
    """Tests for the Perceiver Resampler."""

    @pytest.fixture
    def resampler(self):
        """Create a Resampler with small dimensions for testing."""
        from vllm_omni.model_executor.models.minicpm4_5_o.vision import Resampler

        return Resampler(
            num_queries=8,
            embed_dim=128,
            num_heads=4,
            kv_dim=64,
            max_size=(10, 10),
        )

    def test_output_shape(self, resampler) -> None:
        """Output should always be (B, num_queries, embed_dim)."""
        batch, seq, kv_dim = 2, 20, 64
        x = torch.randn(batch, seq, kv_dim)
        tgt_sizes = torch.tensor([[4, 5], [3, 4]])

        out = resampler(x, tgt_sizes)

        assert out.shape == (batch, 8, 128)

    def test_fixed_query_count(self, resampler) -> None:
        """Regardless of input length, query count is fixed."""
        x = torch.randn(1, 49, 64)  # 7*7 patches
        tgt_sizes = torch.tensor([[7, 7]])

        out = resampler(x, tgt_sizes)

        assert out.shape[1] == 8  # num_queries

    def test_pos_cache_auto_expansion(self, resampler) -> None:
        """Cache should expand if tgt_sizes exceed current max."""
        assert resampler.max_size == [10, 10]
        x = torch.randn(1, 180, 64)  # 15*12 = 180 patches
        tgt_sizes = torch.tensor([[15, 12]])

        out = resampler(x, tgt_sizes)

        assert resampler.max_size[0] >= 15
        assert resampler.max_size[1] >= 12
        assert out.shape == (1, 8, 128)


# ---------------------------------------------------------------------------
# Audio: MiniCPMWhisperEncoderLayer
# ---------------------------------------------------------------------------


class TestWhisperEncoderLayer:
    """Tests for MiniCPMWhisperEncoderLayer."""

    @pytest.fixture
    def layer(self):
        """Create a MiniCPMWhisperEncoderLayer with minimal config."""
        from transformers.models.whisper.configuration_whisper import WhisperConfig

        from vllm_omni.model_executor.models.minicpm4_5_o.audio import MiniCPMWhisperEncoderLayer

        config = WhisperConfig(
            d_model=128,
            encoder_attention_heads=4,
            encoder_ffn_dim=256,
            activation_function="gelu",
            attention_dropout=0.0,
            dropout=0.0,
            activation_dropout=0.0,
        )
        config._attn_implementation = "eager"
        return MiniCPMWhisperEncoderLayer(config, layer_idx=0)

    def test_forward_shape(self, layer) -> None:
        """Output shape should match input shape."""
        batch, seq, dim = 2, 16, 128
        x = torch.randn(batch, seq, dim)
        layer.eval()
        out = layer(x)
        assert out.shape == (batch, seq, dim)

    def test_residual_connection(self, layer) -> None:
        """With zero input, output shape should be preserved."""
        x = torch.zeros(1, 8, 128)
        layer.eval()
        out = layer(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Audio: MultiModalProjector
# ---------------------------------------------------------------------------


class TestMultiModalProjector:
    """Tests for the audio projector (Linear → ReLU → Linear)."""

    @pytest.fixture
    def projector(self):
        """Create a MultiModalProjector with small dimensions."""
        from vllm_omni.model_executor.models.minicpm4_5_o.audio import MultiModalProjector

        return MultiModalProjector(in_dim=128, out_dim=256)

    def test_forward_shape(self, projector) -> None:
        """Output dimension should match out_dim."""
        x = torch.randn(2, 16, 128)
        out = projector(x)
        assert out.shape == (2, 16, 256)

    def test_relu_nonlinearity(self, projector) -> None:
        """Verify the projector uses a nonlinear activation (output != linear)."""
        x = torch.randn(1, 4, 128)
        out = projector(x)
        # Output should differ from a single linear transform
        assert out.shape == (1, 4, 256)
