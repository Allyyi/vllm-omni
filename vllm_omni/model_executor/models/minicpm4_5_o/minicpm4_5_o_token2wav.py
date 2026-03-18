# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenBMB Team.
"""MiniCPM-O 4.5 Token2Wav stage: converts audio codec tokens to waveform.

Uses the StepAudio2 Token2wav vocoder to synthesize audio from
discrete audio codes produced by the TTS stage.

Supports two modes:
  - **Batch mode**: all tokens arrive in a single forward() call.
  - **Streaming mode** (async_chunk / duplex): tokens arrive in chunks
    across multiple forward() calls.  Uses the ``stream()`` API with
    internal flow/hift caches for cross-chunk continuity.
"""

import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)

# Vocoder constants (matching original MiniCPMODuplex)
_CHUNK_SIZE = 25  # tokens per vocoder chunk (~1 s at 24 kHz)
_SILENCE_PREFIX = [4218] * 3  # 3 silence tokens for stream init


class MiniCPMOToken2WavForConditionalGeneration(nn.Module):
    """
    Token2Wav stage of MiniCPM-O 4.5.

    Receives audio token IDs from the TTS stage and converts them to
    a waveform tensor using the Token2wav vocoder (stepaudio2).

    This stage has no learnable parameters loaded from safetensors —
    the vocoder weights are loaded separately from the model assets directory.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.audio_tokenizer = None
        self._model_path = vllm_config.model_config.model

        # Placeholder to satisfy vllm pipeline infrastructure
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

        # Streaming state (persists across forward() calls for same request)
        self._stream_initialized = False
        self._stream_buffer: list[int] = []
        self._pre_lookahead: int = 3  # default, updated from vocoder
        self._flow_cache_base = None
        self._hift_cache_base = None

    def _ensure_asset_dir(self, asset_subpath: str) -> str:
        """Ensure asset directory exists, downloading from HF if needed."""
        model_dir = os.path.join(self._model_path, asset_subpath)
        if not os.path.exists(model_dir):
            try:
                from huggingface_hub import snapshot_download

                repo_dir = snapshot_download(
                    repo_id=self._model_path,
                    allow_patterns=[f"{asset_subpath}/**"],
                )
                model_dir = os.path.join(repo_dir, asset_subpath)
            except Exception as e:
                logger.warning(f"Failed to download asset {asset_subpath}: {e}")
                return model_dir

        return model_dir

    def _init_vocoder(self):
        """Initialize the Token2wav vocoder lazily on first use."""
        if self.audio_tokenizer is not None:
            return

        try:
            from stepaudio2 import Token2wav
        except ImportError:
            raise ImportError("Token2wav requires stepaudio2. Install via: pip install minicpmo-utils[all]")

        model_dir = self._ensure_asset_dir("assets/token2wav")
        self.audio_tokenizer = Token2wav(model_dir, float16=True, n_timesteps=10)
        logger.info(f"Token2wav vocoder initialized from {model_dir}")

    def _init_stream_caches(self):
        """Initialize streaming caches for cross-chunk vocoder state."""
        if self._stream_initialized:
            return

        from vllm_omni.model_executor.models.minicpm4_5_o.utils import (
            torch_clone_recursive,
        )

        self.audio_tokenizer.cache = None

        if hasattr(self.audio_tokenizer, "set_stream_cache"):
            flow_cache, hift_cache = self.audio_tokenizer.set_stream_cache(None)
            self._flow_cache_base = torch_clone_recursive(flow_cache)
            self._hift_cache_base = torch_clone_recursive(hift_cache)

        if hasattr(self.audio_tokenizer, "flow") and hasattr(self.audio_tokenizer.flow, "pre_lookahead_len"):
            self._pre_lookahead = int(self.audio_tokenizer.flow.pre_lookahead_len)

        self._stream_buffer = list(_SILENCE_PREFIX)
        self._stream_initialized = True

    def _reset_stream_state(self):
        """Reset streaming state for a new request/turn."""
        from vllm_omni.model_executor.models.minicpm4_5_o.utils import (
            torch_clone_recursive,
        )

        if self._flow_cache_base is not None:
            self.audio_tokenizer.stream_cache = torch_clone_recursive(self._flow_cache_base)
        if self._hift_cache_base is not None:
            self.audio_tokenizer.hift_cache_dict = torch_clone_recursive(self._hift_cache_base)
        self._stream_buffer = list(_SILENCE_PREFIX)

    def _extract_token_list(self, audio_token_ids: torch.Tensor) -> list[int]:
        """Convert audio_token_ids tensor to a flat list, filtering EOS."""
        if audio_token_ids.dim() == 3:
            token_list = audio_token_ids.squeeze(0)[:, 0].tolist()
        elif audio_token_ids.dim() == 2:
            token_list = audio_token_ids.squeeze(0).tolist()
        else:
            token_list = audio_token_ids.tolist()

        # Filter out EOS tokens
        tts_config = getattr(self.config, "tts_config", None)
        if tts_config is not None:
            num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            eos_id = num_audio_tokens - 1
            token_list = [t for t in token_list if t != eos_id]

        return token_list

    def _generate_batch(self, token_list: list[int]) -> torch.Tensor:
        """Batch mode: convert all tokens to waveform in one shot."""
        try:
            wav_bytes = self.audio_tokenizer(token_list, None)

            import io

            import soundfile as sf

            waveform, _ = sf.read(io.BytesIO(wav_bytes))
            return torch.tensor(waveform, dtype=torch.float32)
        except Exception as e:
            logger.error(f"Token2Wav batch generation failed: {e}")
            return torch.zeros(1)

    def _generate_stream(self, token_list: list[int], is_last_chunk: bool) -> torch.Tensor:
        """Streaming mode: buffer tokens and emit via stream() API."""
        self._stream_buffer.extend(token_list)

        pcm_bytes_list: list[bytes] = []

        # Process buffered tokens in CHUNK_SIZE windows with lookahead
        while len(self._stream_buffer) >= _CHUNK_SIZE + self._pre_lookahead:
            pcm_bytes = self.audio_tokenizer.stream(
                self._stream_buffer[: _CHUNK_SIZE + self._pre_lookahead],
                prompt_wav=None,
            )
            pcm_bytes_list.append(pcm_bytes)
            self._stream_buffer = self._stream_buffer[_CHUNK_SIZE:]

        # Flush remaining tokens on last chunk
        if is_last_chunk and len(self._stream_buffer) > 0:
            pcm_bytes = self.audio_tokenizer.stream(
                self._stream_buffer,
                prompt_wav=None,
                last_chunk=True,
            )
            pcm_bytes_list.append(pcm_bytes)
            self._stream_buffer = []

        if not pcm_bytes_list:
            return torch.zeros(1)

        # Merge PCM and convert to float32 waveform
        all_pcm = b"".join(pcm_bytes_list)
        if len(all_pcm) == 0:
            return torch.zeros(1)

        pcm_np = np.frombuffer(all_pcm, dtype="<i2")
        audio_waveform = pcm_np.astype(np.float32) / 32768.0
        return torch.tensor(audio_waveform, dtype=torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """
        Convert audio token IDs to waveform.

        Args:
            additional_information: dict containing:
                - "audio_token_ids": Tensor of audio codec token IDs from TTS stage
                - "is_streaming" (optional): bool tensor, True for async_chunk mode
                - "is_last_chunk" (optional): bool tensor, True on final chunk
        """
        self._init_vocoder()

        if additional_information is None:
            logger.warning("Token2Wav received no additional_information")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": torch.zeros(1)},
            )

        audio_token_ids = additional_information.get("audio_token_ids")
        if audio_token_ids is None:
            logger.warning("Token2Wav received no audio_token_ids")
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": torch.zeros(1)},
            )

        token_list = self._extract_token_list(audio_token_ids)
        if len(token_list) == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": torch.zeros(1)},
            )

        # Check streaming mode
        is_streaming_tensor = additional_information.get("is_streaming")
        is_streaming = (
            is_streaming_tensor.item()
            if isinstance(is_streaming_tensor, torch.Tensor)
            else bool(is_streaming_tensor)
            if is_streaming_tensor is not None
            else False
        )

        if is_streaming and hasattr(self.audio_tokenizer, "stream"):
            is_last_tensor = additional_information.get("is_last_chunk")
            is_last_chunk = (
                is_last_tensor.item()
                if isinstance(is_last_tensor, torch.Tensor)
                else bool(is_last_tensor)
                if is_last_tensor is not None
                else False
            )

            # Initialize or reset streaming caches on first chunk
            if not self._stream_initialized:
                self._init_stream_caches()
            # If this is the first chunk of a new turn (buffer was flushed),
            # reset vocoder caches for clean state
            if len(self._stream_buffer) == 0 and not is_last_chunk:
                self._reset_stream_state()

            audio_tensor = self._generate_stream(token_list, is_last_chunk)

            # Reset for next request after final chunk
            if is_last_chunk:
                self._stream_buffer = []
        else:
            # Batch mode: all tokens in one shot
            audio_tensor = self._generate_batch(token_list)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audio_tensor},
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Token2Wav has no safetensors weights; vocoder loads from assets dir."""
        loaded = set()
        for name, _ in weights:
            loaded.add(name)
        return loaded
