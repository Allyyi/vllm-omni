# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenBMB Team.
"""MiniCPM-O 4.5 Thinker stage: Qwen3 LLM + SigLip vision + Whisper audio encoder.

This module handles multimodal understanding (images + audio + text) and produces
text hidden states that are passed to downstream TTS and Token2Wav stages.
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.minicpm4_5_o.audio import MiniCPMWhisperEncoder
from vllm_omni.model_executor.models.minicpm4_5_o.modeling_navit_siglip import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
)
from vllm_omni.model_executor.models.minicpm4_5_o.vision import (
    MultiModalProjector,
    Resampler,
)

logger = logging.getLogger(__name__)


class MiniCPMOThinkerForConditionalGeneration(nn.Module):
    """
    Thinker stage of MiniCPM-O 4.5.

    Architecture:
        - Vision: SiglipVisionTransformer → Resampler → (B, 64, 4096)
        - Audio: MiniCPMWhisperEncoder → projection → avg pooling → (B, T, 4096)
        - LLM: Qwen3ForCausalLM (36 layers, 4096 hidden)

    The thinker processes multimodal inputs, runs the LLM, and returns
    hidden states along with generated token IDs for the TTS stage.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "llm.lm_head.": "llm.lm_head.",
            "llm.model.": "llm.model.",
            "llm.": "llm.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config

        # --- Language Model (Qwen3) ---
        self.llm = Qwen3ForCausalLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "llm"),
        )
        self.embed_dim = config.hidden_size

        # --- Vision Module (SigLip + Resampler) ---
        if getattr(config, "init_vision", True):
            vision_config = config.vision_config
            if isinstance(vision_config, dict):
                vision_config = SiglipVisionConfig(**vision_config)

            if getattr(config, "_attn_implementation", "eager") == "flash_attention_2":
                vision_config._attn_implementation = "flash_attention_2"
            else:
                vision_config._attn_implementation = "eager"

            self.vpm = SiglipVisionTransformer(vision_config)
            if getattr(config, "drop_vision_last_layer", False):
                self.vpm.encoder.layers = self.vpm.encoder.layers[:-1]

            setattr(self.vpm, "embed_dim", self.vpm.embeddings.embed_dim)
            setattr(self.vpm, "patch_size", self.vpm.embeddings.patch_size)
            self.vision_dim = self.vpm.embed_dim

            self.resampler = Resampler(
                num_queries=config.query_num,
                embed_dim=self.embed_dim,
                num_heads=self.embed_dim // 128,
                kv_dim=self.vision_dim,
                adaptive=True,
            )
        else:
            self.vpm = None
            self.resampler = None

        # --- Audio Module (Whisper + Projection + Pooling) ---
        if getattr(config, "init_audio", True):
            from transformers import WhisperConfig

            audio_config = config.audio_config
            if isinstance(audio_config, dict):
                audio_config = WhisperConfig(**audio_config)

            if getattr(config, "_attn_implementation", "eager") == "eager":
                audio_config._attn_implementation = "eager"
            else:
                audio_config._attn_implementation = "sdpa"

            self.apm = MiniCPMWhisperEncoder(audio_config)
            audio_output_dim = int(audio_config.encoder_ffn_dim // 4)
            self.audio_pool_step = getattr(config, "audio_pool_step", 5)
            self.audio_avg_pooler = nn.AvgPool1d(self.audio_pool_step, stride=self.audio_pool_step)
            self.audio_projection_layer = MultiModalProjector(in_dim=audio_output_dim, out_dim=self.embed_dim)
            self.audio_encoder_layer = -1
            self.audio_chunk_length = getattr(config, "audio_chunk_length", 1.0)
        else:
            self.apm = None
            self.audio_projection_layer = None
            self.audio_avg_pooler = None

        self.make_empty_intermediate_tensors = self.llm.make_empty_intermediate_tensors

    # ------------------------------------------------------------------
    # Vision embedding
    # ------------------------------------------------------------------

    def get_vision_embedding(self, pixel_values_list, tgt_sizes, image_bound):
        """
        Process images through SigLip + Resampler.

        Args:
            pixel_values_list: list of tensors, each (num_slices, C, H, W)
            tgt_sizes: list of tensors, each (num_slices, 2) — (h_patches, w_patches)
            image_bound: list of (start, end) index pairs per batch item

        Returns:
            List of vision hidden states per batch item.
        """
        if self.vpm is None:
            return []

        dtype = self.llm.model.embed_tokens.weight.dtype
        device = self.llm.model.embed_tokens.weight.device

        vision_hidden_states = []
        all_pixel_values = []
        img_cnt = []

        for pixel_values in pixel_values_list:
            img_cnt.append(len(pixel_values))
            all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

        if all_pixel_values:
            tgt_sizes_filtered = [ts for ts in tgt_sizes if isinstance(ts, torch.Tensor)]
            tgt_sizes_t = torch.vstack(tgt_sizes_filtered).type(torch.int32)

            max_patches = torch.max(tgt_sizes_t[:, 0] * tgt_sizes_t[:, 1])

            all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
            B, L, _ = all_pixel_values.shape
            all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
            for i in range(B):
                patch_attn_mask[i, 0, : tgt_sizes_t[i][0] * tgt_sizes_t[i][1]] = True

            vision_batch_size = getattr(self.config, "vision_batch_size", 16)
            all_pixel_values = all_pixel_values.type(dtype)

            if B > vision_batch_size:
                hs = []
                for i in range(0, B, vision_batch_size):
                    tmp_hs = self.vpm(
                        all_pixel_values[i : i + vision_batch_size],
                        patch_attention_mask=patch_attn_mask[i : i + vision_batch_size],
                        tgt_sizes=tgt_sizes_t[i : i + vision_batch_size],
                    ).last_hidden_state
                    hs.append(tmp_hs)
                vision_embedding = torch.cat(hs, dim=0)
            else:
                vision_embedding = self.vpm(
                    all_pixel_values,
                    patch_attention_mask=patch_attn_mask,
                    tgt_sizes=tgt_sizes_t,
                ).last_hidden_state

            vision_embedding = self.resampler(vision_embedding, tgt_sizes_t)

            start = 0
            for pv in pixel_values_list:
                cnt = len(pv)
                if cnt > 0:
                    vision_hidden_states.append(vision_embedding[start : start + cnt])
                    start += cnt
                else:
                    vision_hidden_states.append([])
        else:
            for _ in pixel_values_list:
                vision_hidden_states.append([])

        return vision_hidden_states

    # ------------------------------------------------------------------
    # Audio embedding
    # ------------------------------------------------------------------

    def subsequent_chunk_mask(
        self,
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        ret = torch.zeros(size, size, device=device, dtype=torch.bool)
        row_indices = torch.arange(size, device=device)
        chunk_indices = row_indices // chunk_size
        if num_left_chunks < 0:
            start_indices = torch.zeros_like(row_indices)
        else:
            start_chunk_indices = torch.clamp(chunk_indices - num_left_chunks, min=0)
            start_indices = start_chunk_indices * chunk_size
        end_chunk_indices = chunk_indices + 1
        end_indices = torch.clamp(end_chunk_indices * chunk_size, max=size)
        col_indices = torch.arange(size, device=device).unsqueeze(0)
        start_indices = start_indices.unsqueeze(1)
        end_indices = end_indices.unsqueeze(1)
        ret = (col_indices >= start_indices) & (col_indices < end_indices)
        return ret

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
        input_lengths_after_pooling = (input_lengths_after_cnn - self.audio_pool_step) // self.audio_pool_step + 1
        input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)
        return input_lengths_after_cnn, input_lengths_after_pooling

    def get_audio_embedding(self, wavforms, audio_feature_lens_raw, chunk_length=-1):
        """
        Process audio through Whisper encoder → projection → average pooling.

        Args:
            wavforms: (batch, 80, frames) mel spectrograms
            audio_feature_lens_raw: list of tensors with per-chunk lengths
            chunk_length: whisper chunk attention length (-1 = full attention)

        Returns:
            List of lists of audio embedding tensors per batch item.
        """
        if self.apm is None or len(wavforms) == 0:
            return []

        audio_feature_lens = torch.hstack(audio_feature_lens_raw)
        batch_size, _, max_mel_seq_len = wavforms.shape
        max_seq_len = (max_mel_seq_len - 1) // 2 + 1

        # Build attention mask
        seq_range = (
            torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device)
            .unsqueeze(0)
            .expand(batch_size, max_seq_len)
        )
        lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
        padding_mask = seq_range >= lengths_expand

        audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
            batch_size, 1, max_seq_len, max_seq_len
        )
        audio_attention_mask = audio_attention_mask_.to(
            dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
        )

        if chunk_length > 0:
            chunk_num_frame = int(chunk_length * 50)
            chunk_mask = self.subsequent_chunk_mask(
                size=max_seq_len,
                chunk_size=chunk_num_frame,
                num_left_chunks=-1,
                device=audio_attention_mask_.device,
            )
            audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

        audio_attention_mask[audio_attention_mask_] = float("-inf")

        audio_states = self.apm(
            wavforms,
            output_hidden_states=True,
            attention_mask=audio_attention_mask,
        ).hidden_states[self.audio_encoder_layer]

        audio_embeds = self.audio_projection_layer(audio_states)
        audio_embeds = audio_embeds.transpose(1, 2)
        audio_embeds = self.audio_avg_pooler(audio_embeds)
        audio_embeds = audio_embeds.transpose(1, 2)

        _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)
        num_audio_tokens = feature_lens_after_pooling

        final_audio_embeds = []
        idx = 0
        for i in range(len(audio_feature_lens_raw)):
            target_audio_embeds = []
            for _ in range(len(audio_feature_lens_raw[i])):
                target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                idx += 1
            final_audio_embeds.append(target_audio_embeds)

        return final_audio_embeds

    # ------------------------------------------------------------------
    # Embedding assembly
    # ------------------------------------------------------------------

    def get_vllm_embedding(self, data: dict) -> tuple[torch.Tensor, list]:
        """Get text embeddings with vision embeddings scattered in."""
        pixel_values_list = data.get("pixel_values", [[]])
        tgt_sizes = data.get("tgt_sizes", [[]])
        image_bound = data.get("image_bound", [[]])

        vision_hidden_states = data.get("vision_hidden_states", None)
        if vision_hidden_states is None:
            vision_hidden_states = self.get_vision_embedding(pixel_values_list, tgt_sizes, image_bound)

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    cur_vllm_emb.scatter_(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    )

        return vllm_embedding, vision_hidden_states

    def get_omni_embedding(
        self,
        data: dict,
        input_embeddings: torch.Tensor,
        chunk_length: float = -1,
    ) -> torch.Tensor:
        """Scatter audio embeddings into the text+vision embedding tensor."""
        wavforms = data.get("audio_features", [])
        audio_feature_lens_raw = data.get("audio_feature_lens", [])

        if len(wavforms) > 0:
            audio_embeddings = self.get_audio_embedding(wavforms, audio_feature_lens_raw, chunk_length)

            if len(audio_embeddings) > 0:
                audio_bounds = data.get("audio_bounds", [])
                stream_input = getattr(self.config, "stream_input", False)

                bs = len(input_embeddings)
                if stream_input:
                    assert bs == 1
                    for i in range(bs):
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device,
                            dtype=input_embeddings.dtype,
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(
                                input_embeddings.device
                            )
                            if embs.shape[0] != len(audio_indices):
                                raise ValueError(
                                    f"Shape mismatch: embeddings {embs.shape} vs indices length {len(audio_indices)}"
                                )
                            input_embeddings[i, audio_indices] = embs.to(input_embeddings.dtype)

        return input_embeddings

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.llm.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.llm.compute_logits(hidden_states)

    def get_language_model(self) -> nn.Module:
        return self.llm

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        skip_prefixes = ["tts."]
        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
