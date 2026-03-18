# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenBMB Team.
"""MiniCPM-O 4.5 unified model: routes to thinker / TTS / token2wav stages.

This is the top-level model class registered in the vLLM-Omni registry.
It dispatches to the correct stage implementation based on
``vllm_config.model_config.model_stage``.
"""

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)


class MiniCPMOForConditionalGeneration(nn.Module):
    """
    Unified MiniCPM-O 4.5 model that routes to the appropriate stage.

    Stages:
        - ``thinker``: Multimodal understanding (SigLip + Whisper + Qwen3 LLM)
        - ``tts``: Text-to-speech (Llama-backbone AR audio code generator)
        - ``token2wav``: Vocoder (StepAudio2 Token2wav → waveform)
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.has_preprocess = False
        self.have_multimodal_outputs = True
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config

        self.model_stage = vllm_config.model_config.model_stage

        if self.model_stage == "thinker":
            self.thinker = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "thinker"),
                hf_config=config,
                architectures=["MiniCPMOThinkerForConditionalGeneration"],
            )
            self.model = self.thinker
            self.tts = None
            self.token2wav = None

        elif self.model_stage == "tts":
            # TTS is a generation stage — no preprocess needed.
            # additional_information comes via runtime_additional_information in kwargs.
            self.thinker = None
            self.tts = self._init_tts_model(config)
            self.model = self.tts
            self.token2wav = None

            # State for async_chunk (duplex) mode: incremental generate_chunk
            self._tts_past_key_values = None
            self._tts_text_start_pos = 0

        elif self.model_stage == "token2wav":
            self.thinker = None
            self.tts = None
            self.token2wav = init_vllm_registered_model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "token2wav"),
                hf_config=config,
                architectures=["MiniCPMOToken2WavForConditionalGeneration"],
            )
            self.model = self.token2wav

        else:
            raise ValueError(f"Invalid model_stage '{self.model_stage}'. Expected one of: thinker, tts, token2wav")

        self.make_empty_intermediate_tensors = (
            self.thinker.make_empty_intermediate_tensors if self.model_stage == "thinker" else (lambda *a, **kw: None)
        )

    # ------------------------------------------------------------------
    # TTS model initialization
    # ------------------------------------------------------------------

    def _init_tts_model(self, config):
        """Initialize the TTS model (MiniCPMTTS) from the tts_config."""
        from vllm_omni.model_executor.models.minicpm4_5_o.minicpm4_5_o_tts import (
            MiniCPMTTS,
        )

        tts_config = config.tts_config
        if isinstance(tts_config, dict):
            from vllm_omni.model_executor.models.minicpm4_5_o.minicpm4_5_o_tts import (
                MiniCPMTTSConfig,
            )

            tts_config = MiniCPMTTSConfig(**tts_config)

        tts_model = MiniCPMTTS(config=tts_config, audio_tokenizer=None)
        return tts_model

    # ------------------------------------------------------------------
    # TTS preprocessing
    # ------------------------------------------------------------------

    def _build_tts_embeddings(self, req_info: dict[str, Any]) -> torch.Tensor:
        """
        Build TTS input embeddings from LLM hidden states and token IDs.

        The hidden_text_merge conditioning:
            tts_embeds = emb_text(llm_tokens) + normalize(projector_semantic(llm_hidden))
            inputs = [spk_embeds, tts_embeds, text_eos_embed, audio_bos_embed]

        Args:
            req_info: dict with "llm_hidden_states" and "llm_token_ids" from
                      the llm2tts stage input processor.

        Returns:
            inputs_embeds: (1, seq_len, hidden_size)
        """
        llm_hidden_states = req_info.get("llm_hidden_states")
        llm_token_ids = req_info.get("llm_token_ids")

        if llm_hidden_states is None or llm_token_ids is None:
            raise ValueError("TTS stage requires llm_hidden_states and llm_token_ids")

        device = self.tts.emb_text.weight.device
        dtype = self.tts.emb_text.weight.dtype

        llm_hidden_states = llm_hidden_states.to(device=device, dtype=torch.float32)
        llm_token_ids = llm_token_ids.to(device=device, dtype=torch.long)

        # Build condition embeddings
        condition_type = getattr(self.tts, "condition_type", "hidden_text_merge")

        if condition_type == "hidden_text_merge":
            # Text embeddings from TTS text vocabulary
            llm_embeds = self.tts.emb_text(llm_token_ids)

            # Project LLM hidden states to TTS hidden size
            hidden_embeds = self.tts.projector_semantic(llm_hidden_states)

            if getattr(self.tts.config, "normalize_projected_hidden", False):
                hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)

            tts_embeds = llm_embeds + hidden_embeds
        else:
            raise NotImplementedError(f"Unsupported TTS condition_type: {condition_type}")

        # Speaker embeddings (empty for non-cloning)
        spk_embeds = torch.ones([0, self.tts.config.hidden_size], device=device, dtype=dtype)

        # Special token embeddings
        audio_bos_id = torch.tensor([self.tts.audio_bos_token_id], device=device, dtype=torch.long)
        audio_bos_embeds = self.tts.emb_text(audio_bos_id)

        text_eos_id = torch.tensor([self.tts.config.text_eos_token_id], device=device, dtype=torch.long)
        text_eos_embed = self.tts.emb_text(text_eos_id)

        # Assemble: [spk_embeds, tts_embeds, text_eos, audio_bos]
        inputs_embeds = torch.cat([spk_embeds, tts_embeds, text_eos_embed, audio_bos_embeds], dim=0).unsqueeze(0)

        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        # ----- Thinker stage -----
        if self.model_stage == "thinker":
            thinker_output = self.thinker.forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

            if isinstance(thinker_output, IntermediateTensors):
                return thinker_output

            text_hidden_states = thinker_output
            return OmniOutput(
                text_hidden_states=text_hidden_states.reshape(-1, text_hidden_states.shape[-1]),
                multimodal_outputs=None,
            )

        # ----- TTS stage -----
        if self.model_stage == "tts":
            # Extract per-request additional_information from generation runner.
            # runtime_additional_information is a list of dicts (one per request).
            runtime_info = kwargs.get("runtime_additional_information")
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                req_info = runtime_info[0]  # batch_size=1
            elif additional_information is not None:
                req_info = additional_information
            else:
                req_info = None

            # Check if this is an async_chunk (duplex) call with "finished" flag
            is_async_chunk = req_info is not None and "finished" in req_info
            is_finished = req_info.get("finished", torch.tensor(False)).item() if is_async_chunk else False

            if req_info is not None:
                tts_inputs_embeds = self._build_tts_embeddings(req_info)
            elif inputs_embeds is not None:
                tts_inputs_embeds = inputs_embeds
            else:
                raise ValueError("TTS stage requires runtime_additional_information or inputs_embeds")

            eos_token = torch.tensor(
                [self.tts.config.num_audio_tokens - 1],
                dtype=torch.long,
                device=tts_inputs_embeds.device,
            )

            if is_async_chunk:
                # Duplex mode: use generate_chunk with KV cache carryover
                tts_temperature = torch.tensor([0.8], dtype=torch.float, device=tts_inputs_embeds.device)
                max_token_per_chunk = 26  # ~1 sec audio (25 tokens/sec + 1)
                min_new_tokens = 0 if is_finished else 26

                audio_token_ids, self._tts_past_key_values = self.tts.generate_chunk(
                    inputs_embeds=tts_inputs_embeds,
                    temperature=tts_temperature,
                    repetition_penalty=1.05,
                    eos_token=eos_token,
                    force_no_stop=False,
                    max_new_token=max_token_per_chunk,
                    min_new_tokens=min_new_tokens,
                    past_key_values=self._tts_past_key_values,
                    text_start_pos=self._tts_text_start_pos,
                )

                # Update position tracking
                self._tts_text_start_pos += tts_inputs_embeds.shape[1] + audio_token_ids.shape[1]

                # Reset state on turn end
                if is_finished:
                    self._tts_past_key_values = None
                    self._tts_text_start_pos = 0

            else:
                # Batch mode: full generation in one shot
                from vllm_omni.model_executor.models.minicpm4_5_o.utils import TTSSamplingParams

                outputs = self.tts.generate(
                    inputs_embeds=tts_inputs_embeds,
                    eos_token=eos_token,
                    sampling_params=TTSSamplingParams(),
                    show_tqdm=False,
                )
                audio_token_ids = outputs.new_ids  # (1, seq_len, num_vq)

            mm_outputs: dict[str, Any] = {"audio_token_ids": audio_token_ids}
            if is_async_chunk:
                mm_outputs["is_streaming"] = torch.tensor(True, dtype=torch.bool)
                mm_outputs["is_last_chunk"] = torch.tensor(is_finished, dtype=torch.bool)

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs=mm_outputs,
            )

        # ----- Token2Wav stage -----
        if self.model_stage == "token2wav":
            # Extract per-request additional_information from generation runner.
            runtime_info = kwargs.get("runtime_additional_information")
            if runtime_info and isinstance(runtime_info, list) and len(runtime_info) > 0:
                token2wav_info = runtime_info[0]
            elif additional_information is not None:
                token2wav_info = additional_information
            else:
                token2wav_info = None

            return self.token2wav.forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                additional_information=token2wav_info,
                **{k: v for k, v in kwargs.items() if k != "runtime_additional_information"},
            )

        raise ValueError(f"Unknown model_stage: {self.model_stage}")

    # ------------------------------------------------------------------
    # Embedding helpers (delegated to thinker)
    # ------------------------------------------------------------------

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "thinker":
            return self.thinker.llm.model.embed_tokens(input_ids)

        # Non-thinker stages: return dummy embeddings
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros(
            input_ids.shape[0],
            hidden_size,
            device=input_ids.device,
            dtype=torch.bfloat16,
        )

    def get_language_model(self) -> nn.Module:
        if self.model_stage == "thinker":
            return self.thinker.get_language_model()
        return None

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, **kwargs) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states

        if self.model_stage == "thinker" and hidden_states is not None:
            return self.thinker.compute_logits(hidden_states)

        return None

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights dispatched by prefix.

        Weight names in the checkpoint:
            - llm.*, vpm.*, resampler.*, apm.*, audio_* → thinker
            - tts.* → TTS stage
            - token2wav assets loaded separately
        """
        loaded_weights: set[str] = set()

        thinker_weights: list[tuple[str, torch.Tensor]] = []
        tts_weights: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            if name.startswith("tts."):
                tts_weights.append((name, tensor))
            else:
                # Everything else (llm.*, vpm.*, resampler.*, apm.*, audio_*)
                # belongs to the thinker
                thinker_weights.append((name, tensor))

        if self.thinker is not None and thinker_weights:
            thinker_loaded = self.thinker.load_weights(thinker_weights)
            loaded_weights.update(thinker_loaded)

        if self.tts is not None and tts_weights:
            tts_loaded = self._load_tts_weights(tts_weights)
            loaded_weights.update(tts_loaded)

        if self.token2wav is not None:
            # Token2Wav has no safetensors weights
            pass

        return loaded_weights

    def _load_tts_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Load TTS weights, stripping the 'tts.' prefix."""
        loaded = set()
        state_dict = {}
        for name, tensor in weights:
            # Strip 'tts.' prefix for the TTS model
            if name.startswith("tts."):
                tts_name = name[4:]  # remove 'tts.'
            else:
                tts_name = name
            state_dict[tts_name] = tensor
            loaded.add(name)

        # Load into TTS model
        missing, unexpected = self.tts.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"TTS missing keys: {missing[:10]}...")
        if unexpected:
            logger.warning(f"TTS unexpected keys: {unexpected[:10]}...")

        return loaded
