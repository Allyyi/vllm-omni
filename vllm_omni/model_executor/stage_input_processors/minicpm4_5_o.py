# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenBMB Team.
"""Stage input processors for MiniCPM-O-4_5: Thinker → TTS → Token2Wav."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.inputs.data import OmniTokensPrompt


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def llm2tts(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create TTS inputs.

    Data flow:
        Stage 0 output.multimodal_output contains:
            - "llm_hidden_states": Tensor[response_len, 4096] — last-layer hidden states
            - "llm_token_ids": Tensor[response_len] — generated token IDs

        These are packaged into OmniTokensPrompt.additional_information for Stage 1.

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for TTS stage
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    tts_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        prompt_len = len(prompt_token_ids)

        # Extract hidden states from pooler_output via multimodal_output.
        # The AR model runner captures hidden_states as {"hidden": ...} in pooler_output,
        # which gets remapped to the engine_output_type key ("latent") by the
        # MultimodalOutputProcessor. So we read "latent" here.
        latent = output.multimodal_output.get("latent")

        if latent is not None:
            llm_hidden_states = latent.clone().detach().to(device=device, dtype=torch.float32)
            # Only use the response portion (after prompt) for TTS conditioning
            llm_hidden_states = llm_hidden_states[prompt_len:]
        else:
            # Fallback: create zero hidden states
            llm_hidden_states = torch.zeros(len(output.token_ids), 4096, device=device, dtype=torch.float32)

        # Token IDs come from the sampled output, not from multimodal_output
        llm_token_ids = torch.tensor(output.token_ids, dtype=torch.long, device=device)

        info = {
            "llm_hidden_states": llm_hidden_states,
            "llm_token_ids": llm_token_ids,
        }

        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # Dummy token — TTS uses embeddings
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return tts_inputs


def tts2token2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process TTS outputs to create Token2Wav inputs.

    Data flow:
        Stage 1 output.multimodal_output contains:
            - "audio_token_ids": Tensor[num_tokens] — generated audio code IDs

        These are packaged into OmniTokensPrompt.additional_information for Stage 2.

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for TTS)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for Token2Wav stage
    """
    tts_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    token2wav_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    for tts_output in tts_outputs:
        output = tts_output.outputs[0]

        # Extract audio token IDs from TTS output
        audio_token_ids = output.multimodal_output.get("audio_token_ids")

        if audio_token_ids is None:
            # Fallback: use output token_ids directly (TTS generates audio codes)
            audio_token_ids = torch.tensor(output.token_ids, dtype=torch.long, device=device)

        info = {
            "audio_token_ids": (
                audio_token_ids.detach().to(device=device)
                if isinstance(audio_token_ids, torch.Tensor)
                else torch.tensor(audio_token_ids, dtype=torch.long, device=device)
            ),
        }

        # Pass through streaming flags from TTS async_chunk mode
        is_streaming = output.multimodal_output.get("is_streaming")
        if is_streaming is not None:
            info["is_streaming"] = is_streaming
            info["is_last_chunk"] = output.multimodal_output.get("is_last_chunk", torch.tensor(False, dtype=torch.bool))

        token2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],  # Dummy token — Token2Wav uses audio codes
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return token2wav_inputs


def llm2tts_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """
    Async chunk version: stream thinker hidden states to TTS per decode step.

    Called by OmniChunkTransferAdapter.save_async on every AR scheduler step.

    Data flow:
        chunk_id == 0 (prefill + first decode):
            - Accumulate prefill hidden states across sub-chunks via request_payload
            - Extract prompt_token_ids for splitting prompt vs response
        chunk_id > 0 (subsequent decode steps):
            - Send incremental decode hidden state + token ID

    The TTS stage receives these chunks via the connector and accumulates
    them to build input embeddings for generate_chunk().
    """
    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]

    llm_hidden_states = pooling_output.get("hidden")
    if llm_hidden_states is None and not is_finished:
        return None

    if chunk_id == 0:
        # First chunk: includes prefill hidden states
        # May span multiple sub-steps during chunked prefill
        prompt_token_ids = list(request.prompt_token_ids)
        all_token_ids = list(request.all_token_ids)

        payload = {
            "llm_hidden_states": (
                llm_hidden_states.detach().cpu() if llm_hidden_states is not None else torch.zeros(1, 4096)
            ),
            "prompt_token_ids": prompt_token_ids,
            "all_token_ids": all_token_ids,
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }

        # Accumulate prefill across multiple sub-chunks if not finished
        saved = transfer_manager.request_payload.get(request_id)
        if saved is not None:
            # Concatenate with previously saved prefill hidden states
            payload["llm_hidden_states"] = torch.cat(
                (saved["llm_hidden_states"], payload["llm_hidden_states"]),
                dim=0,
            )
        if not is_finished and llm_hidden_states is not None:
            # Check if we have enough hidden states (prefill may not be complete)
            num_hidden = payload["llm_hidden_states"].shape[0]
            num_all_tokens = len(all_token_ids)
            if num_hidden < num_all_tokens:
                # Prefill not complete, save and wait for more
                transfer_manager.request_payload[request_id] = payload
                return None

        # Prefill complete or finished — send the payload
        transfer_manager.request_payload.pop(request_id, None)
        return payload

    else:
        # Subsequent decode steps: send incremental hidden state + token ID
        output_token_ids = list(request.output_token_ids)

        payload = {
            "llm_hidden_states": (
                llm_hidden_states.detach().cpu() if llm_hidden_states is not None else torch.zeros(1, 4096)
            ),
            "llm_token_ids": output_token_ids,
            "finished": torch.tensor(is_finished, dtype=torch.bool),
            # Tell the receiving side to replace (not concat) these keys
            "override_keys": ["llm_hidden_states", "llm_token_ids"],
        }
        return payload
