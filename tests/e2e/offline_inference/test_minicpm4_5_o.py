# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline inference tests for MiniCPM-O 4.5 omni-modal model.

Verifies the 3-stage pipeline (sync and async_chunk modes):
  Stage 0  Thinker  (Qwen3-8B + SigLip2 + Whisper) → text + hidden states
  Stage 1  TTS      (LlamaModel decoder)            → audio token IDs
  Stage 2  Token2Wav(CosyVoice2 flow + HiFi-GAN)    → 24 kHz waveform
"""

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    modify_stage_config,
)
from tests.utils import hardware_test

models = ["openbmb/MiniCPM-o-4_5"]


def get_default_config() -> str:
    """Get the base CI stage config path."""
    return str(Path(__file__).parent.parent / "stage_configs" / "minicpmo_ci.yaml")


def get_async_chunk_config() -> str:
    """Create an async_chunk enabled variant via modify_stage_config.

    Enables streaming from Stage 0 (thinker) → Stage 1 (tts) using
    llm2tts_async_chunk. Stage 1 → Stage 2 remains synchronous.
    """
    path = modify_stage_config(
        get_default_config(),
        updates={
            "async_chunk": True,
            "stage_args": {
                0: {
                    "engine_args.custom_process_next_stage_input_func": (
                        "vllm_omni.model_executor.stage_input_processors.minicpm4_5_o.llm2tts_async_chunk"
                    ),
                },
            },
        },
        deletes={"stage_args": {1: ["custom_process_input_func"]}},
    )
    return path


stage_configs = [get_default_config(), get_async_chunk_config()]

test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def get_question(prompt_type: str = "mix") -> str:
    """Return a user-facing prompt string for the given modality scenario."""
    prompts = {
        "mix": "What is recited in the audio? What is in this image?",
        "text_only": "What is the capital of China?",
        "image": "What is in this image?",
        "audio": "What is recited in the audio?",
    }
    return prompts.get(prompt_type, prompts["mix"])


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text_audio(omni_runner, omni_runner_handler) -> None:
    """
    Full 3-stage pipeline: text → text + audio.
    Deploy Setting: minicpmo_ci.yaml (sync) and async_chunk variant
    Input Modal: text
    Output Modal: text + audio
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "prompts": get_question("text_only"),
        "modalities": ["text", "audio"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Thinker stage text-only output.
    Deploy Setting: minicpmo_ci.yaml (sync) and async_chunk variant
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "prompts": get_question("text_only"),
        "modalities": ["text"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_text_to_text_audio(omni_runner, omni_runner_handler) -> None:
    """
    Vision encoder path: image + text → text + audio.
    Deploy Setting: minicpmo_ci.yaml (sync) and async_chunk variant
    Input Modal: image + text
    Output Modal: text + audio
    Input Setting: stream=False
    Datasets: single request
    """
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {
        "prompts": get_question("image"),
        "images": image,
        "modalities": ["text", "audio"],
    }
    omni_runner_handler.send_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_text_to_text_audio(omni_runner, omni_runner_handler) -> None:
    """
    Audio encoder path: audio + text → text + audio.
    Deploy Setting: minicpmo_ci.yaml (sync) and async_chunk variant
    Input Modal: audio + text
    Output Modal: text + audio
    Input Setting: stream=False
    Datasets: single request
    """
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": get_question("audio"),
        "audios": (audio, 16000),
        "modalities": ["text", "audio"],
    }
    omni_runner_handler.send_request(request_config)
