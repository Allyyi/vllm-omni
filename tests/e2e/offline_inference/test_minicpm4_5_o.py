"""
E2E tests for MiniCPM-O 4.5 omni-modal model (text/image/audio → text + audio).

Verifies the 3-stage pipeline:
  Stage 0  Thinker  (Qwen3-8B + SigLip2 + Whisper) → text + hidden states
  Stage 1  TTS      (LlamaModel decoder)            → audio token IDs
  Stage 2  Token2Wav(CosyVoice2 flow + HiFi-GAN)    → 24 kHz waveform

Model weights : env var ``MINICPMO_MODEL_PATH``  (default: openbmb/MiniCPM-o-4_5)
Stage config  : env var ``MINICPMO_STAGE_CONFIG`` (default: tests/e2e/stage_configs/minicpmo_ci.yaml)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import generate_synthetic_audio, generate_synthetic_image
from tests.utils import hardware_test

os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_STAGE_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "stage_configs"

MODEL_PATH = os.environ.get("MINICPMO_MODEL_PATH", "openbmb/MiniCPM-o-4_5")
STAGE_CONFIG = os.environ.get(
    "MINICPMO_STAGE_CONFIG",
    str(_STAGE_CONFIGS_DIR / "minicpmo_ci.yaml"),
)

_SYSTEM_PROMPT = "You are a helpful assistant. You can accept audio and text input and output voice and text."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_prompt(user_content: str) -> str:
    """Build a MiniCPM-O chat-template prompt string."""
    return (
        f"<|im_start|>system\n{_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _create_omni():
    """Instantiate and return an ``Omni`` pipeline."""
    from vllm_omni import Omni

    if not Path(STAGE_CONFIG).exists():
        pytest.skip(f"Stage config not found: {STAGE_CONFIG}")

    return Omni(
        model=MODEL_PATH,
        stage_configs_path=STAGE_CONFIG,
        trust_remote_code=True,
    )


def _validate_text_output(outputs):
    """Assert that at least one output contains non-empty text."""
    for out in outputs:
        if getattr(out, "final_output_type", None) == "text":
            req_out = out.request_output
            if isinstance(req_out, list):
                req_out = req_out[0]
            text = req_out.outputs[0].text
            assert text is not None, "Text output is None"
            assert len(text) > 0, "Text output is empty"
            return text
    pytest.fail("No text output found in pipeline results")


def _validate_audio_output(outputs):
    """Assert that at least one output contains audio data."""
    for out in outputs:
        if getattr(out, "final_output_type", None) == "audio":
            req_out = out.request_output
            if isinstance(req_out, list):
                req_out = req_out[0]
            mm = req_out.outputs[0].multimodal_output
            assert mm is not None, "multimodal_output is None"
            audio = mm.get("audio")
            assert audio is not None, "No 'audio' key in multimodal_output"
            # audio may be np.ndarray, torch.Tensor, or bytes
            if isinstance(audio, np.ndarray):
                assert audio.size > 0, "Audio array is empty"
            else:
                assert len(audio) > 0, "Audio data is empty"
            return audio
    pytest.fail("No audio output found in pipeline results")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_minicpm4_5_o_text_to_text_audio():
    """
    Full 3-stage pipeline: text → text + audio.

    Input : plain text prompt
    Output: text (stage 0) + audio waveform (stage 2)
    """
    omni = _create_omni()
    try:
        prompt = _format_prompt("What is the capital of China?")
        outputs = list(omni.generate([{"prompt": prompt}]))

        assert len(outputs) > 0, "Pipeline produced no outputs"
        _validate_text_output(outputs)
        _validate_audio_output(outputs)
    finally:
        omni.close()


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_minicpm4_5_o_text_only():
    """
    Thinker stage text-only output (downstream stages still present but may
    produce minimal output with a short text response).

    Input : plain text prompt
    Output: text (stage 0)
    """
    omni = _create_omni()
    try:
        prompt = _format_prompt("Say hello.")
        outputs = list(omni.generate([{"prompt": prompt}]))

        assert len(outputs) > 0, "Pipeline produced no outputs"
        _validate_text_output(outputs)
    finally:
        omni.close()


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_minicpm4_5_o_image_text_to_audio():
    """
    Vision encoder path: image + text → text + audio.

    Input : synthetic image (16×16) + text prompt
    Output: text (stage 0) + audio waveform (stage 2)
    """
    omni = _create_omni()
    try:
        image_np = generate_synthetic_image(16, 16)["np_array"]
        user_content = "<image>./</image>What is in this image?"
        prompt = _format_prompt(user_content)

        outputs = list(
            omni.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image_np},
                    }
                ]
            )
        )

        assert len(outputs) > 0, "Pipeline produced no outputs"
        _validate_text_output(outputs)
        _validate_audio_output(outputs)
    finally:
        omni.close()


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"})
def test_minicpm4_5_o_audio_text_to_audio():
    """
    Audio encoder path: audio + text → text + audio.

    Input : synthetic audio (1 s, 16 kHz) + text prompt
    Output: text (stage 0) + audio waveform (stage 2)
    """
    omni = _create_omni()
    try:
        audio_np = generate_synthetic_audio(1, 1, 16000)["np_array"]
        if audio_np.ndim == 2:
            audio_np = audio_np.squeeze()

        user_content = "<audio>./</audio>What is recited in the audio?"
        prompt = _format_prompt(user_content)

        outputs = list(
            omni.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"audio": (audio_np, 16000)},
                    }
                ]
            )
        )

        assert len(outputs) > 0, "Pipeline produced no outputs"
        _validate_text_output(outputs)
        _validate_audio_output(outputs)
    finally:
        omni.close()
