# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPM-O 4.5 stage input processors."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpm4_5_o import (
    _validate_stage_inputs,
    llm2tts,
    llm2tts_async_chunk,
    tts2token2wav,
)

pytestmark = [pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Helpers — lightweight stage / output mocks using SimpleNamespace
# ---------------------------------------------------------------------------

_HIDDEN_DIM = 4096
_SEQ_LEN = 10
_NUM_AUDIO_TOKENS = 20


def _make_thinker_output(*, has_hidden_states: bool = True):
    """Build a fake thinker stage output."""
    mm = {}
    if has_hidden_states:
        mm["llm_hidden_states"] = torch.randn(_SEQ_LEN, _HIDDEN_DIM)
        mm["llm_token_ids"] = torch.randint(0, 50000, (_SEQ_LEN,))
    output = SimpleNamespace(
        multimodal_output=mm,
        token_ids=list(range(_SEQ_LEN)),
    )
    return SimpleNamespace(outputs=[output])


def _make_tts_output(*, has_audio_tokens: bool = True):
    """Build a fake TTS stage output."""
    mm = {}
    if has_audio_tokens:
        mm["audio_token_ids"] = torch.randint(0, 5000, (_NUM_AUDIO_TOKENS,))
    output = SimpleNamespace(
        multimodal_output=mm,
        token_ids=list(range(_NUM_AUDIO_TOKENS)),
    )
    return SimpleNamespace(outputs=[output])


def _make_stage_list(*outputs):
    """Wrap outputs into a list of stage-like objects."""
    return [SimpleNamespace(engine_outputs=out) for out in outputs]


# ---------------------------------------------------------------------------
# _validate_stage_inputs
# ---------------------------------------------------------------------------


class TestValidateStageInputs:
    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_stage_inputs([], [])

    def test_invalid_stage_id_raises(self):
        stages = _make_stage_list([_make_thinker_output()])
        with pytest.raises(IndexError, match="Invalid stage_id"):
            _validate_stage_inputs(stages, [99])

    def test_none_outputs_raises(self):
        stages = [SimpleNamespace(engine_outputs=None)]
        with pytest.raises(RuntimeError, match="no outputs yet"):
            _validate_stage_inputs(stages, [0])

    def test_valid_returns_outputs(self):
        expected = [_make_thinker_output()]
        stages = _make_stage_list(expected)
        assert _validate_stage_inputs(stages, [0]) is expected


# ---------------------------------------------------------------------------
# llm2tts
# ---------------------------------------------------------------------------


class TestLlm2Tts:
    def test_basic(self):
        """With hidden states present, output has correct keys and shapes."""
        thinker_out = _make_thinker_output(has_hidden_states=True)
        stages = _make_stage_list([thinker_out])

        result = llm2tts(stages, [0])

        assert len(result) == 1
        info = result[0]["additional_information"]
        assert "llm_hidden_states" in info
        assert "llm_token_ids" in info
        assert info["llm_hidden_states"].shape == (_SEQ_LEN, _HIDDEN_DIM)
        assert info["llm_token_ids"].shape == (_SEQ_LEN,)
        # prompt_token_ids should be a dummy [0]
        assert result[0]["prompt_token_ids"] == [0]

    def test_fallback_when_hidden_states_missing(self):
        """When llm_hidden_states is absent, falls back to zeros + token_ids."""
        thinker_out = _make_thinker_output(has_hidden_states=False)
        stages = _make_stage_list([thinker_out])

        result = llm2tts(stages, [0])

        assert len(result) == 1
        info = result[0]["additional_information"]
        # Hidden states should be zeros placeholder
        assert torch.all(info["llm_hidden_states"] == 0)
        assert info["llm_hidden_states"].shape == (_SEQ_LEN, _HIDDEN_DIM)
        # Token IDs from output.token_ids
        assert info["llm_token_ids"].shape == (_SEQ_LEN,)

    def test_multiple_outputs(self):
        """Handles multiple thinker outputs (batch)."""
        out1 = _make_thinker_output(has_hidden_states=True)
        out2 = _make_thinker_output(has_hidden_states=True)
        stages = _make_stage_list([out1, out2])

        result = llm2tts(stages, [0])

        assert len(result) == 2


# ---------------------------------------------------------------------------
# tts2token2wav
# ---------------------------------------------------------------------------


class TestTts2Token2Wav:
    def test_basic(self):
        """With audio_token_ids present, output has correct keys and shapes."""
        tts_out = _make_tts_output(has_audio_tokens=True)
        stages = _make_stage_list(None, [tts_out])  # stage 0=None, stage 1=tts

        result = tts2token2wav(stages, [1])

        assert len(result) == 1
        info = result[0]["additional_information"]
        assert "audio_token_ids" in info
        assert info["audio_token_ids"].shape == (_NUM_AUDIO_TOKENS,)
        assert result[0]["prompt_token_ids"] == [0]

    def test_fallback_when_audio_tokens_missing(self):
        """When audio_token_ids is absent, uses output.token_ids."""
        tts_out = _make_tts_output(has_audio_tokens=False)
        stages = _make_stage_list(None, [tts_out])

        result = tts2token2wav(stages, [1])

        info = result[0]["additional_information"]
        assert info["audio_token_ids"].shape == (_NUM_AUDIO_TOKENS,)

    def test_multiple_outputs(self):
        """Handles multiple TTS outputs."""
        out1 = _make_tts_output(has_audio_tokens=True)
        out2 = _make_tts_output(has_audio_tokens=True)
        stages = _make_stage_list(None, [out1, out2])

        result = tts2token2wav(stages, [1])

        assert len(result) == 2


# ---------------------------------------------------------------------------
# llm2tts_async_chunk
# ---------------------------------------------------------------------------


class TestLlm2TtsAsyncChunk:
    def test_returns_none_when_no_data_and_not_finished(self):
        result = llm2tts_async_chunk(
            transfer_manager=None,
            pooling_output={},
            request=None,
            is_finished=False,
        )
        assert result is None

    def test_returns_payload_when_finished(self):
        result = llm2tts_async_chunk(
            transfer_manager=None,
            pooling_output={},
            request=None,
            is_finished=True,
        )
        assert result is not None
        assert result["finished"].item() is True
        # Fallback tensors should be created
        assert result["llm_hidden_states"].shape == (1, _HIDDEN_DIM)
        assert result["llm_token_ids"].shape == (1,)

    def test_returns_payload_when_data_present(self):
        hidden = torch.randn(5, _HIDDEN_DIM)
        token_ids = torch.randint(0, 1000, (5,))
        result = llm2tts_async_chunk(
            transfer_manager=None,
            pooling_output={
                "llm_hidden_states": hidden,
                "llm_token_ids": token_ids,
            },
            request=None,
            is_finished=False,
        )
        assert result is not None
        assert result["finished"].item() is False
        assert result["llm_hidden_states"].shape == (5, _HIDDEN_DIM)
        assert result["llm_token_ids"].shape == (5,)
