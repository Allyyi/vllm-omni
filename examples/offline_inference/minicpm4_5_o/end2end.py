# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline inference example for MiniCPM-O 4.5 using vLLM-Omni.

This example demonstrates the 3-stage pipeline:
  Stage 0  Thinker  (Qwen3-8B + SigLip2 + Whisper) → text + hidden states
  Stage 1  TTS      (LlamaModel decoder)            → audio token IDs
  Stage 2  Token2Wav(CosyVoice2 flow + HiFi-GAN)    → 24 kHz waveform

Usage
-----
    # Install vllm-omni from source first:
    #   cd /path/to/vllm-omni && pip install -e .
    #
    # Then run this example:
    python end2end.py --query-type text
    python end2end.py --query-type text --modalities text          # text-only output
    python end2end.py --query-type use_image --image-path img.jpg
    python end2end.py --query-type use_audio --audio-path clip.wav
    python end2end.py --query-type mixed_modalities

    # See --help for all options.
"""

import os
import time
from typing import NamedTuple

import numpy as np
import soundfile as sf
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.entrypoints.omni import Omni

SEED = 42

SYSTEM_PROMPT = "You are a helpful assistant. You can accept audio and text input and output voice and text."


class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# ---------------------------------------------------------------------------
# Query builders
# ---------------------------------------------------------------------------


def get_text_query(question: str | None = None) -> QueryResult:
    """Build a plain-text query."""
    if question is None:
        question = "What is the capital of China? Answer in 15 words."
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return QueryResult(inputs={"prompt": prompt}, limit_mm_per_prompt={})


def get_image_query(image_path: str | None = None) -> QueryResult:
    """Build a query with an image input."""
    from PIL import Image
    from vllm.multimodal.image import convert_image_mode

    question = "What is in this image? Describe it briefly."
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<image>./</image>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        from vllm.assets.image import ImageAsset

        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"image": image_data},
        },
        limit_mm_per_prompt={"image": 1},
    )


def get_audio_query(
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """Build a query with an audio input."""
    question = "What is recited in the audio?"
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<audio>./</audio>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    if audio_path:
        import librosa

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        from vllm.assets.audio import AudioAsset

        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {"audio": audio_data},
        },
        limit_mm_per_prompt={"audio": 1},
    )


def get_mixed_modalities_query(
    image_path: str | None = None,
    audio_path: str | None = None,
    sampling_rate: int = 16000,
) -> QueryResult:
    """Build a query with both image and audio input."""
    from PIL import Image
    from vllm.multimodal.image import convert_image_mode

    question = "What is recited in the audio? What is the content of this image?"
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"<audio>./</audio><image>./</image>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Load image
    if image_path:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        pil_image = Image.open(image_path)
        image_data = convert_image_mode(pil_image, "RGB")
    else:
        from vllm.assets.image import ImageAsset

        image_data = convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")

    # Load audio
    if audio_path:
        import librosa

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        audio_signal, sr = librosa.load(audio_path, sr=sampling_rate)
        audio_data = (audio_signal.astype(np.float32), sr)
    else:
        from vllm.assets.audio import AudioAsset

        audio_data = AudioAsset("mary_had_lamb").audio_and_sample_rate

    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "audio": audio_data,
                "image": image_data,
            },
        },
        limit_mm_per_prompt={"audio": 1, "image": 1},
    )


# ---------------------------------------------------------------------------
# Query map
# ---------------------------------------------------------------------------
query_map: dict[str, callable] = {
    "text": get_text_query,
    "use_image": get_image_query,
    "use_audio": get_audio_query,
    "mixed_modalities": get_mixed_modalities_query,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args):
    model_name = args.model

    # Resolve query
    query_func = query_map[args.query_type]
    if args.query_type == "mixed_modalities":
        query_result = query_func(
            image_path=args.image_path,
            audio_path=args.audio_path,
            sampling_rate=args.sampling_rate,
        )
    elif args.query_type == "use_image":
        query_result = query_func(image_path=args.image_path)
    elif args.query_type == "use_audio":
        query_result = query_func(audio_path=args.audio_path, sampling_rate=args.sampling_rate)
    else:
        query_result = query_func()

    # Build Omni pipeline (auto-resolves stage config from model name)
    omni_kwargs = dict(
        model=model_name,
        trust_remote_code=True,
        log_stats=args.log_stats,
    )
    if args.stage_configs_path:
        omni_kwargs["stage_configs_path"] = args.stage_configs_path

    omni_llm = Omni(**omni_kwargs)

    # Per-stage sampling params
    thinker_sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=100,
        max_tokens=2048,
        seed=SEED,
        detokenize=True,
    )
    tts_sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        top_k=50,
        max_tokens=4096,
        seed=SEED,
        detokenize=False,
    )
    token2wav_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        seed=SEED,
        detokenize=False,
    )

    sampling_params_list = [
        thinker_sampling_params,
        tts_sampling_params,
        token2wav_sampling_params,
    ]

    # Build prompt list
    if args.txt_prompts is None:
        prompts = [query_result.inputs for _ in range(args.num_prompts)]
    else:
        assert args.query_type == "text", "--txt-prompts only works with --query-type text"
        with open(args.txt_prompts, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
            prompts = [get_text_query(ln).inputs for ln in lines]
            print(f"[Info] Loaded {len(prompts)} prompts from {args.txt_prompts}")

    if args.modalities is not None:
        output_modalities = args.modalities.split(",")
        for prompt in prompts:
            prompt["modalities"] = output_modalities

    # Run inference
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    omni_generator = omni_llm.generate(prompts, sampling_params_list)

    for stage_outputs in omni_generator:
        if stage_outputs.final_output_type == "text":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                text_output = output.outputs[0].text
                out_txt = os.path.join(output_dir, f"{request_id}.txt")
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(f"Prompt:\n{output.prompt}\n\nResponse:\n{text_output}\n")
                print(f"[text]  request={request_id}  → {out_txt}")
                print(f"        {text_output[:200]}")

        elif stage_outputs.final_output_type == "audio":
            for output in stage_outputs.request_output:
                request_id = output.request_id
                audio_tensor = output.outputs[0].multimodal_output["audio"]
                out_wav = os.path.join(output_dir, f"output_{request_id}.wav")
                if hasattr(audio_tensor, "detach"):
                    audio_np = audio_tensor.detach().cpu().numpy()
                elif isinstance(audio_tensor, np.ndarray):
                    audio_np = audio_tensor
                else:
                    audio_np = np.frombuffer(audio_tensor, dtype=np.float32)
                sf.write(out_wav, audio_np, samplerate=24000)
                print(f"[audio] request={request_id}  → {out_wav}")

    elapsed = time.time() - start_time
    print(f"\nDone. {len(prompts)} request(s) processed in {elapsed:.1f}s")
    print(f"Outputs saved to: {output_dir}/")

    omni_llm.close()


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="MiniCPM-O 4.5 offline inference example")
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/MiniCPM-o-4_5",
        help="Model name or path (default: openbmb/MiniCPM-o-4_5)",
    )
    parser.add_argument(
        "--stage-configs-path",
        type=str,
        default=None,
        help="Path to stage config YAML (auto-resolved from model if omitted)",
    )
    parser.add_argument(
        "--query-type",
        type=str,
        default="text",
        choices=list(query_map.keys()),
        help="Type of query to run (default: text)",
    )
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to generate")
    parser.add_argument("--output-dir", type=str, default="output_minicpmo", help="Output directory")
    parser.add_argument("--txt-prompts", type=str, default=None, help="File with text prompts (one per line)")
    parser.add_argument("--image-path", type=str, default=None, help="Path to a local image file")
    parser.add_argument("--audio-path", type=str, default=None, help="Path to a local audio file")
    parser.add_argument("--sampling-rate", type=int, default=16000, help="Audio sampling rate in Hz")
    parser.add_argument(
        "--modalities",
        type=str,
        default=None,
        help="Comma-separated output modalities, e.g. 'text' or 'text,audio'",
    )
    parser.add_argument("--log-stats", action="store_true", default=False, help="Enable stats logging")
    args = parser.parse_args()

    main(args)
