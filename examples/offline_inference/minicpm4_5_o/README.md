# MiniCPM-O 4.5

## Setup

### 1. Install vllm-omni from source

```bash
# Clone the repository
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni

# Install in development mode
pip install -e .
```

### 2. Configure stage settings (optional)

The pipeline stage config is auto-resolved from the model name. To customise
memory allocation for your hardware, see the
[stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/).

You can also pass a custom YAML via `--stage-configs-path`.

## Run examples

### Text → Text + Audio

```bash
cd examples/offline_inference/minicpm4_5_o
bash run_single_prompt.sh
```

Or directly:

```bash
python end2end.py --query-type text --output-dir output_minicpmo
```

### Image + Text → Text + Audio

```bash
# Using default image asset
python end2end.py --query-type use_image

# Using a local image file
python end2end.py --query-type use_image --image-path /path/to/image.jpg
```

### Audio + Text → Text + Audio

```bash
# Using default audio asset
python end2end.py --query-type use_audio

# Using a local audio file
python end2end.py --query-type use_audio --audio-path /path/to/clip.wav
```

### Mixed Modalities (Image + Audio + Text)

```bash
bash run_mixed_modalities.sh

# Or with local files
python end2end.py --query-type mixed_modalities \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/clip.wav
```

### Text-only output (skip audio generation)

```bash
python end2end.py --query-type text --modalities text
```

### Batch mode (from file)

```bash
python end2end.py --query-type text \
    --txt-prompts prompts.txt \
    --output-dir output_batch
```

## Supported query types

| Type | Input | Output |
|------|-------|--------|
| `text` | Plain text | Text + Audio |
| `use_image` | Image + text | Text + Audio |
| `use_audio` | Audio + text | Text + Audio |
| `mixed_modalities` | Image + Audio + text | Text + Audio |

Append `--modalities text` to any command to disable audio generation.
