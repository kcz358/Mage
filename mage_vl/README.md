# Mage-VL Inference

Inference entry points for the Mage-VL model family.

| Model | Purpose | Script | Checkpoint |
|---|---|---|---|
| Mage-VL-Base | Image understanding | `inference_base.py` | `Mage-VL/Mage-VL-Base` |
| Mage-VL-NVC | Neural/traditional codec video understanding | `demo.py` | `Mage-VL/Mage-VL-NVC` |
| Mage-VL-Streaming | Event-gated continuous video commentary | `inference_streaming.py` | `Mage-VL/Mage-VL-Streaming` |

## Installation

Install all Base, NVC, and Streaming dependencies:

```bash
pip install -r mage_vl/requirements.txt
```

Codec-based video inference requires `ffmpeg` and `ffprobe` on `PATH`. The
traditional codec path uses the `cv-preinfer` command supplied by
`codec-video-prep`; the Streaming frames backend uses Decord directly.

## Mage-VL-Base

Offline mode loads `AutoModelForCausalLM.from_pretrained` directly. It supports
images, 32-frame video sampling, and codec video input:

```bash
python mage_vl/inference_base.py \
  --mode offline \
  --image cat.jpg \
  --question "What is in this image?"

python mage_vl/inference_base.py \
  --mode offline \
  --video clip.mp4 \
  --video-backend frames \
  --num-frames 32 \
  --question "Describe this video."

python mage_vl/inference_base.py \
  --mode offline \
  --video clip.mp4 \
  --video-backend codec \
  --codec-engine traditional \
  --num-frames 32 \
  --question "Describe this video."

python mage_vl/inference_base.py \
  --mode offline \
  --video clip.mp4 \
  --video-backend codec \
  --codec-engine neural \
  --num-frames 32 \
  --question "Describe this video."
```

Online mode sends an image or 32 sampled video frames to an OpenAI-compatible
SGLang server:

```bash
python mage_vl/inference_base.py \
  --mode online \
  --image cat.jpg \
  --question "What is in this image?" \
  --base-url http://localhost:30000/v1

python mage_vl/inference_base.py \
  --mode online \
  --video clip.mp4 \
  --num-frames 32 \
  --question "Describe this video." \
  --base-url http://localhost:30000/v1
```

Serve the Base checkpoint with the Mage-VL SGLang branch:

```bash
git clone -b feat/mage-vl https://github.com/kcz358/sglang
cd sglang
pip install -e 'python[all]'
python -m sglang.launch_server \
  --model-path Mage-VL/Mage-VL-Base \
  --trust-remote-code
```

## Mage-VL-NVC

The NVC model supports neural DCVC-RT and traditional H.264/HEVC patch
selection. The model repository bundles the neural codec and checkpoints.

```bash
# Neural codec
python mage_vl/demo.py \
  --model Mage-VL/Mage-VL-NVC \
  --video clip.mp4 \
  --codec neural \
  --question "Describe what happens in this video."

# Traditional codec
python mage_vl/demo.py \
  --model Mage-VL/Mage-VL-NVC \
  --video clip.mp4 \
  --codec traditional
```

## Mage-VL-Streaming

Streaming inference processes a video causally in non-overlapping windows. The
event gate remains silent on routine content and generates a caption when a
response-worthy event is detected.

```bash
python mage_vl/inference_streaming.py \
  --video /path/to/clip.mp4 \
  --checkpoint Mage-VL/Mage-VL-Streaming \
  --video_backend codec \
  --segment_sec 30
```

Use `--video_backend frames` for direct frame sampling. Additional controls
include `--num_frames`, `--cur_fps`, `--max_segments`, `--max_new_tokens`,
`--gate_threshold`, and `--attn_impl`.
