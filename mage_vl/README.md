# Mage-VL Inference

<p align="center">
  <img src="assets/mage-vl-cover.png" alt="Mage-VL" width="100%">
</p>

A single checkpoint, `Mage-VL/Mage-VL-Base`, covers every Mage-VL capability:
image understanding, frame-sampled video, traditional H.264/HEVC codec video,
neural DCVC-RT codec video, and event-gated streaming. The model repository
bundles the codec processor, the neural codec package, and the StreamMind gate
weights, so no separate NVC or Streaming checkpoint is required.

| Capability | Script | Entry point |
|---|---|---|
| Image, frames, traditional codec, neural codec | `inference_base.py` | offline and SGLang online |
| Event-gated continuous video commentary | `inference_streaming.py` | offline |

## Installation

```bash
pip install -r mage_vl/requirements.txt
```

Codec-based video inference requires `ffmpeg` and `ffprobe` on `PATH`. The
traditional codec path uses the `cv-preinfer` command supplied by
`codec-video-prep`; the streaming frames backend uses Decord directly.

## Offline inference

Offline mode loads `AutoModelForCausalLM.from_pretrained` directly and supports
images, frame sampling, and both codec engines:

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

## Online inference

Online mode sends an image or sampled video frames to an OpenAI-compatible
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

Serve the checkpoint with the Mage-VL SGLang branch:

```bash
git clone -b feat/mage-vl https://github.com/kcz358/sglang
cd sglang
pip install -e 'python[all]'
python -m sglang.launch_server \
  --model-path Mage-VL/Mage-VL-Base \
  --trust-remote-code
```

## Streaming inference

Streaming inference processes a video causally in non-overlapping segments. The
gate stays silent on routine content and generates a caption only when a
response-worthy event is detected.

```bash
python mage_vl/inference_streaming.py \
  --video /path/to/clip.mp4 \
  --video_backend codec \
  --segment_sec 8
```

Use `--video_backend frames` for direct frame sampling. Additional controls
include `--num_frames`, `--cur_fps`, `--max_segments`, `--max_new_tokens`,
`--gate_threshold`, and `--attn_impl`.
