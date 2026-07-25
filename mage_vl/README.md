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

## Examples

Two sample inputs ship with the repository:

| File | Content |
|---|---|
| `mage_vl/assets/examples/dog.jpg` | Photo of a dog sitting in front of a patterned rug |
| `mage_vl/assets/examples/soccer-broadcast.mp4` | 30s, 960x540 football broadcast clip |

## Offline inference

Offline mode loads `AutoModelForCausalLM.from_pretrained` directly and supports
images, frame sampling, and both codec engines:

```bash
python mage_vl/inference_base.py \
  --mode offline \
  --image mage_vl/assets/examples/dog.jpg \
  --question "Describe this image in detail."
```

> The image depicts a dog sitting on a patterned rug. The dog appears to be a
> medium-sized breed with a thick, fluffy coat. Its fur is primarily white with
> patches of black and brown. The dog's ears are perked up, and it has a calm and
> attentive expression. [...]

```bash
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend frames \
  --num-frames 32 \
  --question "Describe this video."
```

> The video opens with a man in a black polo shirt, sporting a short haircut,
> standing in a stadium. He is holding a yellow microphone with the BBC Sport
> logo on it. The background reveals a large crowd of spectators. [...]

```bash
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend codec \
  --codec-engine traditional \
  --num-frames 32 \
  --question "Describe this video."
```

> The video opens with a BBC Sport broadcast, featuring a presenter in a black
> shirt holding a yellow microphone. The background reveals a packed stadium,
> with the scoreboard displaying "ENG 1 ARG 2 FT", indicating the final score of
> the match. [...]

```bash
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend codec \
  --codec-engine neural \
  --num-frames 32 \
  --question "Describe this video."
```

> The video opens with a BBC Sport broadcast, featuring a presenter standing in a
> stadium filled with spectators. The presenter, dressed in a black shirt, holds
> a yellow BBC Sport microphone and wears a black earpiece. [...]

## Online inference

Online mode sends an image or sampled video frames to an OpenAI-compatible
SGLang server:

```bash
python mage_vl/inference_base.py \
  --mode online \
  --image mage_vl/assets/examples/dog.jpg \
  --question "Describe this image in detail." \
  --base-url http://localhost:30000/v1

python mage_vl/inference_base.py \
  --mode online \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
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
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video_backend codec \
  --segment_sec 8
```

```text
[t=0.0-8.0s] gate=silence (p=0.19)
[t=8.0-16.0s] gate=response (p=0.55) -> The video features a live sports broadcast from BBC Sport, set in a large stadium filled with spectators. The broadcast focuses on a football match between England and Argentina, with the score displayed as England 1, Argentina 2. [...]
[t=16.0-24.0s] gate=response (p=0.73) -> The video features a sports broadcast set in a large stadium filled with spectators. Four commentators are gathered around a table with a 'BBC Sport' logo, each holding a yellow microphone. [...]
[t=24.0-30.0s] gate=silence (p=0.31)
```

The gate is trained on codec inputs, so `--video_backend codec` is the intended
setting. Use `--video_backend frames` for direct frame sampling. Additional
controls include `--num_frames`, `--cur_fps`, `--max_segments`,
`--max_new_tokens`, `--gate_threshold`, and `--attn_impl`.
