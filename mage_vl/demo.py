#!/usr/bin/env python
"""Mage-VL demo — image & video (chunk + codec) inference in one file.

Loads a Mage-VL multi-component model repo (transformer/ + tokenizer/ + processor/ +
neural_codec/) and answers a question about an image or a video. For video, patch
selection runs through a codec whose backend is selectable:

  * ``--codec traditional``  -> HEVC / h264 block-bit selection
  * ``--codec neural``       -> DCVC-RT neural-codec bit-cost selection (bundled)

Requirements:
  * ``--model`` = a Mage-VL repo dir OR a Hugging Face Hub repo id (e.g.
    ``microsoft/Mage-VL``); a repo id is downloaded/cached automatically. This file
    needs no weights of its own. The DCVC-RT source and ``.tar`` checkpoints for the
    neural codec ship inside ``<model>/neural_codec/`` and are found automatically —
    no ``DCVC_RT_ROOT`` needed.
  * ``ffmpeg`` / ``ffprobe`` on PATH (both codec backends).
  * transformers >= 5.3 (the remote modeling code); ``huggingface_hub`` for repo-id loading.

Examples::

    # image (local repo dir)
    python demo.py --model /path/Mage-VL --image cat.jpg --question "What is in this image?"

    # image (Hugging Face repo id — downloaded + cached)
    python demo.py --model microsoft/Mage-VL --image cat.jpg --question "What is in this image?"

    # video, neural codec
    python demo.py --model /path/Mage-VL --video clip.mp4 \
        --question "Describe what happens." --codec neural

    # video, traditional (HEVC) codec
    python demo.py --model /path/Mage-VL --video clip.mp4 --codec traditional
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_CODEC_ENGINE = {"traditional": "hevc", "neural": "dcvc-rt"}


def _load(model_dir: str, gpu: int):
    """Load model (transformer/) + processor (subdir-aware) from a Mage-VL repo.

    ``model_dir`` may be a local directory OR a Hugging Face Hub repo id (e.g.
    ``microsoft/Mage-VL``); a repo id is downloaded/cached to a local snapshot
    first so the multi-component layout (processor/ transformer/ neural_codec/)
    is resolved on disk.
    """
    import torch
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_dir)  # download/cache -> local path
    # processor code lives in <model>/processor/ — put it on sys.path so the
    # processing module and its siblings (video/codec processing) import.
    sys.path.insert(0, os.path.join(model_dir, "processor"))
    from processing_magevl import MageVLProcessor
    from transformers import AutoModelForCausalLM

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    processor = MageVLProcessor.from_pretrained(model_dir)  # tokenizer/ + processor/ subdirs
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(model_dir, "transformer"),
        trust_remote_code=True, torch_dtype="auto",
    ).to(device).eval()
    return model, processor, device


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="Mage-VL model repo: a local directory OR a Hugging Face Hub "
                         "repo id (e.g. microsoft/Mage-VL), downloaded and cached on first use")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="image file")
    src.add_argument("--video", help="video file")
    ap.add_argument("--question", default="Describe this in detail.")
    ap.add_argument("--codec", choices=list(_CODEC_ENGINE), default="neural",
                    help="video patch-selection codec: traditional (HEVC) or neural (DCVC-RT)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    import torch
    model, processor, device = _load(args.model, args.gpu)

    media_type = "image" if args.image else "video"
    messages = [{"role": "user", "content": [
        {"type": media_type},
        {"type": "text", "text": args.question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if args.image:
        from PIL import Image
        inputs = processor(text=[text], images=[Image.open(args.image).convert("RGB")],
                           return_tensors="pt")
    else:
        engine = _CODEC_ENGINE[args.codec]
        if engine == "dcvc-rt":
            # The DCVC-RT source + .tar checkpoints are bundled in <model>/neural_codec/
            # and resolved automatically; just pin the codec to the model's GPU.
            os.environ.setdefault(
                "DCVC_DEVICE", f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        try:
            inputs = processor(
                text=[text], videos=[args.video], video_backend="codec",
                codec_config={
                    "engine": engine,
                    "patch": int(processor.image_processor.patch_size),
                },
                return_tensors="pt",
            )
        except (RuntimeError, FileNotFoundError, ValueError, subprocess.SubprocessError) as e:
            sys.exit(f"[mage-vl] {args.codec} codec failed: {e}")

    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new = gen[0, inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(new, skip_special_tokens=True).strip()
    print("\n================ ANSWER ================\n" + answer + "\n")


if __name__ == "__main__":
    main()
