"""Satellite -> thermal IR with Stable Diffusion 3.5 Medium + a ControlNet, then a side-by-side comparison with the
real thermal frame.

    python -m eo2ir.generate --n 10 --out eo2ir_runs/zeroshot                    # pretrained Canny ControlNet
    python -m eo2ir.generate --n 10 --controlnet <trained dir> --condition rgb    # a ControlNet fine-tuned on the pairs

Zero-shot (default): the condition is the Canny edge map of the satellite render, so the layout comes from the map
and the thermal appearance from the prompt (day or night). A ControlNet fine-tuned on the pairs takes the
satellite render itself as the condition (`--condition rgb`) and learns the appearance from data.

Runs on CPU in bf16 (about 10 GB of weights; no T5 text encoder) or on CUDA in fp16. Text embeddings are computed
first and the text encoders freed before the transformer and ControlNet load, so peak memory is the denoiser's.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import SD3ControlNetModel, SD3Transformer2DModel, StableDiffusion3ControlNetPipeline
from PIL import Image

from eo2ir import common
from eo2ir.compare import compare_all


def encode_prompts(model: str, device: str) -> dict:
    """{day|night: (prompt_embeds, negative_embeds, pooled, negative_pooled)} with CLIP-L + CLIP-G only."""
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        model, transformer=None, controlnet=None, vae=None, text_encoder_3=None, tokenizer_3=None,
        variant="fp16", torch_dtype=torch.float32,
    ).to(device)
    # without T5 the pipeline pads with zeros of the transformer's joint_attention_dim: give it the config alone
    cfg = SD3Transformer2DModel.load_config(model, subfolder="transformer")
    pipe.transformer = SimpleNamespace(config=SimpleNamespace(joint_attention_dim=cfg["joint_attention_dim"]))
    out = {}
    with torch.no_grad():
        for k, prompt in common.PROMPTS.items():
            pe, ne, pp, npp = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None, negative_prompt=common.NEGATIVE,
                device=device, do_classifier_free_guidance=True,
            )
            out[k] = (pe, ne, pp, npp)
    del pipe
    gc.collect()
    return out


def load_denoiser(model: str, controlnet: str, device: str, dtype: torch.dtype):
    cn = SD3ControlNetModel.from_pretrained(controlnet, torch_dtype=dtype)
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        model, controlnet=cn, text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None, torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def decode(pipe, latents: torch.Tensor) -> Image.Image:
    """Latents -> image with the VAE in fp32 (bf16 decoding bands on CPU)."""
    vae = pipe.vae.to(torch.float32)
    lat = latents.to(torch.float32) / vae.config.scaling_factor + vae.config.shift_factor
    img = vae.decode(lat, return_dict=False)[0]
    pipe.vae.to(pipe.transformer.dtype)
    img = (img[0].clamp(-1, 1) + 1) / 2
    return Image.fromarray(np.round(img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eo2ir_runs/zeroshot"))
    ap.add_argument("--model", default=common.BASE_MODEL)
    ap.add_argument("--controlnet", default=common.CANNY_CONTROLNET)
    ap.add_argument("--condition", choices=("canny", "rgb"), default="canny")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--control-scale", type=float, default=0.8)
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    torch.set_num_threads(max(1, torch.get_num_threads()))
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = common.pick_pairs(a.n, a.split, a.seed)
    (a.out / "pairs.json").write_text(json.dumps([p["id"] for p in pairs], indent=1))
    size = (a.width, a.height)

    t0 = time.perf_counter()
    embeds = encode_prompts(a.model, device)
    print(f"prompts encoded in {time.perf_counter() - t0:.0f} s", flush=True)
    t0 = time.perf_counter()
    pipe = load_denoiser(a.model, a.controlnet, device, dtype)
    print(f"denoiser loaded in {time.perf_counter() - t0:.0f} s on {device} ({dtype})", flush=True)

    log = []
    for i, p in enumerate(pairs, 1):
        stem = p["id"]
        if (a.out / f"{stem}_ir.png").exists():
            continue
        sat = common.rgb_image(common.PAIRS_ROOT / p["rgb_path"])
        cond = common.canny_condition(sat, size) if a.condition == "canny" else sat.resize(size, Image.BICUBIC)
        cond.save(a.out / f"{stem}_cond.png")
        pe, ne, pp, npp = (t.to(device, dtype) for t in embeds[p["time_of_day"]])
        t0 = time.perf_counter()

        def step_log(_pipe, step, _t, kw, t0=t0, i=i):
            print(f"  [{i}/{len(pairs)}] step {step + 1}/{a.steps}  {time.perf_counter() - t0:.0f} s", flush=True)
            return kw

        lat = pipe(
            prompt_embeds=pe, negative_prompt_embeds=ne, pooled_prompt_embeds=pp, negative_pooled_prompt_embeds=npp,
            control_image=cond, width=a.width, height=a.height, num_inference_steps=a.steps,
            guidance_scale=a.guidance, controlnet_conditioning_scale=a.control_scale,
            generator=torch.Generator("cpu").manual_seed(a.seed + i), output_type="latent",
            callback_on_step_end=step_log,
        ).images
        img = decode(pipe, lat).convert("L").resize((320, 256), Image.BICUBIC)
        img.save(a.out / f"{stem}_ir.png")
        secs = time.perf_counter() - t0
        log.append(dict(id=stem, time_of_day=p["time_of_day"], map_key=p["map_key"], seconds=round(secs, 1)))
        (a.out / "timings.json").write_text(json.dumps(log, indent=1))
        print(f"[{i}/{len(pairs)}] {stem} ({p['time_of_day']}) {secs:.0f} s", flush=True)

    compare_all(pairs, a.out)


if __name__ == "__main__":
    main()
