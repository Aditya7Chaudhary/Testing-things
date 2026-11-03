"""Shared pieces of the EO -> IR strategy: the paired dataset, image conversions, the ControlNet condition and the
prompts. The pair folder (PAIRS_ROOT, env EO2IR_PAIRS) holds `metadata.csv` and, per sample, a 320 x 256 satellite
image (`rgb_path`) pixel-aligned with a real 16-bit thermal frame (`thermal_path`)."""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PAIRS_ROOT = Path(os.environ.get("EO2IR_PAIRS", Path(__file__).resolve().parents[1] / "samples"))
BASE_MODEL = "stabilityai/stable-diffusion-3.5-medium"
CANNY_CONTROLNET = "tensorart/SD3.5M-Controlnet-Canny"

PROMPTS = {
    "day": (
        "aerial thermal infrared image, top-down nadir view from a drone, grayscale long-wave infrared, daytime, "
        "sun-heated asphalt roads, rooftops and bare soil glow bright, trees, grass and water are dark, "
        "soft thermal contrast, monochrome"
    ),
    "night": (
        "aerial thermal infrared image, top-down nadir view from a drone, grayscale long-wave infrared, at night, "
        "buildings, roads and water bodies are warm and bright, vegetation and open soil are cold and dark, "
        "low contrast, slightly blurry, monochrome"
    ),
}
NEGATIVE = (
    "colour, colorful, RGB photograph, saturated, cartoon, painting, text, watermark, people, oblique view, sky, "
    "low quality, distorted"
)


def load_rows(split: str | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(PAIRS_ROOT / "metadata.csv", encoding="utf-8")))
    return [r for r in rows if split is None or r["split"] == split]


def pick_pairs(n: int, split: str = "test", seed: int = 0) -> list[dict]:
    """n pairs, half day and half night, each half spread evenly over its maps (one thermal source each), seeded."""
    by_map = defaultdict(list)
    for r in load_rows(split):
        by_map[r["map_key"]].append(r)
    rng = random.Random(seed)
    out = []
    for tod, want in (("day", n // 2), ("night", n - n // 2)):
        keys = sorted(k for k in by_map if by_map[k][0]["time_of_day"] == tod)
        left = {k: rng.sample(by_map[k], len(by_map[k])) for k in keys}  # each pair at most once
        got = []
        while len(got) < want and any(left.values()):
            for k in keys:
                if len(got) < want and left[k]:
                    got.append(left[k].pop())
        out += got
    return sorted(out, key=lambda r: (r["time_of_day"], r["map_key"], r["id"]))


def thermal_unit(path: Path) -> np.ndarray:
    """16-bit thermal frame -> float [0, 1] by its 1-99 % stretch (the frames carry a random auto-gain)."""
    a = np.array(Image.open(path)).astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def grey_unit(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L")).astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def canny_condition(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """White-on-black Canny edges of the satellite render at the generation size (the tensorart ControlNet's input).
    The grey is stretched 1-99 % first (renders can span only ~40 grey levels), then the thresholds follow the
    gradient magnitudes' percentiles, so every scene gives a similar edge density whatever its contrast."""
    g = np.round(grey_unit(img.resize(size, Image.BICUBIC)) * 255).astype(np.uint8)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    lo, hi = np.percentile(mag, (80, 92))
    edges = cv2.Canny(g, float(lo), float(hi), L2gradient=True)
    return Image.fromarray(edges).convert("RGB")


def to_u8(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(a, 0, 1) * 255).astype(np.uint8))
