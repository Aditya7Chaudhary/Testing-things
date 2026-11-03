"""Side-by-side panels and similarity scores of generated IR vs the real thermal frame.

    python -m eo2ir.compare --out eo2ir_runs/zeroshot      # re-score an existing run

Scores per pair, on [0, 1] images (both 1-99 % stretched, 320 x 256):
- ncc: Pearson correlation of the pixel values (1 = same brightness pattern, < 0 = inverted polarity);
- ssim: structural similarity;
- grad_ncc: correlation of the gradient magnitudes (edges in the same places, polarity-free).
The same scores for the plain satellite grey and its inverse are the baselines: a translation is only useful where it
beats them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from eo2ir import common

COLUMNS = ("satellite render", "ControlNet condition", "generated IR", "real thermal")


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))


def grad(a: np.ndarray) -> np.ndarray:
    return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))


def scores(x: np.ndarray, real: np.ndarray) -> dict:
    return dict(
        ncc=round(ncc(x, real), 3),
        ssim=round(float(structural_similarity(x, real, data_range=1.0)), 3),
        grad_ncc=round(ncc(grad(x), grad(real)), 3),
    )


def panel(images: list[Image.Image], title: str) -> Image.Image:
    w, h, head, gap = 320, 256, 22, 6
    out = Image.new("RGB", (len(images) * (w + gap) - gap, h + head), "white")
    d = ImageDraw.Draw(out)
    d.text((4, 4), title, fill="black")
    for i, im in enumerate(images):
        out.paste(im.convert("RGB").resize((w, h), Image.BICUBIC), (i * (w + gap), head))
    return out


def compare_all(pairs: list[dict], out: Path) -> list[dict]:
    rows, panels = [], []
    for p in pairs:
        stem = p["id"]
        if not (out / f"{stem}_ir.png").exists():
            continue
        sat = common.rgb_image(common.PAIRS_ROOT / p["rgb_path"])
        real = common.thermal_unit(common.PAIRS_ROOT / p["thermal_path"])
        gen = common.grey_unit(Image.open(out / f"{stem}_ir.png"))
        grey = common.grey_unit(sat)
        r = dict(id=stem, time_of_day=p["time_of_day"], map_key=p["map_key"],
                 alt_m=round(float(p["altitude_above_takeoff_m"])),
                 generated=scores(gen, real), satellite_grey=scores(grey, real), inverted_grey=scores(1 - grey, real))
        rows.append(r)
        g = r["generated"]
        title = (f"{stem}  {p['time_of_day']}  {r['alt_m']} m  |  generated vs real: ncc {g['ncc']}  ssim {g['ssim']}"
                 f"  grad {g['grad_ncc']}   (satellite grey: ncc {r['satellite_grey']['ncc']})")
        panels.append(panel([sat, Image.open(out / f"{stem}_cond.png"), common.to_u8(gen), common.to_u8(real)],
                            title))
    if not rows:
        return rows
    head = panel([Image.new("RGB", (320, 1), "white")] * 4, "   |   ".join(COLUMNS))
    sheet = Image.new("RGB", (panels[0].width, head.height + sum(x.height + 8 for x in panels)), "white")
    sheet.paste(head, (0, 0))
    y = head.height
    for x in panels:
        sheet.paste(x, (0, y))
        y += x.height + 8
    sheet.save(out / "side_by_side.png")

    summary = {}
    for kind in ("generated", "satellite_grey", "inverted_grey"):
        for tod in ("all", "day", "night"):
            sel = [r[kind] for r in rows if tod == "all" or r["time_of_day"] == tod]
            if sel:
                summary[f"{kind}/{tod}"] = {k: round(float(np.mean([s[k] for s in sel])), 3) for k in sel[0]}
    (out / "scores.json").write_text(json.dumps(dict(summary=summary, pairs=rows), indent=1))
    print(json.dumps(summary, indent=1))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    ids = json.loads((a.out / "pairs.json").read_text())
    by_id = {r["id"]: r for r in common.load_rows()}
    compare_all([by_id[i] for i in ids], a.out)


if __name__ == "__main__":
    main()
