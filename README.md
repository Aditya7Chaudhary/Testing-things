# Testing-things

## eo2ir: satellite image to thermal IR with SD3.5 Medium + ControlNet

Generates a thermal-infrared-looking image from a satellite (optical) image, then compares it side by side with a real thermal frame of the same place.

- **Model:** [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) + [tensorart/SD3.5M-Controlnet-Canny](https://huggingface.co/tensorart/SD3.5M-Controlnet-Canny), zero-shot (no training).
- **Condition:** Canny edges of the satellite image, after a 1-99 % contrast stretch and thresholds that adapt to each image. They fix the layout.
- **Prompt:** one fixed prompt for day and one for night. It sets the thermal look.
- **Text encoders:** CLIP-L and CLIP-G only, with no T5 (saves ~9.5 GB). Prompts are encoded first and the encoders freed before the denoiser loads.
- **Precision:** bf16 on CPU, fp16 on CUDA. The VAE decodes in fp32.

### Sample data (included)

`samples/` holds 20 pixel-aligned pairs, so the comparison runs right after cloning: 10 day and 10 night, mostly towns and cities plus some farmland, shrubland and trees.
- `rgb/<id>.png`: an aerial image (USDA NAIP), 320 x 256, 8-bit RGB.
- `thermal/<id>.png`: an airborne thermal frame of the same ground, 16-bit grey. Day frames come from NASA HyTES and night frames from NASA MASTER.
- `metadata.csv`: id, time of day, scene type, sensor, source flight line, date, location, ground resolution (1.6-6.7 m/px) and altitude.

Each pair is the thermal frame's own georeferenced grid, with the NAIP imagery warped onto it and centre-cropped to 5:4. To use your own pairs, point `EO2IR_PAIRS` at a folder with the same `metadata.csv` columns.

### Run

```
pip install -r requirements.txt
hf auth login                 # SD3.5 Medium is gated: accept its licence on its Hugging Face page first
python -m eo2ir.generate --n 2 --out runs/demo         # 1 day + 1 night pair
python -m eo2ir.generate --n 20 --out runs/all         # all 20 sample pairs
python -m eo2ir.compare --out runs/all                 # re-score an existing run
```

The output folder gets these files:
- `<id>_cond.png`: the edge map;
- `<id>_ir.png`: the generated IR;
- `side_by_side.png`: satellite | condition | generated IR | real thermal;
- `scores.json`: NCC, SSIM and gradient NCC against the real frame, plus the same scores for the plain and inverted satellite grey as baselines.

### Hardware

- **NVIDIA GPU:** about 10 GB of VRAM, about a minute per image.
- **CPU:** about 10 GB of free RAM. With less, it swaps; on a 16 GB laptop that came to about 4 minutes per step.

### Licences

- **SD3.5 Medium and the ControlNet:** Stability AI Community License.
- **This code:** MIT.
- **Sample data:**
  - NAIP imagery: USDA Farm Service Agency, public domain (via Microsoft Planetary Computer).
  - HyTES and MASTER thermal data: NASA JPL / NASA Ames, distributed by NASA DAACs under the NASA Earth Science open data policy. Please cite the HyTES L1 and MASTER L1B products as their DAAC pages ask.
