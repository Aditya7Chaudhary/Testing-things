# Testing-things

## eo2ir: satellite image to thermal IR with SD3.5 Medium + ControlNet

Generates a thermal-infrared-looking image from a satellite (optical) image, then compares it side by side with a real thermal frame of the same place.

- **Model:** [Stable Diffusion 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) + [tensorart/SD3.5M-Controlnet-Canny](https://huggingface.co/tensorart/SD3.5M-Controlnet-Canny), zero-shot (no training).
- **Condition:** Canny edges of the satellite image, after a 1-99 % contrast stretch and thresholds that adapt to each image. They fix the layout.
- **Prompt:** one fixed prompt for day and one for night. It sets the thermal look.
- **Text encoders:** CLIP-L and CLIP-G only, with no T5 (saves ~9.5 GB). Prompts are encoded first and the encoders freed before the denoiser loads.
- **Precision:** bf16 on CPU, fp16 on CUDA. The VAE decodes in fp32.

### Data layout

`EO2IR_PAIRS` points to a folder with `metadata.csv` (columns `id, split, time_of_day, map_key, rgb_path, thermal_path, altitude_above_takeoff_m`). Each row is a satellite image (`rgb_path`, 8-bit RGB) pixel-aligned with a thermal frame (`thermal_path`, 16-bit grey), both 320 x 256. No data is included in this repo.

### Run

```
pip install -r requirements.txt
hf auth login                 # SD3.5 Medium is gated: accept its licence on its Hugging Face page first
set EO2IR_PAIRS=D:\path\to\pairs        # PowerShell: $env:EO2IR_PAIRS = "D:\path\to\pairs"
python -m eo2ir.generate --n 2 --out runs/zeroshot_2        # 1 day + 1 night pair from the test split
python -m eo2ir.compare --out runs/zeroshot_2               # re-score an existing run
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
