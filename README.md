<!-- Badges -->
[![AGPL-3.0](https://custom-icon-badges.herokuapp.com/badge/license-AGPL--3.0-8BB80A.svg?logo=law&l)]() [![Python](https://custom-icon-badges.herokuapp.com/badge/Python-3572A5.svg?logo=Python&logoColor=white)]() <img src="https://img.shields.io/badge/-Ubuntu-6F52B5.svg?logo=ubuntu&style=flat">

# yoloe-savpe-detector

Realtime object detection using [YOLOE](https://github.com/THU-MIG/yoloe) visual prompt embeddings on an Intel RealSense camera stream. Click an object in a single frame, let SAM2 segment it, and the model tracks all matching objects in the live feed using SAVPE (Semantic-Activated Visual Prompt Embedding). Each detection also gets a rotated bounding box (xywhr) computed from the largest cluster of its instance mask.

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Intel RealSense camera (via `pyrealsense2`)
- CUDA-capable GPU (PyTorch is installed from the `cu130` index; both YOLOE and SAM2 run on CUDA device 0 by default)
- YOLOE segmentation weights (`yoloe-26x-seg.pt`) in the project root; change `MODEL_NAME` in `savpe.py` to use another variant (e.g. `yoloe-11l-seg.pt`)
- SAM2 weights (`sam2.1_b.pt`); Ultralytics downloads them into the project root on first run, so place the file there manually when running offline

## Installation

```bash
git clone https://github.com/R-Tatara/yoloe-savpe-detector.git
cd yoloe-savpe-detector
uv sync
```

## Project Structure

- `main.py` — RealSense camera setup and overall control flow
- `segmentation.py` — SAM2 click-to-mask selection and mask confirmation UI
- `savpe_prompt.py` — converts the SAM2 mask into the YOLOE visual prompt (enclosing bbox + class)
- `prompt_save.py` — saves the visual prompt (reference image, mask, `prompt.json`) to `prompt/`
- `prompt_load.py` — loads and validates the saved visual prompt
- `savpe.py` — YOLOE SAVPE initialization and the detection/tracking loop
- `clustering.py` — mask denoising, largest-cluster extraction and rotated bbox (xywhr) computation

## Usage

```bash
uv run python main.py
```


## License

AGPL-3.0

This project depends on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), so it is distributed under the same license.
