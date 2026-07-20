<!-- Badges -->
[![AGPL-3.0](https://custom-icon-badges.herokuapp.com/badge/license-AGPL--3.0-8BB80A.svg?logo=law&l)]() [![Python](https://custom-icon-badges.herokuapp.com/badge/Python-3572A5.svg?logo=Python&logoColor=white)]() <img src="https://img.shields.io/badge/-Ubuntu-6F52B5.svg?logo=ubuntu&style=flat">

# yoloe-savpe-detector

Realtime object detection using [YOLOE](https://github.com/THU-MIG/yoloe) visual prompt embeddings on an Intel RealSense camera stream. Select any object in a single frame with your mouse, and the model tracks all matching objects in the live feed using SAVPE (Semantic-Activated Visual Prompt Embedding).

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Intel RealSense camera (via `pyrealsense2`)
- CUDA-capable GPU (PyTorch is installed from the `cu130` index)
- YOLOE segmentation weights (`yoloe-11s-seg.pt`) in the project root

## Installation

```bash
git clone https://github.com/R-Tatara/yoloe-savpe-detector.git
cd yoloe-savpe-detector
uv sync
```

## Usage

```bash
uv run python main.py
```

- A window opens showing the first camera frame — drag a box around the object you want to track, then press Enter
- The live detection window then displays all matches with tracking IDs (BoT-SORT)
- Press `q` to quit

## LISENCE

AGPL-3.0

This project depends on [Ultralytics](https://github.com/ultralytics/ultralytics) (AGPL-3.0), so it is distributed under the same license.
