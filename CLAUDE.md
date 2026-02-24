# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```bash
pip install -e .
```

For the streaming module (separate install):
```bash
pip install -r da3_streaming/requirements.txt
```

Download model weights for streaming:
```bash
bash da3_streaming/scripts/download_weights.sh
```

## Linting and Code Quality

```bash
# Run pre-commit hooks on all files
pre-commit run --all-files

# Flake8 linting (config in .flake8)
flake8 src/
```

## Running the Interfaces

**Gradio Web UI:**
```bash
python -m depth_anything_3.app.gradio_app
```

**CLI (see docs/CLI.md for full options):**
```bash
python -m depth_anything_3.cli --help
```

**Python API (see docs/API.md):**
```python
from depth_anything_3.api import DepthAnything3
```

**Benchmarking (see docs/BENCHMARK.md):**
```bash
python -m depth_anything_3.bench.evaluator --help
```

## Architecture

The codebase is a depth estimation system with three access layers over a shared model core:

### Entry Points
- `src/depth_anything_3/api.py` — Public Python API for programmatic use
- `src/depth_anything_3/cli.py` — CLI for batch processing
- `src/depth_anything_3/app/gradio_app.py` — Interactive Gradio web UI

### Model Core (`src/depth_anything_3/model/`)
- `da3.py` — Main Depth-Anything-3 model
- `dpt.py` / `dualdpt.py` / `gsdpt.py` — Dense Prediction Transformer variants (single-view, dual, Gaussian-splatting-integrated)
- `dinov2/` — DINOv2 self-supervised ViT backbone
- `cam_enc.py` / `cam_dec.py` — Camera parameter encoder/decoder for multi-view inference
- `reference_view_selector.py` — Selects optimal reference views for multi-view depth fusion

### Service Layer (`src/depth_anything_3/services/`)
Sits between the entry points and model core:
- `inference_service.py` — Orchestrates model inference
- `backend.py` — Backend processing logic
- `input_handlers.py` — Input preprocessing dispatch

### Gradio App Modules (`src/depth_anything_3/app/modules/`)
- `model_inference.py` — Inference pipeline wired to the UI
- `event_handlers.py` — User interaction callbacks
- `visualization.py` — Depth map and feature visualization rendering
- `ui_components.py` — Gradio component definitions
- `file_handlers.py` — File upload/download I/O

### Utilities (`src/depth_anything_3/utils/`)
- `export/` — Multi-format output exporters: COLMAP, NPZ, GLB, Gaussian Splatting (`.ply`)
- `geometry.py`, `ray_utils.py` — 3D geometry math
- `alignment.py`, `pose_align.py` — Camera pose alignment
- `read_write_model.py` — COLMAP model I/O
- `model_loading.py` — Checkpoint loading helpers

### Benchmarking (`src/depth_anything_3/bench/`)
Registry-based evaluation framework. Supported datasets: DTU, DTU64, ETH3D, ScanNet++, 7-Scenes, HiRoom. Dataset configs live in `src/depth_anything_3/bench/configs/`.

### Streaming Module (`da3_streaming/`)
A separate, self-contained module for video/streaming depth estimation with loop closure detection. Key components:
- `da3_streaming.py` — Main streaming pipeline
- `loop_utils/` — Loop closure detection and Sim3 pose refinement
- `fastloop/` — Fast loop closure solver
- `loop_utils/salad/` — SALAD loop closure framework (git submodule)

## Key Design Patterns

- **Registry pattern**: Models and benchmark datasets are registered via `src/depth_anything_3/registry.py` and `src/depth_anything_3/bench/registries.py`
- **Config-driven**: Model and dataset configurations use YAML files under `src/depth_anything_3/configs/` and `da3_streaming/configs/`
- **Source layout**: Package is under `src/` — import as `depth_anything_3`, not `src.depth_anything_3`
- **Multi-view vs monocular**: The model supports both monocular (single image) and multi-view (scene-level) depth estimation; the `reference_view_selector` and `cam_enc/cam_dec` are only active in multi-view mode
