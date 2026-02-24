# Product Requirements Document
## Depth Anything 3 (DA3)

**Version:** 1.0
**Date:** February 2026
**Status:** Active

---

## 1. Overview

Depth Anything 3 (DA3) is a visual geometry foundation model that predicts spatially consistent depth and camera poses from arbitrary visual inputs — with or without known camera poses. It unifies monocular depth, multi-view depth, pose estimation, and 3D Gaussian Splatting under a single depth-ray representation using a plain transformer backbone (DINOv2).

**Paper:** arXiv:2511.10647
**Project page:** https://depth-anything-3.github.io
**HuggingFace demo:** https://huggingface.co/spaces/depth-anything/Depth-Anything-3

---

## 2. Goals

| Goal | Description |
|------|-------------|
| G1 | Single model handles monocular, multi-view, and pose-conditioned depth |
| G2 | Minimal architecture: plain ViT backbone + depth-ray representation |
| G3 | Outperform DA2 (monocular) and VGGT (multi-view / pose) |
| G4 | Support Gaussian Splatting for novel view synthesis |
| G5 | Provide researcher- and practitioner-friendly APIs and tooling |

---

## 3. Model Requirements

### 3.1 Model Series

| Series | Models | Key Capabilities |
|--------|--------|-----------------|
| Main (any-view) | DA3-Giant, DA3-Large, DA3-Base, DA3-Small | Relative depth, pose est., pose-conditioned depth |
| Giant variants | DA3-Giant, DA3-Giant-1.1 | + 3D Gaussian Splatting |
| Metric | DA3Metric-Large | Metric depth with sky segmentation |
| Monocular | DA3Mono-Large | High-accuracy relative depth |
| Nested | DA3Nested-Giant-Large, DA3Nested-Giant-Large-1.1 | All capabilities + metric scale |

Models with the `-1.1` suffix are preferred; they fix a training bug and offer significantly better performance on street scenes.

### 3.2 Input Specifications

- Images: file paths, NumPy arrays, or PIL Images
- Extrinsics: (N, 4, 4) float32 — world-to-camera (OpenCV/COLMAP convention)
- Intrinsics: (N, 3, 3) float32 — camera intrinsics
- Processing resolution: default 504px (configurable)

### 3.3 Output Specifications

- Depth maps: (N, H, W) float32
- Confidence maps: (N, H, W) float32
- Extrinsics: (N, 3, 4) float32
- Intrinsics: (N, 3, 3) float32
- Preprocessed images: (N, H, W, 3) uint8
- Auxiliary: intermediate features, Gaussian Splats data

---

## 4. Interface Requirements

### 4.1 Python API

- Class: `DepthAnything3` in `depth_anything_3.api`
- Primary method: `inference()` — accepts images, optional poses, returns `Prediction`
- Support for HuggingFace `from_pretrained()` model loading
- Export formats: `mini_npz`, `npz`, `glb`, `gs_ply`, `gs_video`, `feat_vis`, `depth_vis`

### 4.2 Command-Line Interface

- Entry point: `da3`
- Commands: `auto`, `image`, `images`, `video`, `colmap`, `backend`, `gradio`, `gallery`
- Backend service: keeps model in GPU memory; supports REST inference API
- Auto-mode: detects input type (image / directory / video / COLMAP) automatically

### 4.3 Gradio Web Application

- Interactive depth estimation and visualisation
- Gallery browser for reviewing results
- Shareable public links via Gradio share mode
- Pre-caching of example scenes

---

## 5. Export Format Requirements

| Format | Contents | Required Params |
|--------|----------|-----------------|
| `mini_npz` | depth, conf, exts, ixts | — |
| `npz` | All of mini_npz + images | — |
| `glb` | 3D point cloud + camera wireframes | conf_thresh_percentile, num_max_points, show_cameras |
| `gs_ply` | 3DGS PLY (SuperSplat-compatible) | `infer_gs=True`, Giant model |
| `gs_video` | Rasterised 3DGS video | `infer_gs=True`, Giant model |
| `feat_vis` | PCA feature visualisation video | export_feat_layers |
| `depth_vis` | Colour-coded depth maps | — |

Multiple formats can be combined: `"mini_npz-glb-depth_vis"`.

---

## 6. Benchmark Requirements

### 6.1 Supported Datasets

| Dataset | Task | Primary Metrics |
|---------|------|-----------------|
| ETH3D | Pose + Recon | AUC@3°, AUC@30°, F-score, Overall |
| ScanNet++ | Pose + Recon | AUC@3°, F-score |
| 7Scenes | Pose + Recon | AUC@3°, F-score |
| HiRoom | Pose + Recon | AUC@3°, F-score |
| DTU-49 | Recon only | Overall (mm) |
| DTU-64 | Pose only | AUC@3°, AUC@30° |

### 6.2 Evaluation Modes

- `pose` — camera pose estimation accuracy
- `recon_unposed` — 3D reconstruction using predicted poses
- `recon_posed` — 3D reconstruction using ground-truth poses

### 6.3 Performance Targets (DA3-Giant)

| Metric | Target | Achieved |
|--------|--------|---------|
| Pose AUC@3° (avg 5 datasets) | > 0.65 | 0.6705 |
| Pose AUC@30° (avg) | > 0.93 | 0.9436 |
| Recon F-score unposed (avg 4 datasets) | > 0.70 | 0.7345 |
| Recon F-score posed (avg 4 datasets) | > 0.75 | 0.7978 |

---

## 7. Streaming Module Requirements

The `da3_streaming` module must support:

- Ultra-long video sequences (>1000 frames)
- Sliding-window streaming inference
- GPU memory usage < 12 GB
- Loop closure detection and Sim3 pose refinement
- Separate installable package

---

## 8. Non-Functional Requirements

| Requirement | Specification |
|-------------|---------------|
| Default inference resolution | 504px (configurable) |
| Backend GPU persistence | Model stays loaded between requests |
| Multi-GPU support | Auto-distributed across available GPUs |
| Source layout | `src/depth_anything_3/` — import as `depth_anything_3` |
| Config system | YAML-driven; registry pattern for models and datasets |
| Python version | >= 3.10 for Gradio app |

---

## 9. Documentation

### 9.1 Source Documentation (docs/)

| File | Contents |
|------|----------|
| `docs/API.md` | Full Python API reference with examples and parameter table |
| `docs/CLI.md` | All CLI commands, flags, and usage examples |
| `docs/BENCHMARK.md` | Dataset download, evaluation pipeline, metrics, expected results |
| `docs/funcs/ref_view_strategy.md` | Reference view selection strategies and recommendations |

### 9.2 Compiled Manual

A comprehensive LaTeX manual consolidating all documentation has been created:

| File | Description |
|------|-------------|
| `docs/manual.tex` | LaTeX source — 22-section manual covering API, CLI, benchmarks, and reference view selection |
| `docs/manual.pdf` | Compiled PDF (22 pages) — print-ready user manual |

**Manual Contents:**

1. Introduction and model capabilities
2. Model Zoo (all available checkpoints with feature matrix)
3. Installation (core package, optional components, streaming module)
4. Python API (class reference, all parameters, export formats, return types)
5. CLI Reference (all 8 commands with full parameter tables and examples)
6. Reference View Selection (4 strategies with algorithm details and recommendations)
7. Benchmark Evaluation (dataset setup, evaluation modes, metrics, expected results)
8. Streaming Module
9. FAQ, Citation, and License

**Compilation:**
```bash
cd docs/
pdflatex manual.tex && pdflatex manual.tex   # second pass for TOC
```

Requires TeX Live 2023 or equivalent. The manual is generated from the source docs
in `docs/` and should be recompiled whenever those docs are updated.

---

## 10. Licensing

| Model Group | License |
|-------------|---------|
| DA3-Giant, DA3-Large, DA3Nested-*, DA3Metric-Large | CC BY-NC 4.0 |
| DA3-Base, DA3-Small, DA3Mono-Large | Apache 2.0 |

---

## 11. Citation

```
@article{depthanything3,
  title   = {Depth Anything 3: Recovering the visual space from any views},
  author  = {Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen
             and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal = {arXiv preprint arXiv:2511.10647},
  year    = {2025}
}
```
