"""DA3 depth estimation wrapper using the Python API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".bmp"}


def estimate_depth(
    images_dir: str,
    export_dir: str,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run DA3 depth estimation on a directory of images.

    Uses the DA3 Python API directly (no subprocess) for reliability.
    Outputs NPZ depth maps (depth + confidence + extrinsics + intrinsics)
    and optionally a GLB mesh.

    Args:
        images_dir: Directory containing PNG/JPEG images.
        export_dir: Directory where results will be exported.
        model_name: HuggingFace model ID or local path. Defaults to
                    the metric-depth giant model.

    Returns:
        dict with keys:
            - npz_dir: path to directory containing NPZ files
            - glb_path: path to GLB mesh (may be None if not exported)
            - num_images: number of images processed
            - export_dir: export directory path
    """
    from depth_anything_3.api import DepthAnything3

    images_dir_path = Path(images_dir)
    export_dir_path = Path(export_dir)
    export_dir_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        f for f in images_dir_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(
            f"No supported images found in {images_dir}. "
            f"Supported extensions: {SUPPORTED_IMAGE_EXTENSIONS}"
        )

    logger.info(f"Loading DA3 model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name)

    logger.info(f"Running inference on {len(image_paths)} images -> {export_dir}")
    prediction = model.inference(
        image_paths,
        export_dir=str(export_dir_path),
        export_format="mini_npz-glb",
    )

    # Locate NPZ files
    npz_files = list(export_dir_path.rglob("*.npz"))
    npz_dir = str(npz_files[0].parent) if npz_files else str(export_dir_path)

    # Locate GLB file
    glb_files = list(export_dir_path.rglob("*.glb"))
    glb_path = str(glb_files[0]) if glb_files else None

    logger.info(f"Inference complete. {len(npz_files)} NPZ files, GLB: {glb_path}")

    return {
        "npz_dir": npz_dir,
        "glb_path": glb_path,
        "num_images": len(image_paths),
        "export_dir": str(export_dir_path),
    }
