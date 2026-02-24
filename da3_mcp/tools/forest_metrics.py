"""Forest inventory metrics extraction from CHM GeoTIFF or NPZ depth maps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default minimum height to be considered canopy (m)
MIN_CANOPY_HEIGHT_M = 1.5


def _load_chm_array(chm_path: Path) -> tuple[np.ndarray, float]:
    """Load CHM raster and return (array, resolution_m)."""
    import rasterio

    with rasterio.open(str(chm_path)) as src:
        chm = src.read(1).astype(np.float32)
        nodata = src.nodata
        # Try to read pixel size from transform
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        resolution_m = (res_x + res_y) / 2.0

    if nodata is not None:
        chm = np.where(chm == nodata, np.nan, chm)

    return chm, resolution_m


def extract_metrics(
    chm_path: str | None = None,
    npz_dir: str | None = None,
    min_height_m: float = MIN_CANOPY_HEIGHT_M,
    resolution_m: float = 0.5,
) -> dict[str, Any]:
    """Compute forest inventory statistics from a CHM GeoTIFF.

    If `chm_path` is provided it is used directly. If only `npz_dir` is
    provided, `generate_chm` is called first (writing to a temp file).

    Args:
        chm_path: Path to an existing CHM GeoTIFF.
        npz_dir: Directory containing DA3 NPZ files (used if chm_path is None).
        min_height_m: Minimum height (m) to classify a pixel as canopy.
        resolution_m: Cell size (m) used when generating CHM from npz_dir.

    Returns:
        dict of forest metrics.
    """
    if chm_path is None and npz_dir is None:
        raise ValueError("Provide either chm_path or npz_dir.")

    if chm_path is None:
        import tempfile

        from da3_mcp.tools.chm_generation import generate_chm

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name

        logger.info(f"No CHM provided — generating from NPZ dir: {npz_dir}")
        result = generate_chm(npz_dir, tmp_path, resolution_m=resolution_m)
        chm_path = result["chm_path"]
        resolution_m = result["resolution_m"]

    chm_path_obj = Path(chm_path)
    chm, resolution_m = _load_chm_array(chm_path_obj)

    # All valid pixels
    valid = ~np.isnan(chm) & (chm >= 0)
    canopy = chm[valid & (chm >= min_height_m)]
    above_2m = chm[valid & (chm >= 2.0)]

    total_pixels = chm.size
    valid_pixels = int(np.sum(valid))
    canopy_pixels = len(canopy)

    area_ha = total_pixels * resolution_m ** 2 / 10_000

    metrics: dict[str, Any] = {
        "mean_canopy_height_m": float(np.nanmean(canopy)) if len(canopy) > 0 else 0.0,
        "max_canopy_height_m": float(np.nanmax(chm[valid])) if valid_pixels > 0 else 0.0,
        "p95_canopy_height_m": float(np.nanpercentile(canopy, 95)) if len(canopy) > 0 else 0.0,
        "p75_canopy_height_m": float(np.nanpercentile(canopy, 75)) if len(canopy) > 0 else 0.0,
        "canopy_cover_pct": float(canopy_pixels / total_pixels * 100),
        "mean_height_above_2m": float(np.nanmean(above_2m)) if len(above_2m) > 0 else 0.0,
        "area_ha": float(area_ha),
        "resolution_m": float(resolution_m),
        "chm_path": str(chm_path_obj),
        "min_height_threshold_m": float(min_height_m),
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "canopy_pixels": canopy_pixels,
    }

    logger.info(
        f"Forest metrics — mean height: {metrics['mean_canopy_height_m']:.2f} m, "
        f"cover: {metrics['canopy_cover_pct']:.1f}%, "
        f"area: {metrics['area_ha']:.3f} ha"
    )

    return metrics
