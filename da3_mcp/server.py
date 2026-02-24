"""DA3 MCP Server — Forest Canopy Height Model tools.

Exposes 8 MCP tools via stdio transport for use with Claude Desktop
or any MCP-compatible client.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stderr)])
logger = logging.getLogger(__name__)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("da3-forest-chm")


# ---------------------------------------------------------------------------
# Tool 1: Convert RAW images to PNG
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_convert_raw(raw_dir: str, output_dir: str) -> dict[str, Any]:
    """Convert drone RAW photos to PNG and extract EXIF GPS metadata.

    Args:
        raw_dir: Directory containing RAW image files (.CR2, .CR3, .NEF,
                 .ARW, .DNG, .RAF, .ORF, .RW2, .PEF).
        output_dir: Directory where converted PNG files will be saved.

    Returns:
        dict with:
            - png_dir: path to the PNG output directory
            - gps_json: path to gps_metadata.json
            - images: list of {filename, output_path, lat, lon, alt}
            - count: number of images converted
    """
    from da3_mcp.tools.raw_conversion import convert_raw_images

    return convert_raw_images(raw_dir, output_dir)


# ---------------------------------------------------------------------------
# Tool 2: Run DA3 depth estimation
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_estimate_depth(
    images_dir: str,
    export_dir: str,
    model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE",
) -> dict[str, Any]:
    """Run DA3 multi-view depth estimation on a directory of images.

    Produces NPZ files containing metric depth maps, camera intrinsics,
    and extrinsics for each image, plus an optional GLB 3D mesh.

    Args:
        images_dir: Directory containing PNG/JPEG images.
        export_dir: Directory where NPZ and GLB results will be saved.
        model_name: HuggingFace model ID or local path.
                    Default: depth-anything/DA3NESTED-GIANT-LARGE

    Returns:
        dict with:
            - npz_dir: directory containing NPZ depth maps
            - glb_path: path to GLB mesh (or None)
            - num_images: number of images processed
            - export_dir: export directory path
    """
    from da3_mcp.tools.depth_estimation import estimate_depth

    return estimate_depth(images_dir, export_dir, model_name)


# ---------------------------------------------------------------------------
# Tool 3: Generate CHM GeoTIFF
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_generate_chm(
    npz_dir: str,
    output_path: str,
    resolution_m: float = 0.5,
) -> dict[str, Any]:
    """Generate a Canopy Height Model (CHM) GeoTIFF from DA3 NPZ depth maps.

    Pipeline: unproject depth maps → 3D point cloud → RANSAC ground plane
    filtering → rasterize max vegetation height → write georeferenced GeoTIFF.

    GPS georeferencing is applied automatically if a gps_metadata.json file
    is found in or near the npz_dir (produced by da3_convert_raw).

    Args:
        npz_dir: Directory containing DA3 NPZ output files.
        output_path: Path for the output CHM GeoTIFF (e.g., /results/chm.tif).
        resolution_m: Raster cell size in metres (default 0.5 m).

    Returns:
        dict with:
            - chm_path: path to output GeoTIFF
            - shape: [rows, cols] of the raster
            - resolution_m: cell resolution used
            - crs: coordinate reference system string
            - has_georef: whether GPS-based georeferencing was applied
            - max_height_m: maximum canopy height in the CHM
            - coverage_pct: percentage of grid cells with valid data
    """
    from da3_mcp.tools.chm_generation import generate_chm

    return generate_chm(npz_dir, output_path, resolution_m)


# ---------------------------------------------------------------------------
# Tool 4: Extract forest inventory metrics
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_extract_metrics(
    chm_path: str | None = None,
    npz_dir: str | None = None,
    min_height_m: float = 1.5,
    resolution_m: float = 0.5,
) -> dict[str, Any]:
    """Extract forest inventory statistics from a CHM GeoTIFF or NPZ depth maps.

    Computes standard LiDAR-equivalent metrics: mean/max/p95 canopy height,
    canopy cover percentage, area, and more.

    Args:
        chm_path: Path to an existing CHM GeoTIFF (preferred).
        npz_dir: Directory with DA3 NPZ files (used if chm_path is None;
                 generates a temporary CHM first).
        min_height_m: Minimum height (m) to classify a pixel as canopy
                      (default 1.5 m).
        resolution_m: Cell size (m) when generating CHM from npz_dir.

    Returns:
        dict with:
            - mean_canopy_height_m
            - max_canopy_height_m
            - p95_canopy_height_m
            - p75_canopy_height_m
            - canopy_cover_pct
            - mean_height_above_2m
            - area_ha
            - resolution_m
            - chm_path
    """
    from da3_mcp.tools.forest_metrics import extract_metrics

    return extract_metrics(chm_path, npz_dir, min_height_m, resolution_m)


# ---------------------------------------------------------------------------
# Tool 5: Full end-to-end pipeline
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_full_pipeline(
    raw_dir: str,
    output_dir: str,
    model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE",
    chm_resolution_m: float = 0.5,
    min_canopy_height_m: float = 1.5,
) -> dict[str, Any]:
    """Run the complete RAW → CHM → forest metrics pipeline.

    Sequentially executes:
        1. da3_convert_raw    — RAW photos → PNG + GPS metadata
        2. da3_estimate_depth — PNG images → DA3 NPZ depth maps
        3. da3_generate_chm   — NPZ depth maps → CHM GeoTIFF
        4. da3_extract_metrics — CHM → forest inventory metrics

    Args:
        raw_dir: Directory containing drone RAW image files.
        output_dir: Root directory for all outputs.
        model_name: DA3 model to use for depth estimation.
        chm_resolution_m: CHM raster cell size in metres (default 0.5 m).
        min_canopy_height_m: Minimum height threshold for canopy
                              classification (default 1.5 m).

    Returns:
        dict with:
            - png_dir: converted images directory
            - npz_dir: depth map directory
            - chm_path: CHM GeoTIFF path
            - metrics: forest inventory statistics dict
            - summary: human-readable summary string
    """
    from da3_mcp.tools.chm_generation import generate_chm
    from da3_mcp.tools.depth_estimation import estimate_depth
    from da3_mcp.tools.forest_metrics import extract_metrics
    from da3_mcp.tools.raw_conversion import convert_raw_images

    output_root = Path(output_dir)
    png_dir = str(output_root / "01_png")
    depth_dir = str(output_root / "02_depth")
    chm_path = str(output_root / "03_chm" / "canopy_height_model.tif")

    logger.info("=== DA3 Full Pipeline ===")

    # Step 1
    logger.info("Step 1/4: Converting RAW images...")
    raw_result = convert_raw_images(raw_dir, png_dir)

    # Step 2
    logger.info("Step 2/4: Running DA3 depth estimation...")
    depth_result = estimate_depth(png_dir, depth_dir, model_name)

    # Step 3
    logger.info("Step 3/4: Generating CHM GeoTIFF...")
    chm_result = generate_chm(depth_result["npz_dir"], chm_path, chm_resolution_m)

    # Step 4
    logger.info("Step 4/4: Extracting forest metrics...")
    metrics = extract_metrics(
        chm_path=chm_result["chm_path"],
        min_height_m=min_canopy_height_m,
        resolution_m=chm_resolution_m,
    )

    # Save metrics JSON alongside CHM
    metrics_json_path = Path(chm_result["chm_path"]).parent / "forest_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    summary = (
        f"Pipeline complete. "
        f"{raw_result['count']} images processed. "
        f"CHM: {chm_result['shape'][0]}x{chm_result['shape'][1]} px "
        f"@ {chm_resolution_m} m/px. "
        f"Mean canopy height: {metrics['mean_canopy_height_m']:.1f} m, "
        f"cover: {metrics['canopy_cover_pct']:.1f}%, "
        f"area: {metrics['area_ha']:.2f} ha."
    )
    logger.info(summary)

    return {
        "png_dir": png_dir,
        "npz_dir": depth_result["npz_dir"],
        "chm_path": chm_result["chm_path"],
        "metrics_json": str(metrics_json_path),
        "metrics": metrics,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Tool 6: List results in a directory
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_list_results(directory: str) -> dict[str, Any]:
    """List DA3 output files in a directory.

    Args:
        directory: Path to a directory to inspect.

    Returns:
        dict with categorised file listings:
            - tif_files: GeoTIFF rasters
            - npz_files: depth map arrays
            - json_files: metadata / metrics JSON
            - glb_files: 3D mesh files
            - png_files: converted images
            - all_files: flat list of all files
            - total_size_mb: total size in megabytes
    """
    root = Path(directory)
    if not root.exists():
        return {"error": f"Directory not found: {directory}"}

    all_files = sorted(f for f in root.rglob("*") if f.is_file())

    def _collect(exts: set[str]) -> list[str]:
        return [str(f) for f in all_files if f.suffix.lower() in exts]

    total_bytes = sum(f.stat().st_size for f in all_files)

    return {
        "tif_files": _collect({".tif", ".tiff"}),
        "npz_files": _collect({".npz"}),
        "json_files": _collect({".json"}),
        "glb_files": _collect({".glb"}),
        "png_files": _collect({".png"}),
        "all_files": [str(f) for f in all_files],
        "total_size_mb": round(total_bytes / 1024 / 1024, 2),
    }


# ---------------------------------------------------------------------------
# Tool 7: Start DA3 backend server
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_start_backend(
    model_dir: str = "",
    port: int = 7860,
) -> dict[str, Any]:
    """Start the DA3 FastAPI backend server in a background process.

    Args:
        model_dir: Optional path to local model weights directory.
                   Leave empty to use HuggingFace auto-download.
        port: Port number for the backend (default 7860).

    Returns:
        dict with:
            - url: backend URL
            - pid: process ID
            - status: 'started' or 'already_running'
    """
    import subprocess
    import time

    import requests

    url = f"http://127.0.0.1:{port}"

    # Check if already running
    try:
        resp = requests.get(f"{url}/health", timeout=2)
        if resp.status_code == 200:
            return {"url": url, "status": "already_running", "pid": None}
    except Exception:
        pass

    cmd = [sys.executable, "-m", "depth_anything_3.app.gradio_app", "--port", str(port)]
    if model_dir:
        cmd += ["--model-dir", model_dir]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Brief wait to allow server startup
    time.sleep(3)

    return {"url": url, "pid": proc.pid, "status": "started"}


# ---------------------------------------------------------------------------
# Tool 8: Check backend status
# ---------------------------------------------------------------------------


@mcp.tool()
def da3_backend_status(url: str = "http://127.0.0.1:7860") -> dict[str, Any]:
    """Check the status of a running DA3 backend server.

    Args:
        url: Base URL of the DA3 backend (default http://127.0.0.1:7860).

    Returns:
        dict with:
            - url: the URL checked
            - reachable: True if server responded
            - status_code: HTTP status code (or None)
            - error: error message if unreachable
    """
    import requests

    try:
        resp = requests.get(f"{url}/health", timeout=5)
        return {
            "url": url,
            "reachable": True,
            "status_code": resp.status_code,
            "error": None,
        }
    except requests.exceptions.ConnectionError:
        return {
            "url": url,
            "reachable": False,
            "status_code": None,
            "error": "Connection refused — is the backend running?",
        }
    except Exception as e:
        return {
            "url": url,
            "reachable": False,
            "status_code": None,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
