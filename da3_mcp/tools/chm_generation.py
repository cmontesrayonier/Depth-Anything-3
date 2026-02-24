"""CHM (Canopy Height Model) generation from DA3 depth maps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _load_npz_frames(npz_dir: Path) -> list[dict]:
    """Load all NPZ frames from a directory.

    Each NPZ is expected to contain:
        - depth: (H, W) metric depth in metres
        - extrinsics: (4, 4) camera-to-world transform (or world-to-camera)
        - intrinsics: (3, 3) or (4,) [fx, fy, cx, cy]
    """
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    frames = []
    for npz_path in npz_files:
        data = np.load(str(npz_path), allow_pickle=True)
        frame = {"path": npz_path}

        # Depth
        if "depth" in data:
            frame["depth"] = data["depth"]
        else:
            raise KeyError(f"'depth' key missing in {npz_path}")

        # Extrinsics — DA3 mini_npz stores as 'exts' (world-to-camera 4x4)
        for key in ("exts", "extrinsics", "ext"):
            if key in data:
                frame["extrinsics"] = data[key]
                break
        else:
            raise KeyError(f"Extrinsics key not found in {npz_path}. Keys: {list(data.keys())}")

        # Intrinsics — DA3 mini_npz stores as 'ixts' (3x3 K matrix)
        for key in ("ixts", "intrinsics", "ixt"):
            if key in data:
                frame["intrinsics"] = data[key]
                break
        else:
            raise KeyError(f"Intrinsics key not found in {npz_path}. Keys: {list(data.keys())}")

        frames.append(frame)

    logger.info(f"Loaded {len(frames)} NPZ frames from {npz_dir}")
    return frames


def _unproject_to_world(frame: dict) -> np.ndarray:
    """Unproject a depth frame to 3D world-space points.

    Returns Nx3 array of (X, Y, Z) world coordinates.
    """
    depth = frame["depth"]  # (H, W)
    ext = frame["extrinsics"]  # (4,4) world-to-camera
    ixt = frame["intrinsics"]  # (3,3) or similar

    H, W = depth.shape

    # Parse intrinsics
    if ixt.ndim == 2 and ixt.shape == (3, 3):
        fx, fy = ixt[0, 0], ixt[1, 1]
        cx, cy = ixt[0, 2], ixt[1, 2]
    elif ixt.ndim == 1 and len(ixt) >= 4:
        fx, fy, cx, cy = ixt[0], ixt[1], ixt[2], ixt[3]
    else:
        raise ValueError(f"Unexpected intrinsics shape: {ixt.shape}")

    # Pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = depth > 0

    u_v = u[valid].astype(np.float64)
    v_v = v[valid].astype(np.float64)
    d_v = depth[valid].astype(np.float64)

    # Back-project to camera space
    Xc = (u_v - cx) * d_v / fx
    Yc = (v_v - cy) * d_v / fy
    Zc = d_v

    cam_pts = np.stack([Xc, Yc, Zc, np.ones_like(Zc)], axis=1)  # (N,4)

    # Extrinsics: world-to-camera → invert to get camera-to-world
    c2w = np.linalg.inv(ext.astype(np.float64))
    world_pts = (c2w @ cam_pts.T).T[:, :3]  # (N,3)

    return world_pts


def _fit_ground_plane(points: np.ndarray, max_trials: int = 500) -> tuple[np.ndarray, float]:
    """Fit a ground plane using RANSAC on the lowest-Z points.

    Returns (plane_normal [3,], plane_d) where plane eq is n·x + d = 0.
    """
    from sklearn.linear_model import RANSACRegressor

    # Use the bottom 30% of points by Z for ground fitting
    z_threshold = np.percentile(points[:, 2], 30)
    low_pts = points[points[:, 2] <= z_threshold]

    if len(low_pts) < 10:
        low_pts = points

    X = low_pts[:, :2]  # (N,2) — X,Y
    y = low_pts[:, 2]   # (N,) — Z

    ransac = RANSACRegressor(max_trials=max_trials, residual_threshold=0.3, random_state=42)
    ransac.fit(X, y)

    # Plane: Z = a*X + b*Y + c  →  normal = [-a, -b, 1], d = -c
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    d = -c

    logger.info(f"Ground plane fitted (RANSAC inlier ratio: "
                f"{ransac.inlier_mask_.mean():.2f})")
    return normal, d, ransac.estimator_


def _height_above_ground(points: np.ndarray, ground_estimator) -> np.ndarray:
    """Compute signed height of each point above the fitted ground plane."""
    xy = points[:, :2]
    z_ground = ground_estimator.predict(xy)
    return points[:, 2] - z_ground


def _load_gps_bounds(npz_dir: Path) -> dict | None:
    """Try to load GPS bounds from a gps_metadata.json in a parent directory."""
    for parent in [npz_dir, npz_dir.parent, npz_dir.parent.parent]:
        gps_json = parent / "gps_metadata.json"
        if gps_json.exists():
            import json
            with open(gps_json) as f:
                data = json.load(f)
            lats = [r["lat"] for r in data if r.get("lat") is not None]
            lons = [r["lon"] for r in data if r.get("lon") is not None]
            if lats and lons:
                return {
                    "lat_min": min(lats), "lat_max": max(lats),
                    "lon_min": min(lons), "lon_max": max(lons),
                    "lat_center": sum(lats) / len(lats),
                    "lon_center": sum(lons) / len(lons),
                }
    return None


def _rasterize_chm(
    xy: np.ndarray,
    heights: np.ndarray,
    resolution_m: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Bin heights into a 2D grid, keeping the maximum value per cell."""
    from scipy.stats import binned_statistic_2d

    n_x = max(1, int(np.ceil((x_max - x_min) / resolution_m)))
    n_y = max(1, int(np.ceil((y_max - y_min) / resolution_m)))

    result = binned_statistic_2d(
        xy[:, 0], xy[:, 1], heights,
        statistic="max",
        bins=[n_x, n_y],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    chm = result.statistic.T  # (n_y, n_x), origin at bottom-left
    chm = np.where(np.isnan(chm), np.nan, chm)
    chm = np.flipud(chm)  # flip so row 0 = north (top)

    return chm.astype(np.float32)


def generate_chm(
    npz_dir: str,
    output_path: str,
    resolution_m: float = 0.5,
) -> dict[str, Any]:
    """Generate a Canopy Height Model GeoTIFF from DA3 NPZ depth maps.

    Pipeline:
        1. Unproject depth maps to 3D world-space point cloud
        2. Fit ground plane (RANSAC) and compute above-ground heights
        3. Rasterize max height per grid cell
        4. Georeference using GPS bounding box and write GeoTIFF

    Args:
        npz_dir: Directory containing DA3 NPZ output files.
        output_path: Path for the output GeoTIFF file (e.g., chm.tif).
        resolution_m: Grid cell size in metres (default 0.5 m).

    Returns:
        dict with keys:
            - chm_path: path to output GeoTIFF
            - shape: (rows, cols) of the raster
            - resolution_m: cell resolution used
            - crs: coordinate reference system string
            - has_georef: whether GPS data was used for georeferencing
    """
    import rasterio
    from rasterio.transform import from_bounds

    npz_dir_path = Path(npz_dir)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load and unproject all frames ---
    frames = _load_npz_frames(npz_dir_path)

    all_points = []
    for frame in frames:
        try:
            pts = _unproject_to_world(frame)
            all_points.append(pts)
        except Exception as e:
            logger.warning(f"Skipping frame {frame['path'].name}: {e}")

    if not all_points:
        raise RuntimeError("No points could be unprojected from NPZ frames.")

    points = np.concatenate(all_points, axis=0)
    logger.info(f"Total points: {len(points):,}")

    # --- Step 2: Ground filtering ---
    normal, d, ground_estimator = _fit_ground_plane(points)
    heights = _height_above_ground(points, ground_estimator)

    # Keep only vegetation points (above 0.2 m, below 80 m)
    veg_mask = (heights > 0.2) & (heights < 80.0)
    veg_pts = points[veg_mask]
    veg_h = heights[veg_mask]

    if len(veg_pts) == 0:
        raise RuntimeError("No vegetation points found above ground plane.")

    logger.info(f"Vegetation points: {len(veg_pts):,} "
                f"({100 * len(veg_pts) / len(points):.1f}% of total)")

    # --- Step 3: Rasterize in scene XY space ---
    x_min, x_max = veg_pts[:, 0].min(), veg_pts[:, 0].max()
    y_min, y_max = veg_pts[:, 1].min(), veg_pts[:, 1].max()

    chm = _rasterize_chm(veg_pts[:, :2], veg_h, resolution_m, x_min, x_max, y_min, y_max)
    logger.info(f"CHM shape: {chm.shape}, max height: {np.nanmax(chm):.2f} m")

    # --- Step 4: Georeference and write GeoTIFF ---
    gps = _load_gps_bounds(npz_dir_path)
    has_georef = gps is not None

    if has_georef:
        try:
            from pyproj import Transformer

            # Project GPS bounding box to UTM
            lat_c = gps["lat_center"]
            lon_c = gps["lon_center"]
            utm_zone = int((lon_c + 180) / 6) + 1
            hemisphere = "north" if lat_c >= 0 else "south"
            utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"

            transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)

            x_sw, y_sw = transformer.transform(gps["lon_min"], gps["lat_min"])
            x_ne, y_ne = transformer.transform(gps["lon_max"], gps["lat_max"])

            transform = from_bounds(x_sw, y_sw, x_ne, y_ne, chm.shape[1], chm.shape[0])
            crs = utm_crs
            logger.info(f"Georeferenced to UTM zone {utm_zone}{hemisphere[0].upper()}")

        except Exception as e:
            logger.warning(f"Georeferencing failed ({e}), falling back to scene coordinates.")
            has_georef = False

    if not has_georef:
        from rasterio.transform import from_origin
        transform = from_origin(x_min, y_max, resolution_m, resolution_m)
        crs = "EPSG:4978"  # WGS84 ECEF as fallback placeholder
        logger.warning("No GPS data found — CHM will not be geographically referenced.")

    with rasterio.open(
        str(output_path_obj),
        "w",
        driver="GTiff",
        height=chm.shape[0],
        width=chm.shape[1],
        count=1,
        dtype=chm.dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan,
        compress="lzw",
    ) as dst:
        dst.write(chm, 1)

    logger.info(f"CHM GeoTIFF saved: {output_path_obj}")

    return {
        "chm_path": str(output_path_obj),
        "shape": list(chm.shape),
        "resolution_m": resolution_m,
        "crs": crs,
        "has_georef": has_georef,
        "max_height_m": float(np.nanmax(chm)),
        "coverage_pct": float(np.sum(~np.isnan(chm)) / chm.size * 100),
    }
