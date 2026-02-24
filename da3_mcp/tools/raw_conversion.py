"""RAW image conversion and EXIF GPS extraction."""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"}


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    """Convert degrees/minutes/seconds tuple to decimal degrees."""
    degrees, minutes, seconds = dms
    # piexif returns rationals as (numerator, denominator) tuples
    if isinstance(degrees, tuple):
        degrees = degrees[0] / degrees[1]
    if isinstance(minutes, tuple):
        minutes = minutes[0] / minutes[1]
    if isinstance(seconds, tuple):
        seconds = seconds[0] / seconds[1]
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


def _extract_gps_from_image(image_path: Path) -> dict[str, float] | None:
    """Extract GPS coordinates from image EXIF data."""
    try:
        import piexif
        from PIL import Image

        img = Image.open(image_path)
        exif_bytes = img.info.get("exif", b"")
        if not exif_bytes:
            return None

        exif_dict = piexif.load(exif_bytes)
        gps = exif_dict.get("GPS", {})
        if not gps:
            return None

        # GPS IFD tags
        GPS_LATITUDE = 2
        GPS_LATITUDE_REF = 1
        GPS_LONGITUDE = 4
        GPS_LONGITUDE_REF = 3
        GPS_ALTITUDE = 6
        GPS_ALTITUDE_REF = 5

        if GPS_LATITUDE not in gps or GPS_LONGITUDE not in gps:
            return None

        lat = _dms_to_decimal(gps[GPS_LATITUDE], gps.get(GPS_LATITUDE_REF, b"N").decode())
        lon = _dms_to_decimal(gps[GPS_LONGITUDE], gps.get(GPS_LONGITUDE_REF, b"E").decode())

        alt = None
        if GPS_ALTITUDE in gps:
            alt_raw = gps[GPS_ALTITUDE]
            if isinstance(alt_raw, tuple):
                alt = alt_raw[0] / alt_raw[1]
            else:
                alt = float(alt_raw)
            alt_ref = gps.get(GPS_ALTITUDE_REF, 0)
            if alt_ref == 1:
                alt = -alt

        return {"lat": lat, "lon": lon, "alt": alt}

    except Exception as e:
        logger.warning(f"Could not extract GPS from {image_path}: {e}")
        return None


def _convert_single_raw(raw_path: Path, output_dir: Path) -> dict[str, Any]:
    """Convert a single RAW file to PNG and extract GPS."""
    import imageio
    import rawpy

    output_path = output_dir / (raw_path.stem + ".png")

    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=False,
        )

    imageio.imwrite(str(output_path), rgb)
    logger.info(f"Converted {raw_path.name} -> {output_path.name}")

    gps = _extract_gps_from_image(raw_path)

    return {
        "filename": raw_path.name,
        "output_path": str(output_path),
        "lat": gps["lat"] if gps else None,
        "lon": gps["lon"] if gps else None,
        "alt": gps["alt"] if gps else None,
    }


def convert_raw_images(raw_dir: str, output_dir: str) -> dict[str, Any]:
    """Convert RAW photos to PNG and extract EXIF GPS metadata.

    Args:
        raw_dir: Directory containing RAW image files.
        output_dir: Directory where PNG files will be saved.

    Returns:
        dict with keys:
            - png_dir: path to output directory
            - gps_json: path to saved GPS JSON file
            - images: list of {filename, output_path, lat, lon, alt}
            - count: number of images converted
    """
    raw_dir_path = Path(raw_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(
        f for f in raw_dir_path.iterdir() if f.suffix.lower() in RAW_EXTENSIONS
    )

    if not raw_files:
        raise ValueError(f"No RAW files found in {raw_dir}. "
                         f"Supported extensions: {RAW_EXTENSIONS}")

    logger.info(f"Found {len(raw_files)} RAW files in {raw_dir}")

    results = []
    for raw_file in raw_files:
        try:
            result = _convert_single_raw(raw_file, output_dir_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to convert {raw_file.name}: {e}")

    gps_json_path = output_dir_path / "gps_metadata.json"
    with open(gps_json_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Converted {len(results)}/{len(raw_files)} images. GPS saved to {gps_json_path}")

    return {
        "png_dir": str(output_dir_path),
        "gps_json": str(gps_json_path),
        "images": results,
        "count": len(results),
    }
