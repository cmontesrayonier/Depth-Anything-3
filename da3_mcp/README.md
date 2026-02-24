# DA3 MCP Server — Forest Canopy Height Model

An MCP (Model Context Protocol) server that wraps the Depth-Anything-3 (DA3) pipeline
to produce Canopy Height Models (CHM) and forest inventory metrics from drone RAW photos.

## Overview

```
Drone RAW photos  →  PNG conversion  →  DA3 depth estimation  →  3D point cloud
                                                                        ↓
                              Forest metrics  ←  CHM GeoTIFF  ←  Ground filtering
```

## Installation

```bash
# Install MCP server dependencies (inside the DA3 Python environment)
pip install -r da3_mcp/requirements.txt
```

## Usage

### With Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "da3-forest-chm": {
      "command": "python",
      "args": ["-m", "da3_mcp.server"],
      "cwd": "/path/to/Depth-Anything-3"
    }
  }
}
```

Restart Claude Desktop — the 8 DA3 tools will appear in the MCP panel.

### From the command line

```bash
# Start server (blocks on stdio — for testing with an MCP client)
python -m da3_mcp.server
```

## Available Tools

| Tool | Description |
|------|-------------|
| `da3_convert_raw` | Convert RAW photos to PNG; extract EXIF GPS |
| `da3_estimate_depth` | Run DA3 depth estimation; export NPZ + GLB |
| `da3_generate_chm` | Build CHM GeoTIFF from NPZ depth maps |
| `da3_extract_metrics` | Compute forest inventory statistics |
| `da3_full_pipeline` | End-to-end: RAW → CHM → metrics |
| `da3_list_results` | List output files in a directory |
| `da3_start_backend` | Start the DA3 FastAPI backend |
| `da3_backend_status` | Check backend health |

## End-to-End Example

In a Claude Desktop conversation with the MCP server connected:

> "Run `da3_full_pipeline` on my RAW photos in `/data/flight_01/raw`,
> save outputs to `/data/flight_01/results`."

This will:
1. Convert RAW → PNG and extract GPS coordinates
2. Run DA3 depth estimation (downloads model weights automatically on first run)
3. Generate a georeferenced CHM GeoTIFF
4. Compute and return forest metrics JSON

## Output Files

After a successful pipeline run:

```
output_dir/
├── 01_png/
│   ├── IMG_0001.png
│   ├── ...
│   └── gps_metadata.json          # GPS coords per image
├── 02_depth/
│   ├── frame_0000.npz             # depth + extrinsics + intrinsics
│   ├── ...
│   └── scene.glb                  # 3D mesh (optional)
└── 03_chm/
    ├── canopy_height_model.tif    # GeoTIFF CHM (UTM projected)
    └── forest_metrics.json        # Inventory statistics
```

## Forest Metrics

The metrics JSON contains:

| Metric | Description |
|--------|-------------|
| `mean_canopy_height_m` | Mean height of pixels above min threshold |
| `max_canopy_height_m` | Maximum canopy height |
| `p95_canopy_height_m` | 95th percentile canopy height |
| `p75_canopy_height_m` | 75th percentile canopy height |
| `canopy_cover_pct` | % of total area classified as canopy |
| `mean_height_above_2m` | Mean height for pixels > 2 m |
| `area_ha` | Total surveyed area in hectares |

## Supported RAW Formats

CR2, CR3, NEF, ARW, DNG, RAF, ORF, RW2, PEF (via `rawpy`).

## Requirements

- Python 3.9–3.13
- DA3 installed (`pip install -e .` from repo root)
- Dependencies in `da3_mcp/requirements.txt`
- GPU recommended for depth estimation (CUDA or MPS)

## Architecture Notes

- The MCP server runs in-process with the DA3 Python API (no subprocess calls for inference)
- CHM georeferencing uses EXIF GPS bounding box projected to UTM via `pyproj`
- Ground filtering uses `scikit-learn` RANSAC on the lowest 30% of 3D points
- Rasterization uses `scipy.stats.binned_statistic_2d` (max height per cell)
