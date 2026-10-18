"""Geospatial preprocessing pipeline for LISS-IV multi-spectral imagery.

Validates single-band rasters, stacks them into a unified multi-band GeoTIFF,
generates a spatial validity mask for the collar/background, and exports a
complete processing manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window


EXPECTED_BAND_FILENAMES = ("BAND2.tif", "BAND3.tif", "BAND4.tif")
BAND_ORDER_NAMES = ("BAND2", "BAND3", "BAND4")


class PreprocessingError(Exception):
    """Raised when validation or preprocessing fails."""


def validate_inputs(
    band_paths: Sequence[Union[str, Path]],
) -> Dict[str, Any]:
    """Validate existence, format, and spatial alignment of input single-band GeoTIFFs.

    Args:
        band_paths: Sequence of file paths for BAND2, BAND3, and BAND4 in exact order.

    Returns:
        Dictionary containing verified dimensions, CRS, transform, resolution,
        and data type.

    Raises:
        PreprocessingError: If files are missing, cannot be opened, or differ in
            dimensions, CRS, transform, resolution, or data type.
    """
    if len(band_paths) != 3:
        raise PreprocessingError(
            f"Expected exactly 3 band paths, received {len(band_paths)}: {band_paths}"
        )

    resolved_paths: List[Path] = []
    for p in band_paths:
        path_obj = Path(p)
        if not path_obj.is_file():
            raise PreprocessingError(f"Required source band file does not exist: {path_obj}")
        resolved_paths.append(path_obj)

    ref_meta: Optional[Dict[str, Any]] = None

    for idx, path_obj in enumerate(resolved_paths):
        try:
            with rasterio.open(path_obj) as src:
                if src.count != 1:
                    raise PreprocessingError(
                        f"Expected single-band raster in {path_obj.name}, found count={src.count}"
                    )

                meta = {
                    "width": src.width,
                    "height": src.height,
                    "crs": src.crs,
                    "crs_wkt": src.crs.to_wkt() if src.crs else None,
                    "transform": src.transform,
                    "res": src.res,
                    "dtype": src.dtypes[0],
                }

                if ref_meta is None:
                    ref_meta = meta
                else:
                    if meta["width"] != ref_meta["width"] or meta["height"] != ref_meta["height"]:
                        raise PreprocessingError(
                            f"Dimension mismatch in {path_obj.name}: ({meta['width']}x{meta['height']}) "
                            f"!= reference ({ref_meta['width']}x{ref_meta['height']})"
                        )
                    if meta["crs"] != ref_meta["crs"]:
                        raise PreprocessingError(
                            f"CRS mismatch in {path_obj.name}: {meta['crs']} != reference {ref_meta['crs']}"
                        )
                    if not np.allclose(
                        np.array(meta["transform"]), np.array(ref_meta["transform"]), atol=1e-5
                    ):
                        raise PreprocessingError(
                            f"Transform mismatch in {path_obj.name}: {meta['transform']} "
                            f"!= reference {ref_meta['transform']}"
                        )
                    if not np.allclose(np.array(meta["res"]), np.array(ref_meta["res"]), atol=1e-5):
                        raise PreprocessingError(
                            f"Resolution mismatch in {path_obj.name}: {meta['res']} != reference {ref_meta['res']}"
                        )
                    if meta["dtype"] != ref_meta["dtype"]:
                        raise PreprocessingError(
                            f"Data type mismatch in {path_obj.name}: {meta['dtype']} != reference {ref_meta['dtype']}"
                        )
        except rasterio.errors.RasterioIOError as err:
            raise PreprocessingError(f"Failed to open raster {path_obj}: {err}") from err

    assert ref_meta is not None
    return {
        "width": ref_meta["width"],
        "height": ref_meta["height"],
        "crs": ref_meta["crs"],
        "crs_wkt": ref_meta["crs_wkt"],
        "transform": ref_meta["transform"],
        "res": ref_meta["res"],
        "dtype": ref_meta["dtype"],
        "paths": [str(p) for p in resolved_paths],
    }


def compute_or_lookup_hashes(
    band_paths: Sequence[Path],
    known_manifest_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Retrieve SHA-256 hashes from an existing valid manifest or compute them directly.

    Args:
        band_paths: List of band file paths.
        known_manifest_path: Optional path to an existing raw manifest.

    Returns:
        Mapping of filename to SHA-256 hex digest.
    """
    hashes: Dict[str, str] = {}
    cached_hashes: Dict[str, Tuple[int, str]] = {}

    if known_manifest_path and known_manifest_path.is_file():
        try:
            with open(known_manifest_path, "r", encoding="utf-8") as fp:
                manifest_data = json.load(fp)
                for entry in manifest_data.get("files", []):
                    if "filename" in entry and "sha256" in entry and "size_bytes" in entry:
                        cached_hashes[entry["filename"]] = (entry["size_bytes"], entry["sha256"])
        except (json.JSONDecodeError, OSError):
            cached_hashes = {}

    for p in band_paths:
        file_size = p.stat().st_size
        if p.name in cached_hashes and cached_hashes[p.name][0] == file_size:
            hashes[p.name] = cached_hashes[p.name][1]
        else:
            hasher = hashlib.sha256()
            with open(p, "rb") as fp:
                while chunk := fp.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            hashes[p.name] = hasher.hexdigest()

    return hashes


def stack_bands_and_generate_mask(
    band_paths: Sequence[Union[str, Path]],
    output_stacked_path: Union[str, Path],
    output_mask_path: Union[str, Path],
    tile_size: int = 512,
    compression: str = "deflate",
) -> Dict[str, Any]:
    """Stack three single-band rasters and compute an explicit spatial validity mask.

    Processes the rasters in memory-efficient tiled windows without reading
    the entire satellite scene into memory.

    Validity mask rule: A pixel is valid (1) if at least one of the three bands
    is nonzero; collar/invalid (0) only if all three bands are zero.

    Args:
        band_paths: Sequence of 3 band paths: (BAND2, BAND3, BAND4).
        output_stacked_path: Destination path for 3-band stacked GeoTIFF.
        output_mask_path: Destination path for 1-band validity mask GeoTIFF.
        tile_size: Internal tile dimension (width and height) for GeoTIFF output.
        compression: Lossless compression method (default: 'deflate').

    Returns:
        Dictionary detailing output statistics, pixel counts, and configurations.
    """
    validated = validate_inputs(band_paths)
    resolved_bands = [Path(p) for p in band_paths]

    out_stacked = Path(output_stacked_path)
    out_mask = Path(output_mask_path)
    out_stacked.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    width = validated["width"]
    height = validated["height"]
    crs = validated["crs"]
    transform = validated["transform"]
    dtype = validated["dtype"]

    stacked_profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 3,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": tile_size,
        "blockysize": tile_size,
        "compress": compression,
        "predictor": 2,  # Horizontal differencing for integer rasters
    }

    mask_profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": tile_size,
        "blockysize": tile_size,
        "compress": compression,
    }

    total_valid_pixels = 0
    total_collar_pixels = 0

    with (
        rasterio.open(resolved_bands[0]) as src2,
        rasterio.open(resolved_bands[1]) as src3,
        rasterio.open(resolved_bands[2]) as src4,
        rasterio.open(out_stacked, "w", **stacked_profile) as dst_stacked,
        rasterio.open(out_mask, "w", **mask_profile) as dst_mask,
    ):
        dst_stacked.set_band_description(1, "BAND2")
        dst_stacked.set_band_description(2, "BAND3")
        dst_stacked.set_band_description(3, "BAND4")
        dst_mask.set_band_description(1, "VALID_MASK")

        for _, window in dst_stacked.block_windows():
            b2 = src2.read(1, window=window)
            b3 = src3.read(1, window=window)
            b4 = src4.read(1, window=window)

            dst_stacked.write(b2, 1, window=window)
            dst_stacked.write(b3, 2, window=window)
            dst_stacked.write(b4, 3, window=window)

            # Validity rule: valid (1) if at least one band is non-zero; collar (0) if all three bands are 0.
            valid_chunk = ((b2 != 0) | (b3 != 0) | (b4 != 0)).astype(np.uint8)
            dst_mask.write(valid_chunk, 1, window=window)

            chunk_valid = int(np.count_nonzero(valid_chunk))
            total_valid_pixels += chunk_valid
            total_collar_pixels += int(valid_chunk.size - chunk_valid)

    total_pixels = width * height
    return {
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "valid_pixels": total_valid_pixels,
        "collar_pixels": total_collar_pixels,
        "valid_fraction": float(total_valid_pixels / total_pixels) if total_pixels > 0 else 0.0,
        "tile_size": tile_size,
        "compression": compression,
        "stacked_size_bytes": out_stacked.stat().st_size,
        "mask_size_bytes": out_mask.stat().st_size,
    }


def create_manifest(
    band_paths: Sequence[Path],
    output_stacked_path: Path,
    output_mask_path: Path,
    manifest_path: Path,
    stats: Dict[str, Any],
    source_hashes: Dict[str, str],
    validated_meta: Dict[str, Any],
) -> Path:
    """Write comprehensive JSON processing manifest detailing outputs, rules, and versions."""
    manifest_data = {
        "pipeline_stage": "preprocess",
        "description": "Geospatial 3-band stacking and validity mask generation for LISS-IV imagery",
        "uncalibrated_statement": (
            "Source digital numbers (DN) remain strictly uncalibrated, unnormalized, "
            "unclipped, and unchanged. Radiometric calibration to Top-Of-Atmosphere "
            "radiance or reflectance is deliberately deferred until target model "
            "and evaluation loss specifications are finalized."
        ),
        "band_ordering": list(BAND_ORDER_NAMES),
        "source_inputs": [
            {
                "filename": p.name,
                "relative_path": str(p).replace("\\", "/"),
                "size_bytes": p.stat().st_size,
                "sha256": source_hashes.get(p.name, ""),
                "width": validated_meta["width"],
                "height": validated_meta["height"],
                "crs": str(validated_meta["crs"]),
                "dtype": str(validated_meta["dtype"]),
            }
            for p in band_paths
        ],
        "outputs": {
            "stacked_image": {
                "filename": output_stacked_path.name,
                "relative_path": str(output_stacked_path).replace("\\", "/"),
                "size_bytes": stats["stacked_size_bytes"],
                "width": stats["width"],
                "height": stats["height"],
                "count": 3,
                "band_names": list(BAND_ORDER_NAMES),
                "dtype": str(validated_meta["dtype"]),
                "crs": str(validated_meta["crs"]),
                "transform": [float(x) for x in validated_meta["transform"]],
                "compression": stats["compression"],
                "tile_size": stats["tile_size"],
            },
            "validity_mask": {
                "filename": output_mask_path.name,
                "relative_path": str(output_mask_path).replace("\\", "/"),
                "size_bytes": stats["mask_size_bytes"],
                "width": stats["width"],
                "height": stats["height"],
                "count": 1,
                "dtype": "uint8",
                "crs": str(validated_meta["crs"]),
                "transform": [float(x) for x in validated_meta["transform"]],
                "valid_pixels": stats["valid_pixels"],
                "collar_pixels": stats["collar_pixels"],
                "valid_fraction": stats["valid_fraction"],
                "compression": stats["compression"],
                "tile_size": stats["tile_size"],
            },
        },
        "validity_mask_rule": {
            "rule": "valid = 1 if (BAND2 != 0 or BAND3 != 0 or BAND4 != 0) else 0",
            "rule_description": "Valid if at least one of the three bands is nonzero; collar/invalid only if all three bands are zero.",
            "collar_definition": "Pixels where all 3 spectral bands are zero represent background collar padding.",
            "assumptions_and_limitations": (
                "For this standard georeferenced LISS-IV product, zero-valued border "
                "pixels result from rotating the angled satellite pass into the north-up "
                "UTM coordinate reference frame. Across the valid terrain scene, minimum DN "
                "values are > 40. Therefore, (BAND2==0 and BAND3==0 and BAND4==0) unambiguously "
                "identifies collar padding for this scene. Note that in other sensor products or "
                "deep water/shadow regions, DN=0 could occasionally occur on valid terrain."
            ),
        },
        "environment": {
            "python_version": platform.python_version(),
            "rasterio_version": rasterio.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_data, fp, indent=2)

    return manifest_path


def preprocess_product(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    tile_size: int = 512,
    compression: str = "deflate",
) -> Dict[str, Any]:
    """Execute complete preprocessing workflow on an extracted LISS-IV product directory.

    Args:
        input_dir: Directory containing BAND2.tif, BAND3.tif, BAND4.tif.
        output_dir: Directory where stacked.tif, valid_mask.tif, and manifest.json will be saved.
        tile_size: Internal tile dimension for output rasters.
        compression: Lossless compression method (default: 'deflate').

    Returns:
        Summary dictionary of the execution results.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    band_paths = [in_dir / fname for fname in EXPECTED_BAND_FILENAMES]
    validated = validate_inputs(band_paths)

    output_stacked = out_dir / "stacked.tif"
    output_mask = out_dir / "valid_mask.tif"
    manifest_path = out_dir / "manifest.json"

    # Check for raw manifest to avoid redundant SHA-256 computation on 1.76 GB
    raw_manifest = in_dir.parent / "manifest.json"
    source_hashes = compute_or_lookup_hashes(band_paths, known_manifest_path=raw_manifest)

    stats = stack_bands_and_generate_mask(
        band_paths=band_paths,
        output_stacked_path=output_stacked,
        output_mask_path=output_mask,
        tile_size=tile_size,
        compression=compression,
    )

    create_manifest(
        band_paths=band_paths,
        output_stacked_path=output_stacked,
        output_mask_path=output_mask,
        manifest_path=manifest_path,
        stats=stats,
        source_hashes=source_hashes,
        validated_meta=validated,
    )

    return {
        "status": "success",
        "stacked_path": str(output_stacked),
        "mask_path": str(output_mask),
        "manifest_path": str(manifest_path),
        "stats": stats,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for geospatial preprocessing stage."""
    parser = argparse.ArgumentParser(
        description="Stack LISS-IV bands and generate spatial validity mask."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA",
        help="Path to directory containing BAND2.tif, BAND3.tif, and BAND4.tif",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim/stacked_product",
        help="Destination directory for stacked.tif, valid_mask.tif, and manifest.json",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile dimension (width and height) for output GeoTIFFs (default: 512)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="deflate",
        help="Lossless compression algorithm (default: deflate)",
    )

    args = parser.parse_args(argv)

    try:
        print(f"Starting preprocessing for input: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        result = preprocess_product(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            compression=args.compression,
        )
        print(f"Preprocessing completed successfully.")
        print(f"  Stacked raster: {result['stacked_path']} ({result['stats']['stacked_size_bytes']:,} bytes)")
        print(f"  Validity mask:  {result['mask_path']} ({result['stats']['mask_size_bytes']:,} bytes)")
        print(f"  Valid pixels:   {result['stats']['valid_pixels']:,} / {result['stats']['total_pixels']:,} ({result['stats']['valid_fraction']*100:.2f}%)")
        print(f"  Manifest:       {result['manifest_path']}")
        return 0
    except PreprocessingError as err:
        print(f"Error during preprocessing: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
