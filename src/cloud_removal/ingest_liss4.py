"""Ingestion, validation, and DVC tracking preparation for authentic LISS-IV scenes.

Verifies product provenance, metadata integrity, cloud cover threshold (<= 2%),
georeferencing, and raster structure for Resourcesat-2A LISS-IV scenes acquired
from ISRO's Bhoonidhi portal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import rasterio

from cloud_removal.preprocess import PreprocessingError, parse_liss4_metadata

REQUIRED_RASTERS = ("BAND2.tif", "BAND3.tif", "BAND4.tif")


class IngestionError(Exception):
    """Raised when scene fails authentication or validation criteria."""


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hexadecimal hash digest of a file in streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as fp:
        while chunk := fp.read(8 * 1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_liss4_scene(
    product_dir: Path,
    max_cloud_percent: float = 2.0,
    allow_missing_cloud_percent: bool = False,
) -> Dict[str, Any]:
    """Validate that an extracted directory is a genuine Resourcesat-2A LISS-IV product.

    Args:
        product_dir: Path to directory containing scene files.
        max_cloud_percent: Maximum allowable cloud cover percentage (default: 2.0%).
        allow_missing_cloud_percent: If True, do not reject scenes where CloudPercent
            is recorded as -NA- in official metadata.

    Returns:
        Dictionary of verified scene metadata, file paths, and hashes.

    Raises:
        IngestionError: If the product fails sensor identity, cloud threshold, or
            band presence validation.
    """
    if not product_dir.is_dir():
        raise IngestionError(f"Product directory does not exist: {product_dir}")

    # Parse and verify metadata
    try:
        metadata = parse_liss4_metadata(product_dir)
    except PreprocessingError as err:
        raise IngestionError(f"Failed to parse product metadata: {err}") from err

    # 1. Verify satellite and sensor identity
    sat_id = metadata.get("sat_id", "")
    sensor = metadata.get("sensor", "")
    if sat_id not in ("IRS-R2A", "IRS-R2", "RESOURCESAT-2A", "RESOURCESAT-2"):
        raise IngestionError(
            f"Invalid satellite identity: expected Resourcesat-2/2A, got '{sat_id}'"
        )
    if sensor not in ("L4FX", "LISS4", "LISS-IV"):
        raise IngestionError(f"Invalid sensor identity: expected L4FX (LISS-IV), got '{sensor}'")

    # 2. Verify cloud cover threshold (<= 2.0%)
    cloud_percent = metadata.get("cloud_percent")
    if cloud_percent is not None:
        if cloud_percent > max_cloud_percent:
            raise IngestionError(
                f"Scene cloud cover ({cloud_percent:.1f}%) exceeds maximum allowable threshold ({max_cloud_percent:.1f}%)."
            )
    else:
        if not allow_missing_cloud_percent:
            raise IngestionError(
                "CloudPercent is recorded as '-NA-' in metadata. To ingest, verify cloud-freeness "
                "via visual inspection of browse images and supply '--allow-missing-cloud-percent'."
            )

    # 3. Verify presence and spatial consistency of all required raster bands
    raster_info: List[Dict[str, Any]] = []
    ref_shape = None
    ref_crs = None

    for b_name in REQUIRED_RASTERS:
        b_path = product_dir / b_name
        if not b_path.is_file():
            raise IngestionError(f"Required band raster missing: {b_path}")

        try:
            with rasterio.open(b_path) as src:
                if src.count != 1:
                    raise IngestionError(
                        f"Expected single-band raster in {b_name}, got count={src.count}"
                    )
                if src.dtypes[0] not in ("uint16", "int16"):
                    raise IngestionError(
                        f"Expected 16-bit integer raster in {b_name}, got {src.dtypes[0]}"
                    )

                if ref_shape is None:
                    ref_shape = (src.height, src.width)
                    ref_crs = src.crs
                else:
                    if (src.height, src.width) != ref_shape:
                        raise IngestionError(
                            f"Dimension mismatch in {b_name}: {(src.height, src.width)} != {ref_shape}"
                        )
                    if src.crs != ref_crs:
                        raise IngestionError(f"CRS mismatch in {b_name}: {src.crs} != {ref_crs}")

                raster_info.append(
                    {
                        "filename": b_name,
                        "path": str(b_path),
                        "width": src.width,
                        "height": src.height,
                        "crs": str(src.crs),
                        "dtype": str(src.dtypes[0]),
                        "size_bytes": b_path.stat().st_size,
                    }
                )
        except rasterio.errors.RasterioIOError as err:
            raise IngestionError(f"Failed to read GeoTIFF {b_path}: {err}") from err

    return {
        "status": "valid",
        "product_dir": str(product_dir.resolve()),
        "satellite": sat_id,
        "sensor": sensor,
        "date_of_pass": metadata.get("date_of_pass"),
        "cloud_percent": cloud_percent,
        "sun_elevation_deg": metadata.get("sun_elevation_deg"),
        "solar_zenith_deg": metadata.get("solar_zenith_deg"),
        "dimensions": ref_shape,
        "crs": str(ref_crs),
        "rasters": raster_info,
        "metadata": metadata,
    }


def ingest_liss4_product(
    source_path: Union[str, Path],
    destination_raw_dir: Union[str, Path] = "data/raw",
    max_cloud_percent: float = 2.0,
    allow_missing_cloud_percent: bool = False,
) -> Dict[str, Any]:
    """Ingest, validate, and register an authentic Resourcesat-2A LISS-IV scene.

    Args:
        source_path: Path to downloaded product ZIP archive or extracted folder.
        destination_raw_dir: Destination parent folder under data/raw/.
        max_cloud_percent: Maximum acceptable cloud cover percentage.
        allow_missing_cloud_percent: Whether to allow scenes where CloudPercent is -NA-.

    Returns:
        Structured summary with validated metadata, destination directory, and DVC instructions.
    """
    src = Path(source_path)
    dest_base = Path(destination_raw_dir)
    dest_base.mkdir(parents=True, exist_ok=True)

    if src.is_file() and src.suffix.lower() == ".zip":
        ots_name = src.stem
        extract_dir = dest_base / ots_name
        if extract_dir.exists():
            raise IngestionError(f"Destination directory already exists: {extract_dir}")
        print(f"Extracting archive {src.name} to {extract_dir}...")
        with zipfile.ZipFile(src, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        product_dir = extract_dir
    elif src.is_dir():
        product_dir = src
    else:
        raise IngestionError(f"Source must be a directory or .zip file: {src}")

    # Validate scene
    validated = validate_liss4_scene(
        product_dir=product_dir,
        max_cloud_percent=max_cloud_percent,
        allow_missing_cloud_percent=allow_missing_cloud_percent,
    )

    # Compute SHA-256 for all constituent files
    print("Computing file checksums for dataset manifest...")
    manifest_files: List[Dict[str, Any]] = []
    for item in sorted(product_dir.glob("*")):
        if item.is_file():
            manifest_files.append(
                {
                    "filename": item.name,
                    "size_bytes": item.stat().st_size,
                    "sha256": compute_file_sha256(item),
                }
            )

    # Write manifest
    manifest_path = product_dir / "manifest.json"
    manifest_payload = {
        "dataset_name": "Resourcesat-2A LISS-IV Clear Reference",
        "satellite": validated["satellite"],
        "sensor": validated["sensor"],
        "date_of_pass": validated["date_of_pass"],
        "cloud_percent": validated["cloud_percent"],
        "crs": validated["crs"],
        "dimensions": validated["dimensions"],
        "files": manifest_files,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest_payload, fp, indent=2)

    dvc_rel_path = (
        product_dir.relative_to(Path.cwd())
        if product_dir.is_relative_to(Path.cwd())
        else product_dir
    )
    dvc_command = f"dvc add {dvc_rel_path}"

    return {
        "status": "ingested_successfully",
        "product_dir": str(product_dir),
        "manifest_path": str(manifest_path),
        "validated_info": validated,
        "dvc_command": dvc_command,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for authentic LISS-IV scene ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest and validate authentic Resourcesat-2A LISS-IV products from Bhoonidhi."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to downloaded scene folder or .zip file",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="data/raw",
        help="Target data directory (default: data/raw)",
    )
    parser.add_argument(
        "--max-cloud-percent",
        type=float,
        default=2.0,
        help="Maximum permissible cloud percentage (default: 2.0)",
    )
    parser.add_argument(
        "--allow-missing-cloud-percent",
        action="store_true",
        help="Permit ingestion if CloudPercent is -NA- (requires manual visual cloud inspection)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source directory without copying or creating manifests",
    )

    args = parser.parse_args(argv)

    try:
        src_path = Path(args.source)
        if args.check_only:
            val = validate_liss4_scene(
                src_path,
                max_cloud_percent=args.max_cloud_percent,
                allow_missing_cloud_percent=args.allow_missing_cloud_percent,
            )
            print("==================================================")
            print("LISS-IV Product Validation: SUCCESS")
            print("==================================================")
            print(f"Satellite / Sensor: {val['satellite']} / {val['sensor']}")
            print(f"Acquisition Date:   {val['date_of_pass']}")
            print(
                f"Cloud Cover:        {val['cloud_percent']}%"
                if val["cloud_percent"] is not None
                else "Cloud Cover:        -NA-"
            )
            print(f"Spatial CRS:        {val['crs']}")
            print(f"Scene Dimensions:   {val['dimensions'][1]} x {val['dimensions'][0]} pixels")
            return 0

        res = ingest_liss4_product(
            source_path=src_path,
            destination_raw_dir=args.target_dir,
            max_cloud_percent=args.max_cloud_percent,
            allow_missing_cloud_percent=args.allow_missing_cloud_percent,
        )
        print("==================================================")
        print("LISS-IV Scene Ingested Successfully")
        print("==================================================")
        print(f"Product Directory: {res['product_dir']}")
        print(f"Manifest:          {res['manifest_path']}")
        print(f"DVC Versioning:    {res['dvc_command']}")
        print("==================================================")
        return 0
    except IngestionError as err:
        print(f"Ingestion validation error: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
