"""Geospatial preprocessing and radiometric calibration pipeline for LISS-IV multi-spectral imagery.

Validates single-band rasters, stacks them into a unified multi-band GeoTIFF,
generates a spatial validity mask for the collar/background, converts raw 10-bit
Digital Numbers (DN) to at-sensor spectral radiance and Top-Of-Atmosphere (TOA)
reflectance, prepares false-color composite (FCC) model inputs [B4, B3, B2],
and exports comprehensive processing manifests.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
import torch

EXPECTED_BAND_FILENAMES = ("BAND2.tif", "BAND3.tif", "BAND4.tif")
BAND_ORDER_NAMES = ("BAND2", "BAND3", "BAND4")

# Authoritative mean exo-atmospheric solar spectral irradiance (ESUN) values for
# ISRO Resourcesat-2 / Resourcesat-2A LISS-IV sensor bands (VNIR).
# Source: Pandya et al. (2013), "At-sensor Solar Exo-atmospheric Irradiance,
# Rayleigh Optical Thickness and Spectral parameters of RS-2 Sensors",
# Space Applications Centre (SAC), Indian Space Research Organisation (ISRO),
# Technical Report No. SAC/EPSA/SPSG/SR/01/2013.
#
# Units: mW / (cm^2 * um) (equivalent to 10 * W / (m^2 * um)).
# When radiance is expressed in mW / (cm^2 * sr * um) and ESUN in mW / (cm^2 * um),
# the units cancel out, yielding dimensionless reflectance.
RESOURCESAT2_LISS4_ESUN: Dict[str, float] = {
    "BAND2": 181.89,  # Band 2 (Green: 0.52 - 0.59 um)
    "BAND3": 156.96,  # Band 3 (Red:   0.62 - 0.68 um)
    "BAND4": 110.48,  # Band 4 (NIR:   0.77 - 0.86 um)
}

# Spectral band definitions and cross-sensor mapping
LISS4_BAND_INFO: Dict[str, Dict[str, Any]] = {
    "BAND2": {
        "band_number": 2,
        "name": "Green",
        "wavelength_min_um": 0.52,
        "wavelength_max_um": 0.59,
        "s2_equivalent_band": "B03",
        "native_resolution_m": 5.8,
        "product_resolution_m": 5.0,
    },
    "BAND3": {
        "band_number": 3,
        "name": "Red",
        "wavelength_min_um": 0.62,
        "wavelength_max_um": 0.68,
        "s2_equivalent_band": "B04",
        "native_resolution_m": 5.8,
        "product_resolution_m": 5.0,
    },
    "BAND4": {
        "band_number": 4,
        "name": "NIR",
        "wavelength_min_um": 0.77,
        "wavelength_max_um": 0.86,
        "s2_equivalent_band": "B08",
        "native_resolution_m": 5.8,
        "product_resolution_m": 5.0,
    },
}


class PreprocessingError(Exception):
    """Raised when validation or preprocessing fails."""


def calculate_earth_sun_distance(
    date_input: Union[str, datetime.date, datetime.datetime, int],
) -> float:
    """Calculate Earth-Sun distance (d) in astronomical units (AU) for a given date.

    Uses standard astronomical orbital mechanics:
        d = 1.0 - 0.01672 * cos(2 * pi * (DOY - 4) / 365.25)

    Args:
        date_input: Date as string ('DD-MMM-YYYY', 'YYYY-MM-DD', etc.), date/datetime object,
            or integer day-of-year (1-366).

    Returns:
        Earth-Sun distance in AU (typically between 0.9832 and 1.0167 AU).

    Raises:
        ValueError: If string date cannot be parsed or DOY is out of range.
    """
    if isinstance(date_input, int):
        doy = date_input
    elif isinstance(date_input, (datetime.date, datetime.datetime)):
        doy = date_input.timetuple().tm_yday
    elif isinstance(date_input, str):
        clean_str = date_input.strip()
        parsed = None
        for fmt in ("%d-%b-%Y", "%d-%B-%Y", "%Y-%m-%d", "%Y%m%d", "%d/%m/%Y", "%d-%m-%Y"):
            try:
                parsed = datetime.datetime.strptime(clean_str, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"Could not parse acquisition date string: '{date_input}'")
        doy = parsed.timetuple().tm_yday
    else:
        raise TypeError(f"Unsupported date_input type: {type(date_input).__name__}")

    if not (1 <= doy <= 366):
        raise ValueError(f"Day of year must be in 1-366, got {doy}")

    theta = 2.0 * math.pi * (doy - 4) / 365.25
    d = 1.0 - 0.01672 * math.cos(theta)
    return float(d)


def parse_liss4_metadata(meta_path_or_dir: Union[str, Path]) -> Dict[str, Any]:
    """Parse authoritative product and calibration metadata from LISS-IV product files.

    Locates and parses BAND_META.txt or *.meta files within an extracted product directory
    or from a direct file path.

    Args:
        meta_path_or_dir: Path to metadata file (e.g. BAND_META.txt) or product directory.

    Returns:
        Structured dictionary containing:
            - sat_id: Satellite identifier ('IRS-R2A')
            - sensor: Sensor identifier ('L4FX')
            - sub_scene: Sub-scene ('A')
            - date_of_pass: Acquisition date string ('03-JUN-2026')
            - earth_sun_distance_au: Calculated Earth-Sun distance in AU
            - sun_elevation_deg: Sun elevation angle at scene center in degrees
            - sun_azimuth_deg: Sun azimuth angle at scene center in degrees
            - solar_zenith_deg: Solar zenith angle in degrees (90 - sun_elevation)
            - bits_per_pixel: Quantization bit depth (10)
            - max_dn: Theoretical maximum digital number (1023)
            - min_dn: Theoretical minimum digital number (0)
            - calibration: Per-band L_min, L_max limits in mW / (cm^2 * sr * um)
            - cloud_percent: Scene-level cloud cover percentage if present, or None
            - metadata_file: Absolute path of the parsed metadata file

    Raises:
        PreprocessingError: If metadata file cannot be found or essential calibration
            parameters (L_min, L_max) are missing.
    """
    path_obj = Path(meta_path_or_dir)
    target_file: Optional[Path] = None

    if path_obj.is_dir():
        band_meta = path_obj / "BAND_META.txt"
        if band_meta.is_file():
            target_file = band_meta
        else:
            meta_candidates = list(path_obj.glob("*.meta"))
            if meta_candidates:
                target_file = meta_candidates[0]
            else:
                raise PreprocessingError(
                    f"No BAND_META.txt or *.meta file found in directory {path_obj}"
                )
    elif path_obj.is_file():
        target_file = path_obj
    else:
        raise PreprocessingError(f"Metadata path does not exist: {path_obj}")

    kv: Dict[str, str] = {}
    with open(target_file, "r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip()

    # Extract required satellite and sensor identity
    sat_id = kv.get("SatID", "IRS-R2A")
    sensor = kv.get("Sensor", "L4FX")
    sub_scene = kv.get("SubScene", "A")
    date_of_pass = kv.get("DateOfPass", "")

    # Parse Sun geometry
    sun_elev_raw = kv.get("SunElevationAtCenter")
    if sun_elev_raw is None or sun_elev_raw == "-NA-":
        raise PreprocessingError(f"Missing SunElevationAtCenter in metadata file: {target_file}")
    sun_elevation_deg = float(sun_elev_raw)
    solar_zenith_deg = float(90.0 - sun_elevation_deg)

    sun_azim_raw = kv.get("SunAzimuthAtCenter")
    sun_azimuth_deg = float(sun_azim_raw) if (sun_azim_raw and sun_azim_raw != "-NA-") else 0.0

    # Parse Quantization bit depth
    bits_raw = kv.get("BitsPerPixel", "10")
    bits_per_pixel = int(bits_raw) if bits_raw.isdigit() else 10
    max_dn = float((1 << bits_per_pixel) - 1)
    min_dn = 0.0

    # Parse per-band radiometric calibration limits (L_min, L_max)
    calibration: Dict[str, Dict[str, Any]] = {}
    for band_key in ("B2", "B3", "B4"):
        out_name = f"BAND{band_key[1]}"
        l_min_k = f"{band_key}_Lmin"
        l_max_k = f"{band_key}_Lmax"

        if l_min_k not in kv or l_max_k not in kv:
            raise PreprocessingError(
                f"Missing radiometric calibration parameter ({l_min_k} or {l_max_k}) in {target_file}"
            )

        l_min = float(kv[l_min_k])
        l_max = float(kv[l_max_k])
        if l_max <= l_min:
            raise PreprocessingError(
                f"Invalid calibration parameters for {band_key}: L_max ({l_max}) <= L_min ({l_min})"
            )

        calibration[out_name] = {
            "l_min": l_min,
            "l_max": l_max,
            "units": "mW/(cm^2*sr*um)",
            "dn_min": min_dn,
            "dn_max": max_dn,
        }

    # Calculate Earth-Sun distance
    earth_sun_distance = 1.0
    if date_of_pass and date_of_pass != "-NA-":
        try:
            earth_sun_distance = calculate_earth_sun_distance(date_of_pass)
        except ValueError:
            earth_sun_distance = 1.0

    # Parse Cloud Cover
    cloud_raw = kv.get("CloudPercent")
    cloud_percent: Optional[float] = None
    if cloud_raw and cloud_raw != "-NA-":
        try:
            cloud_percent = float(cloud_raw)
        except ValueError:
            cloud_percent = None

    return {
        "sat_id": sat_id,
        "sensor": sensor,
        "sub_scene": sub_scene,
        "date_of_pass": date_of_pass,
        "earth_sun_distance_au": earth_sun_distance,
        "sun_elevation_deg": sun_elevation_deg,
        "sun_azimuth_deg": sun_azimuth_deg,
        "solar_zenith_deg": solar_zenith_deg,
        "bits_per_pixel": bits_per_pixel,
        "min_dn": min_dn,
        "max_dn": max_dn,
        "calibration": calibration,
        "cloud_percent": cloud_percent,
        "metadata_file": str(target_file.resolve()),
    }


def convert_dn_to_radiance(
    dn: Union[np.ndarray, float, int],
    l_min: float,
    l_max: float,
    dn_min: float = 0.0,
    dn_max: float = 1023.0,
    mask_collar_zeros: bool = False,
) -> np.ndarray:
    """Convert raw LISS-IV Digital Numbers (DN) into spectral radiance.

    Standard linear calibration formula:
        L_lambda = L_min + ((L_max - L_min) / (DN_max - DN_min)) * (DN - DN_min)

    Args:
        dn: Raw integer Digital Number array or scalar (0-1023 for 10-bit).
        l_min: Minimum spectral radiance at DN_min in mW / (cm^2 * sr * um).
        l_max: Maximum spectral radiance at DN_max in mW / (cm^2 * sr * um).
        dn_min: Lower calibration endpoint (default: 0.0).
        dn_max: Upper calibration endpoint (default: 1023.0 for 10-bit LISS-IV).
        mask_collar_zeros: If True, pixels with DN == 0 are preserved as 0.0 radiance
            instead of being mapped to L_min (used for satellite collar boundary).

    Returns:
        Spectral radiance array in mW / (cm^2 * sr * um) as float32 array.

    Raises:
        ValueError: If dn_max <= dn_min or l_max <= l_min.
    """
    if dn_max <= dn_min:
        raise ValueError(
            f"Invalid DN endpoints: dn_max ({dn_max}) must be greater than dn_min ({dn_min})"
        )
    if l_max <= l_min:
        raise ValueError(
            f"Invalid radiance limits: l_max ({l_max}) must be greater than l_min ({l_min})"
        )

    dn_arr = np.asarray(dn, dtype=np.float32)
    slope = (l_max - l_min) / (dn_max - dn_min)
    radiance = l_min + slope * (dn_arr - dn_min)

    if mask_collar_zeros:
        radiance = np.where(dn_arr == 0.0, 0.0, radiance)

    return radiance.astype(np.float32)


def convert_radiance_to_toa_reflectance(
    radiance: Union[np.ndarray, float],
    esun: float,
    solar_zenith_deg: float,
    earth_sun_distance_au: float = 1.0,
) -> np.ndarray:
    """Convert spectral radiance to Top-Of-Atmosphere (TOA) reflectance.

    Formula:
        rho_TOA = (pi * L_lambda * d^2) / (ESUN_lambda * cos(theta_s))

    Args:
        radiance: Spectral radiance in mW / (cm^2 * sr * um).
        esun: Mean exo-atmospheric solar spectral irradiance in mW / (cm^2 * um)
            (must match unit system of radiance).
        solar_zenith_deg: Solar zenith angle theta_s in degrees (0 <= theta_s < 90).
        earth_sun_distance_au: Earth-Sun distance d in astronomical units (AU).

    Returns:
        Dimensionless Top-Of-Atmosphere reflectance array.

    Raises:
        ValueError: If esun <= 0, d <= 0, or solar_zenith_deg outside [0, 90).
    """
    if esun <= 0.0:
        raise ValueError(f"ESUN must be strictly positive, got {esun}")
    if earth_sun_distance_au <= 0.0:
        raise ValueError(
            f"Earth-Sun distance must be strictly positive, got {earth_sun_distance_au}"
        )
    if not (0.0 <= solar_zenith_deg < 90.0):
        raise ValueError(f"Solar zenith angle must be in [0, 90) degrees, got {solar_zenith_deg}")

    rad_arr = np.asarray(radiance, dtype=np.float32)
    theta_rad = math.radians(solar_zenith_deg)
    cos_theta = math.cos(theta_rad)

    numerator = math.pi * rad_arr * (earth_sun_distance_au**2)
    denominator = esun * cos_theta
    toa_reflectance = numerator / denominator
    return toa_reflectance.astype(np.float32)


def prepare_liss4_fcc_model_input(
    stacked_dn: Union[np.ndarray, torch.Tensor],
    metadata: Dict[str, Any],
    esun_dict: Optional[Dict[str, float]] = None,
    normalize_for_model: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Convert a 3-band LISS-IV DN array [BAND2, BAND3, BAND4] to FCC model input tensor [B4, B3, B2].

    Follows the verified physical calibration chain:
        Stored DN -> Calibrated Radiance -> TOA Reflectance -> Channel Permutation to FCC -> Model Normalization

    Args:
        stacked_dn: 3-band array shaped (3, H, W) in source order [BAND2 (Green), BAND3 (Red), BAND4 (NIR)].
        metadata: Parsed metadata dictionary from parse_liss4_metadata.
        esun_dict: Optional dictionary with 'BAND2', 'BAND3', 'BAND4' ESUN values. Defaults to
            authoritative Resourcesat-2 LISS-IV constants from Pandya et al. (2013).
        normalize_for_model: If True, clips reflectance to [0.0, 1.0] expected by the U-Net.
        device: PyTorch device for output tensor.

    Returns:
        Tuple of (model_tensor, valid_mask):
            - model_tensor: PyTorch tensor shaped (1, 3, H, W) with channels [B4 (NIR), B3 (Red), B2 (Green)].
            - valid_mask: Boolean numpy array (H, W) where True indicates valid scene terrain.

    Raises:
        ValueError: If input does not have 3 channels or required metadata is missing.
    """
    if isinstance(stacked_dn, torch.Tensor):
        dn_np = stacked_dn.detach().cpu().numpy()
    else:
        dn_np = np.asarray(stacked_dn)

    if dn_np.ndim != 3 or dn_np.shape[0] != 3:
        raise ValueError(f"Expected 3-band array shaped (3, H, W), got shape {dn_np.shape}")

    solar_zenith = float(metadata["solar_zenith_deg"])
    d_au = float(metadata.get("earth_sun_distance_au", 1.0))
    calib = metadata["calibration"]

    active_esun = esun_dict if esun_dict is not None else RESOURCESAT2_LISS4_ESUN
    for b_name in ("BAND2", "BAND3", "BAND4"):
        if b_name not in active_esun:
            raise ValueError(f"Missing ESUN value for band '{b_name}' in provided ESUN dictionary")

    # Extract source bands
    b2_dn = dn_np[0]  # Green
    b3_dn = dn_np[1]  # Red
    b4_dn = dn_np[2]  # NIR

    # Identify spatial validity collar: valid if at least one band is non-zero
    valid_mask = (b2_dn != 0) | (b3_dn != 0) | (b4_dn != 0)

    # Convert each band from DN -> Radiance -> TOA Reflectance
    # Band 2 (Green)
    rad_b2 = convert_dn_to_radiance(
        b2_dn,
        l_min=calib["BAND2"]["l_min"],
        l_max=calib["BAND2"]["l_max"],
        mask_collar_zeros=True,
    )
    refl_b2 = convert_radiance_to_toa_reflectance(rad_b2, active_esun["BAND2"], solar_zenith, d_au)

    # Band 3 (Red)
    rad_b3 = convert_dn_to_radiance(
        b3_dn,
        l_min=calib["BAND3"]["l_min"],
        l_max=calib["BAND3"]["l_max"],
        mask_collar_zeros=True,
    )
    refl_b3 = convert_radiance_to_toa_reflectance(rad_b3, active_esun["BAND3"], solar_zenith, d_au)

    # Band 4 (NIR)
    rad_b4 = convert_dn_to_radiance(
        b4_dn,
        l_min=calib["BAND4"]["l_min"],
        l_max=calib["BAND4"]["l_max"],
        mask_collar_zeros=True,
    )
    refl_b4 = convert_radiance_to_toa_reflectance(rad_b4, active_esun["BAND4"], solar_zenith, d_au)

    # Reorder channels to FCC: [B4 (NIR), B3 (Red), B2 (Green)]
    # This precisely matches Sentinel-2 [B08 (NIR), B04 (Red), B03 (Green)]
    fcc_stack = np.stack([refl_b4, refl_b3, refl_b2], axis=0)

    # Apply collar masking: collar pixels are set to 0.0
    fcc_stack[:, ~valid_mask] = 0.0

    if normalize_for_model:
        fcc_stack = np.clip(fcc_stack, 0.0, 1.0)

    tensor = torch.from_numpy(fcc_stack).unsqueeze(0).to(dtype=torch.float32)
    if device is not None:
        tensor = tensor.to(device)

    return tensor, valid_mask


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
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="Parse and display radiometric calibration metadata and ESUN parameters, then exit",
    )

    args = parser.parse_args(argv)

    if args.show_calibration:
        try:
            meta = parse_liss4_metadata(args.input_dir)
            print("==================================================")
            print("LISS-IV Product Metadata & Radiometric Calibration")
            print("==================================================")
            print(
                f"Satellite / Sensor: {meta['sat_id']} / {meta['sensor']} (Subscene {meta['sub_scene']})"
            )
            print(
                f"Acquisition Date:   {meta['date_of_pass']} (Earth-Sun Distance: {meta['earth_sun_distance_au']:.4f} AU)"
            )
            print(
                f"Sun Geometry:       Elevation {meta['sun_elevation_deg']:.2f} deg | Zenith {meta['solar_zenith_deg']:.2f} deg | Azimuth {meta['sun_azimuth_deg']:.2f} deg"
            )
            print(
                f"Bit Depth:          {meta['bits_per_pixel']}-bit (DN range: {meta['min_dn']:.0f} - {meta['max_dn']:.0f})"
            )
            print(
                f"Cloud Cover:        {meta['cloud_percent']}%"
                if meta["cloud_percent"] is not None
                else "Cloud Cover:        Not available (-NA-)"
            )
            print("\nRadiometric Limits (L_min, L_max in mW / cm^2 / sr / um):")
            for b_name, c_info in meta["calibration"].items():
                esun_val = RESOURCESAT2_LISS4_ESUN.get(b_name, 0.0)
                print(
                    f"  {b_name}: L_min = {c_info['l_min']:7.2f} | L_max = {c_info['l_max']:7.2f} | ESUN = {esun_val:6.2f} mW/(cm^2*um)"
                )
            print("==================================================")
            return 0
        except PreprocessingError as err:
            print(f"Error parsing calibration metadata: {err}", file=sys.stderr)
            return 1

    try:
        print(f"Starting preprocessing for input: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        result = preprocess_product(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            tile_size=args.tile_size,
            compression=args.compression,
        )
        print("Preprocessing completed successfully.")
        print(
            f"  Stacked raster: {result['stacked_path']} ({result['stats']['stacked_size_bytes']:,} bytes)"
        )
        print(
            f"  Validity mask:  {result['mask_path']} ({result['stats']['mask_size_bytes']:,} bytes)"
        )
        print(
            f"  Valid pixels:   {result['stats']['valid_pixels']:,} / {result['stats']['total_pixels']:,} ({result['stats']['valid_fraction']*100:.2f}%)"
        )
        print(f"  Manifest:       {result['manifest_path']}")
        return 0
    except PreprocessingError as err:
        print(f"Error during preprocessing: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
