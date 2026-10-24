"""Unit tests for the geospatial preprocessing pipeline."""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from cloud_removal.preprocess import (
    EXPECTED_BAND_FILENAMES,
    PreprocessingError,
    preprocess_product,
    validate_inputs,
)


@pytest.fixture
def synthetic_product_fixture(tmp_path: Path) -> Tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic 3-band LISS-IV product directory for testing."""
    in_dir = tmp_path / "raw_product"
    in_dir.mkdir(parents=True, exist_ok=True)

    width, height = 32, 32
    crs = "EPSG:32643"
    transform = from_origin(300000.0, 2100000.0, 5.0, 5.0)

    # Deterministic pixel data
    np.random.seed(123)
    b2 = np.random.randint(100, 300, size=(height, width), dtype=np.uint16)
    b3 = np.random.randint(200, 400, size=(height, width), dtype=np.uint16)
    b4 = np.random.randint(300, 600, size=(height, width), dtype=np.uint16)

    # Collar background: row 0 and col 0 are all zeros across all 3 bands
    b2[0, :] = 0
    b3[0, :] = 0
    b4[0, :] = 0
    b2[:, 0] = 0
    b3[:, 0] = 0
    b4[:, 0] = 0

    # Test edge case: pixel at (1, 1) has b2=0, b3=0, but b4=350 (valid pixel)
    b2[1, 1] = 0
    b3[1, 1] = 0
    b4[1, 1] = 350

    bands_data = [b2, b3, b4]
    for idx, fname in enumerate(EXPECTED_BAND_FILENAMES):
        file_path = in_dir / fname
        with rasterio.open(
            file_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=np.uint16,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(bands_data[idx], 1)

    return in_dir, b2, b3, b4


def test_validate_inputs_success(synthetic_product_fixture):
    """Verify input validation succeeds when bands are correctly aligned."""
    in_dir, _, _, _ = synthetic_product_fixture
    paths = [in_dir / f for f in EXPECTED_BAND_FILENAMES]
    meta = validate_inputs(paths)

    assert meta["width"] == 32
    assert meta["height"] == 32
    assert "32643" in str(meta["crs"])
    assert meta["dtype"] == "uint16"
    assert len(meta["paths"]) == 3


def test_validate_inputs_missing_file(synthetic_product_fixture):
    """Verify clear error is raised if a source band file is missing."""
    in_dir, _, _, _ = synthetic_product_fixture
    paths = [in_dir / "BAND2.tif", in_dir / "BAND3.tif", in_dir / "MISSING_BAND4.tif"]
    with pytest.raises(PreprocessingError, match="does not exist"):
        validate_inputs(paths)


def test_validate_inputs_dimension_mismatch(tmp_path: Path):
    """Verify clear error is raised if source bands have mismatched dimensions."""
    b2_path = tmp_path / "BAND2.tif"
    b3_path = tmp_path / "BAND3.tif"
    b4_path = tmp_path / "BAND4.tif"

    crs = "EPSG:32643"
    t = from_origin(0, 0, 5, 5)

    with rasterio.open(
        b2_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=1,
        dtype=np.uint16,
        crs=crs,
        transform=t,
    ) as dst:
        dst.write(np.ones((32, 32), dtype=np.uint16), 1)
    # B3 has width 30 instead of 32
    with rasterio.open(
        b3_path,
        "w",
        driver="GTiff",
        width=30,
        height=32,
        count=1,
        dtype=np.uint16,
        crs=crs,
        transform=t,
    ) as dst:
        dst.write(np.ones((32, 30), dtype=np.uint16), 1)
    with rasterio.open(
        b4_path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=1,
        dtype=np.uint16,
        crs=crs,
        transform=t,
    ) as dst:
        dst.write(np.ones((32, 32), dtype=np.uint16), 1)

    with pytest.raises(PreprocessingError, match="Dimension mismatch"):
        validate_inputs([b2_path, b3_path, b4_path])


def test_validate_inputs_transform_mismatch(tmp_path: Path):
    """Verify clear error is raised if source bands have mismatched affine transforms."""
    b2_path = tmp_path / "BAND2.tif"
    b3_path = tmp_path / "BAND3.tif"
    b4_path = tmp_path / "BAND4.tif"

    crs = "EPSG:32643"
    t1 = from_origin(100, 100, 5, 5)
    t2 = from_origin(200, 200, 5, 5)

    with rasterio.open(
        b2_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype=np.uint16,
        crs=crs,
        transform=t1,
    ) as dst:
        dst.write(np.ones((16, 16), dtype=np.uint16), 1)
    with rasterio.open(
        b3_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype=np.uint16,
        crs=crs,
        transform=t2,
    ) as dst:
        dst.write(np.ones((16, 16), dtype=np.uint16), 1)
    with rasterio.open(
        b4_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype=np.uint16,
        crs=crs,
        transform=t1,
    ) as dst:
        dst.write(np.ones((16, 16), dtype=np.uint16), 1)

    with pytest.raises(PreprocessingError, match="Transform mismatch"):
        validate_inputs([b2_path, b3_path, b4_path])


def test_stacking_and_mask_generation(tmp_path: Path, synthetic_product_fixture):
    """Verify successful 3-band stacking, pixel fidelity, validity mask rule, and manifest."""
    in_dir, b2_orig, b3_orig, b4_orig = synthetic_product_fixture
    out_dir = tmp_path / "output_stacked"

    result = preprocess_product(in_dir, out_dir, tile_size=16, compression="deflate")
    assert result["status"] == "success"

    stacked_path = Path(result["stacked_path"])
    mask_path = Path(result["mask_path"])
    manifest_path = Path(result["manifest_path"])

    assert stacked_path.is_file()
    assert mask_path.is_file()
    assert manifest_path.is_file()

    # 1. Verify stacked raster
    with rasterio.open(stacked_path) as src_stacked:
        assert src_stacked.count == 3
        assert src_stacked.width == 32
        assert src_stacked.height == 32
        assert src_stacked.dtypes == ("uint16", "uint16", "uint16")
        assert src_stacked.descriptions == ("BAND2", "BAND3", "BAND4")
        assert src_stacked.profile.get("tiled", False)

        b1_stacked = src_stacked.read(1)
        b2_stacked = src_stacked.read(2)
        b3_stacked = src_stacked.read(3)

        # Check exact pixel preservation
        np.testing.assert_array_equal(b1_stacked, b2_orig)
        np.testing.assert_array_equal(b2_stacked, b3_orig)
        np.testing.assert_array_equal(b3_stacked, b4_orig)

    # 2. Verify validity mask
    with rasterio.open(mask_path) as src_mask:
        assert src_mask.count == 1
        assert src_mask.width == 32
        assert src_mask.height == 32
        assert src_mask.dtypes == ("uint8",)
        assert src_mask.crs == src_stacked.crs
        assert np.allclose(src_mask.transform, src_stacked.transform)

        mask_data = src_mask.read(1)
        # Row 0 and col 0 must be 0 (all bands zero)
        assert np.all(mask_data[0, :] == 0)
        assert np.all(mask_data[:, 0] == 0)

        # Pixel (1, 1) has b4 > 0, so must be valid (1)
        assert mask_data[1, 1] == 1

        # All other pixels must be valid (1)
        interior = mask_data[1:, 1:]
        assert np.all(interior == 1)

    # 3. Verify manifest
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)

    assert manifest["pipeline_stage"] == "preprocess"
    assert manifest["band_ordering"] == ["BAND2", "BAND3", "BAND4"]
    assert "uncalibrated_statement" in manifest
    assert "validity_mask_rule" in manifest
    assert len(manifest["source_inputs"]) == 3
    assert manifest["outputs"]["stacked_image"]["count"] == 3
    assert manifest["outputs"]["validity_mask"]["count"] == 1


def test_idempotent_rerun(tmp_path: Path, synthetic_product_fixture):
    """Verify that re-running preprocessing succeeds and produces identical outputs."""
    in_dir, _, _, _ = synthetic_product_fixture
    out_dir = tmp_path / "output_rerun"

    res1 = preprocess_product(in_dir, out_dir, tile_size=16)
    with rasterio.open(res1["stacked_path"]) as src1:
        data1 = src1.read()

    res2 = preprocess_product(in_dir, out_dir, tile_size=16)
    with rasterio.open(res2["stacked_path"]) as src2:
        data2 = src2.read()

    np.testing.assert_array_equal(data1, data2)


def test_validity_mask_rule_at_least_one_nonzero(tmp_path: Path):
    """Verify that validity mask is 1 if at least one band is nonzero, and 0 only if all are 0."""
    in_dir = tmp_path / "truth_table_product"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "truth_table_output"

    width, height = 8, 1
    crs = "EPSG:32643"
    transform = from_origin(100.0, 100.0, 5.0, 5.0)

    # 8 combinations of (b2 > 0, b3 > 0, b4 > 0)
    b2_data = np.array([[0, 100, 0, 0, 100, 100, 0, 100]], dtype=np.uint16)
    b3_data = np.array([[0, 0, 200, 0, 200, 0, 200, 200]], dtype=np.uint16)
    b4_data = np.array([[0, 0, 0, 300, 0, 300, 300, 300]], dtype=np.uint16)

    # Expected: 0 only when all 3 are 0 (index 0); 1 for all other 7 cases
    expected_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)

    for fname, data in zip(EXPECTED_BAND_FILENAMES, [b2_data, b3_data, b4_data]):
        with rasterio.open(
            in_dir / fname,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=np.uint16,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(data, 1)

    result = preprocess_product(in_dir, out_dir, tile_size=16)
    with rasterio.open(result["mask_path"]) as src_mask:
        mask_out = src_mask.read(1)

    np.testing.assert_array_equal(mask_out, expected_mask)
    assert result["stats"]["valid_pixels"] == 7
    assert result["stats"]["collar_pixels"] == 1


# ==============================================================================
# Radiometric TOA Radiance & Reflectance Calibration Tests (Task 3)
# ==============================================================================


def test_calculate_earth_sun_distance():
    """Verify Earth-Sun distance calculation across dates, seasons, and formats."""
    from cloud_removal.preprocess import calculate_earth_sun_distance

    # Perihelion (early January, ~DOY 4): d should be minimum (~0.983 AU)
    d_perihelion = calculate_earth_sun_distance(4)
    assert pytest.approx(d_perihelion, abs=1e-3) == 0.9833

    # Aphelion (early July, ~DOY 185): d should be maximum (~1.0167 AU)
    d_aphelion = calculate_earth_sun_distance(185)
    assert pytest.approx(d_aphelion, abs=1e-3) == 1.0167

    # June 3 (DOY 154): d should be ~1.0142 AU
    d_june3_str = calculate_earth_sun_distance("03-JUN-2026")
    d_iso_str = calculate_earth_sun_distance("2026-06-03")
    assert pytest.approx(d_june3_str, abs=1e-4) == 1.0142
    assert pytest.approx(d_june3_str, abs=1e-5) == d_iso_str

    # Invalid date strings and DOYs
    with pytest.raises(ValueError, match="Could not parse"):
        calculate_earth_sun_distance("invalid-date-string")

    with pytest.raises(ValueError, match="Day of year must be in 1-366"):
        calculate_earth_sun_distance(400)


def test_parse_liss4_metadata_real_scene():
    """Verify metadata parser on actual Resourcesat-2A LISS-IV scene in repository."""
    from cloud_removal.preprocess import parse_liss4_metadata

    meta_dir = Path("data/raw/RAF03JUN2026049233009500060SSANSTUC00GTDA")
    if not meta_dir.is_dir():
        pytest.skip("LISS-IV raw product directory not present in workspace")

    meta = parse_liss4_metadata(meta_dir)

    assert meta["sat_id"] == "IRS-R2A"
    assert meta["sensor"] == "L4FX"
    assert meta["sub_scene"] == "A"
    assert meta["date_of_pass"] == "03-JUN-2026"
    assert pytest.approx(meta["sun_elevation_deg"], abs=1e-3) == 70.0265
    assert pytest.approx(meta["solar_zenith_deg"], abs=1e-3) == 19.9735
    assert meta["bits_per_pixel"] == 10
    assert meta["max_dn"] == 1023.0
    assert meta["min_dn"] == 0.0

    # Calibration limits
    calib = meta["calibration"]
    assert "BAND2" in calib and "BAND3" in calib and "BAND4" in calib
    assert calib["BAND2"]["l_min"] == 0.0 and calib["BAND2"]["l_max"] == 52.0
    assert calib["BAND3"]["l_min"] == 0.0 and calib["BAND3"]["l_max"] == 47.0
    assert calib["BAND4"]["l_min"] == 0.0 and calib["BAND4"]["l_max"] == 31.5


def test_parse_liss4_metadata_synthetic_errors(tmp_path: Path):
    """Verify that metadata parser rejects files missing required calibration keys."""
    from cloud_removal.preprocess import parse_liss4_metadata

    invalid_meta = tmp_path / "BAND_META.txt"
    with open(invalid_meta, "w", encoding="utf-8") as f:
        f.write(
            "SatID= IRS-R2A\nSensor= L4FX\nSunElevationAtCenter= 65.0\n"
        )  # Missing B2_Lmin, B2_Lmax

    with pytest.raises(PreprocessingError, match="Missing radiometric calibration"):
        parse_liss4_metadata(invalid_meta)


def test_convert_dn_to_radiance_analytical():
    """Verify linear DN to radiance conversion against known analytical values."""
    from cloud_removal.preprocess import convert_dn_to_radiance

    dn = np.array([0, 511.5, 1023], dtype=np.float32)
    l_min = 10.0
    l_max = 50.0

    radiance = convert_dn_to_radiance(dn, l_min=l_min, l_max=l_max, dn_min=0.0, dn_max=1023.0)

    assert pytest.approx(radiance[0], abs=1e-5) == 10.0  # DN=0 -> L_min
    assert pytest.approx(radiance[1], abs=1e-5) == 30.0  # DN=midpoint -> L_mid
    assert pytest.approx(radiance[2], abs=1e-5) == 50.0  # DN=1023 -> L_max

    # Test collar zero masking
    rad_masked = convert_dn_to_radiance(dn, l_min=l_min, l_max=l_max, mask_collar_zeros=True)
    assert rad_masked[0] == 0.0  # Preserved as collar zero despite l_min > 0
    assert pytest.approx(rad_masked[2], abs=1e-5) == 50.0

    # Test invalid endpoints
    with pytest.raises(ValueError, match="dn_max .* must be greater"):
        convert_dn_to_radiance(dn, l_min=0.0, l_max=50.0, dn_min=1000.0, dn_max=500.0)

    with pytest.raises(ValueError, match="l_max .* must be greater"):
        convert_dn_to_radiance(dn, l_min=50.0, l_max=20.0)


def test_convert_radiance_to_toa_reflectance_analytical():
    """Verify radiance to TOA reflectance calculation against analytical formulation."""
    import math

    from cloud_removal.preprocess import convert_radiance_to_toa_reflectance

    # Setup known parameters:
    # L_lambda = 26.0 mW/(cm^2*sr*um)
    # ESUN = 181.89 mW/(cm^2*um)
    # solar_zenith = 60 deg -> cos(60) = 0.5
    # d = 1.0 AU
    # Expected: (pi * 26.0 * 1.0^2) / (181.89 * 0.5) = 81.68140899 / 90.945 = 0.89814073
    radiance = np.array([26.0], dtype=np.float32)
    refl = convert_radiance_to_toa_reflectance(
        radiance=radiance,
        esun=181.89,
        solar_zenith_deg=60.0,
        earth_sun_distance_au=1.0,
    )

    expected = (math.pi * 26.0) / (181.89 * 0.5)
    assert pytest.approx(float(refl[0]), abs=1e-5) == expected

    # Test parameter validation
    with pytest.raises(ValueError, match="ESUN must be strictly positive"):
        convert_radiance_to_toa_reflectance(radiance, esun=-10.0, solar_zenith_deg=45.0)

    with pytest.raises(ValueError, match="Solar zenith angle must be in"):
        convert_radiance_to_toa_reflectance(radiance, esun=180.0, solar_zenith_deg=95.0)

    with pytest.raises(ValueError, match="Earth-Sun distance must be"):
        convert_radiance_to_toa_reflectance(
            radiance, esun=180.0, solar_zenith_deg=45.0, earth_sun_distance_au=0.0
        )


def test_prepare_liss4_fcc_model_input():
    """Verify prepare_liss4_fcc_model_input converts [B2, B3, B4] DN to [B4, B3, B2] FCC tensor."""
    import torch

    from cloud_removal.preprocess import prepare_liss4_fcc_model_input

    # 3-band DN array (3, 8, 8)
    H, W = 8, 8
    b2_dn = np.full((H, W), 200, dtype=np.uint16)  # Band 2 (Green)
    b3_dn = np.full((H, W), 300, dtype=np.uint16)  # Band 3 (Red)
    b4_dn = np.full((H, W), 400, dtype=np.uint16)  # Band 4 (NIR)

    # Set collar pixel at (0, 0)
    b2_dn[0, 0] = 0
    b3_dn[0, 0] = 0
    b4_dn[0, 0] = 0

    stacked = np.stack([b2_dn, b3_dn, b4_dn], axis=0)  # Shape (3, 8, 8)

    mock_metadata = {
        "solar_zenith_deg": 20.0,
        "earth_sun_distance_au": 1.0142,
        "calibration": {
            "BAND2": {"l_min": 0.0, "l_max": 52.0},
            "BAND3": {"l_min": 0.0, "l_max": 47.0},
            "BAND4": {"l_min": 0.0, "l_max": 31.5},
        },
    }

    tensor, valid_mask = prepare_liss4_fcc_model_input(
        stacked_dn=stacked,
        metadata=mock_metadata,
        normalize_for_model=True,
    )

    # Output tensor shape must be (1, 3, 8, 8)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, H, W)
    assert tensor.dtype == torch.float32

    # Validity mask shape (8, 8)
    assert valid_mask.shape == (H, W)
    assert not valid_mask[0, 0]  # Collar pixel is False
    assert valid_mask[1, 1]  # Non-collar pixel is True

    # Collar pixel must be 0.0 across all model channels
    assert torch.all(tensor[0, :, 0, 0] == 0.0)

    # Verify channel permuting to FCC [B4 (NIR), B3 (Red), B2 (Green)]
    # B4 (NIR) has L_max=31.5, ESUN=110.48; B3 has L_max=47.0, ESUN=156.96; B2 has L_max=52.0, ESUN=181.89
    arr = tensor[0].numpy()
    assert np.all(arr >= 0.0)
    assert np.all(arr <= 1.0)
    # Channel values should be non-zero on valid pixels
    assert np.all(arr[:, 1:, 1:] > 0.0)
