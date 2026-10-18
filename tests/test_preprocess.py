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
    stack_bands_and_generate_mask,
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

    with rasterio.open(b2_path, "w", driver="GTiff", width=32, height=32, count=1, dtype=np.uint16, crs=crs, transform=t) as dst:
        dst.write(np.ones((32, 32), dtype=np.uint16), 1)
    # B3 has width 30 instead of 32
    with rasterio.open(b3_path, "w", driver="GTiff", width=30, height=32, count=1, dtype=np.uint16, crs=crs, transform=t) as dst:
        dst.write(np.ones((32, 30), dtype=np.uint16), 1)
    with rasterio.open(b4_path, "w", driver="GTiff", width=32, height=32, count=1, dtype=np.uint16, crs=crs, transform=t) as dst:
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

    with rasterio.open(b2_path, "w", driver="GTiff", width=16, height=16, count=1, dtype=np.uint16, crs=crs, transform=t1) as dst:
        dst.write(np.ones((16, 16), dtype=np.uint16), 1)
    with rasterio.open(b3_path, "w", driver="GTiff", width=16, height=16, count=1, dtype=np.uint16, crs=crs, transform=t2) as dst:
        dst.write(np.ones((16, 16), dtype=np.uint16), 1)
    with rasterio.open(b4_path, "w", driver="GTiff", width=16, height=16, count=1, dtype=np.uint16, crs=crs, transform=t1) as dst:
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
    b2_data = np.array([[0, 100, 0,   0,   100, 100, 0,   100]], dtype=np.uint16)
    b3_data = np.array([[0, 0,   200, 0,   200, 0,   200, 200]], dtype=np.uint16)
    b4_data = np.array([[0, 0,   0,   300, 0,   300, 300, 300]], dtype=np.uint16)

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

