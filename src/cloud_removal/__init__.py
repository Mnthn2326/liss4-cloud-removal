"""LISS4 Cloud Removal package."""

from cloud_removal.inspect import inspect_raster, read_raster_window
from cloud_removal.patch_dataset import (
    DatasetSplits,
    LISS4PatchDataset,
    PatchConfig,
    PatchMetadata,
    PatchSample,
    extract_paired_patches,
)
from cloud_removal.sen12mscr import (
    DEFAULT_RGB_BANDS,
    DEFAULT_RGB_NAMES,
    PairedOpticalSample,
    SEN12MSCRConfig,
    SEN12MSCRDataset,
    validate_sen12mscr_subset,
)
from cloud_removal.synthetic_clouds import (
    CloudConfig,
    SyntheticCloudGenerator,
    blend_synthetic_clouds,
    generate_cloud_mask,
)
from cloud_removal.rice2 import RICE2Config, RICE2Dataset, RICE2Sample, build_rice2_manifest

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "inspect_raster",
    "read_raster_window",
    "CloudConfig",
    "generate_cloud_mask",
    "blend_synthetic_clouds",
    "SyntheticCloudGenerator",
    "PatchConfig",
    "PatchMetadata",
    "PatchSample",
    "DatasetSplits",
    "extract_paired_patches",
    "LISS4PatchDataset",
    "SEN12MSCRConfig",
    "SEN12MSCRDataset",
    "PairedOpticalSample",
    "validate_sen12mscr_subset",
    "DEFAULT_RGB_BANDS",
    "DEFAULT_RGB_NAMES",
    "RICE2Config",
    "RICE2Dataset",
    "RICE2Sample",
    "build_rice2_manifest",
]
