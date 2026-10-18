"""Setup verification tests for the LISS4 Cloud Removal project."""

import importlib


def test_package_import():
    """Verify that the cloud_removal package can be imported and has a valid version."""
    import cloud_removal

    assert hasattr(cloud_removal, "__version__")
    assert cloud_removal.__version__ == "0.1.0"


def test_core_dependencies_import():
    """Verify that all core technology stack dependencies can be imported successfully."""
    core_packages = ["torch", "dvc", "mlflow", "rasterio", "numpy", "pytest"]
    for pkg in core_packages:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Failed to import core dependency: {pkg}"
