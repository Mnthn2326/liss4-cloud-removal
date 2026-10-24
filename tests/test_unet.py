"""Unit tests for conventional U-Net architecture."""

from pathlib import Path

import pytest
import torch

from cloud_removal.models.unet import UNet


def test_unet_initialization():
    """UNet should initialize with default and custom channels."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    assert model.in_channels == 3
    assert model.out_channels == 3
    assert model.base_channels == 8
    assert model.count_parameters() > 0

    cfg = model.get_config()
    assert cfg["model_type"] == "UNet"
    assert cfg["base_channels"] == 8
    assert cfg["total_trainable_params"] == model.count_parameters()


def test_unet_forward_shape_preservation():
    """UNet must preserve exact spatial dimensions (B, C, H, W)."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    model.eval()

    # Test with batch of 64x64 patches
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 64, 64)
    assert torch.all(torch.isfinite(out)), "Model output contains non-finite values (NaN/Inf)"


def test_unet_256_resolution():
    """UNet must process full 256x256 satellite patches correctly."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    model.eval()

    x = torch.rand(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 256, 256)
    assert torch.all(torch.isfinite(out)), "Model output contains non-finite values (NaN/Inf)"


def test_unet_output_bounds_with_sigmoid():
    """Outputs must strictly lie in [0.0, 1.0] when use_sigmoid is True."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8, use_sigmoid=True)
    model.eval()

    # Pass inputs with extreme values
    x = torch.randn(2, 3, 32, 32) * 50.0
    with torch.no_grad():
        out = model(x)

    assert torch.all(torch.isfinite(out)), "Model output contains non-finite values (NaN/Inf)"
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_unet_bilinear_upsampling():
    """UNet with bilinear upsampling should maintain spatial dimension."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8, bilinear=True)
    model.eval()

    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 3, 64, 64)


def test_unet_checkpoint_save_and_reload(tmp_path: Path):
    """Saved model checkpoint must reload state dict and yield identical predictions."""
    model1 = UNet(in_channels=3, out_channels=3, base_channels=8)
    model1.eval()

    ckpt_file = tmp_path / "test_unet.pt"
    torch.save(
        {
            "model_state_dict": model1.state_dict(),
            "model_config": model1.get_config(),
        },
        ckpt_file,
    )

    checkpoint = torch.load(ckpt_file)
    model2 = UNet(in_channels=3, out_channels=3, base_channels=8)
    model2.load_state_dict(checkpoint["model_state_dict"])
    model2.eval()

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_infer_load_and_predict(tmp_path: Path):
    """load_model_from_checkpoint and predict_patch must work together seamlessly."""
    from cloud_removal.infer import load_model_from_checkpoint, predict_patch

    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    ckpt_file = tmp_path / "infer_test.pt"
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "val_mae": 0.123,
        },
        ckpt_file,
    )

    loaded_model, ckpt = load_model_from_checkpoint(ckpt_file, device=torch.device("cpu"))
    assert ckpt["epoch"] == 1
    assert ckpt["val_mae"] == 0.123

    # Test single 3D patch prediction
    x = torch.rand(3, 32, 32)
    pred = predict_patch(loaded_model, x, device=torch.device("cpu"))
    assert pred.shape == (3, 32, 32)
    assert torch.all(torch.isfinite(pred)), "Predicted patch contains non-finite values (NaN/Inf)"
    assert torch.all(pred >= 0.0) and torch.all(pred <= 1.0)


def test_unet_prior_reflectance_initialization():
    """UNet with prior_reflectance should initialize predictions near the prior mean."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8, prior_reflectance=0.107)
    model.eval()

    # Pass zeros/small noise; output should center around ~0.107
    x = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        out = model(x)

    assert pytest.approx(out.mean().item(), abs=0.03) == 0.107

    # Invalid prior reflectance should raise ValueError
    with pytest.raises(ValueError, match="prior_reflectance must be strictly between 0 and 1"):
        UNet(in_channels=3, out_channels=3, base_channels=8, prior_reflectance=-0.5)

    with pytest.raises(ValueError, match="prior_reflectance must be strictly between 0 and 1"):
        UNet(in_channels=3, out_channels=3, base_channels=8, prior_reflectance=1.5)


def test_unet_gradient_flow_all_parameters():
    """All trainable parameters must receive non-zero gradients during backpropagation."""
    model = UNet(in_channels=3, out_channels=3, base_channels=8)
    model.train()

    x = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)

    pred = model(x)
    loss = torch.nn.L1Loss()(pred, target)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} has None grad"
            assert not torch.all(p.grad == 0), f"Parameter {name} has all-zero grad"


def test_unet_fast_overfit_small_batch():
    """UNet should strictly decrease loss on a tiny fixed batch within few iterations."""
    torch.manual_seed(42)
    model = UNet(in_channels=3, out_channels=3, base_channels=8, prior_reflectance=0.107)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    x = torch.rand(2, 3, 32, 32)
    target = torch.full((2, 3, 32, 32), 0.107)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]}, final={losses[-1]}"


def test_evaluate_full_split_mock(tmp_path: Path):
    """evaluate_full_split should run evaluation and return metric dictionaries."""
    from cloud_removal.infer import evaluate_full_split

    ckpt_file = "checkpoints/best_unet.pt"
    if not Path(ckpt_file).is_file():
        pytest.skip("best_unet.pt not present")

    report = evaluate_full_split(
        checkpoint_path=ckpt_file,
        manifest_path="data/raw/sen12mscr/manifest.json",
        split="val",
        save_indices=[0],
        output_dir=tmp_path / "figures",
        device_str="cpu",
    )

    assert report["split"] == "val"
    assert report["sample_count"] == 144
    assert "macro_mae" in report["baseline_cloudy_input"]
    assert "macro_mae" in report["unet_prediction"]
    assert len(report["saved_figures"]) == 1


def test_compact_unet_shape_and_parameters():
    """Compact U-Net (base_channels=16) must have ~1.94M params and output [B, 3, 256, 256]."""
    model = UNet(
        in_channels=3, out_channels=3, base_channels=16, use_sigmoid=True, prior_reflectance=0.107
    )
    params = model.count_parameters()
    assert params == 1942611

    x = torch.rand(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 3, 256, 256)
    assert torch.all(torch.isfinite(out))
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_compact_unet_checkpoint_save_and_reload(tmp_path: Path):
    """Compact U-Net checkpoint must reload with exact parameters and identical predictions."""
    from cloud_removal.infer import load_model_from_checkpoint

    model1 = UNet(
        in_channels=3, out_channels=3, base_channels=16, use_sigmoid=True, prior_reflectance=0.107
    )
    model1.eval()
    ckpt_file = tmp_path / "compact_test.pt"

    payload = {
        "epoch": 1,
        "model_state_dict": model1.state_dict(),
        "model_config": model1.get_config(),
        "val_mae": 0.025,
    }
    torch.save(payload, ckpt_file)

    model2, reloaded_payload = load_model_from_checkpoint(ckpt_file, device=torch.device("cpu"))
    assert model2.count_parameters() == 1942611
    assert reloaded_payload["model_config"]["base_channels"] == 16

    x = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_compact_unet_one_step_training():
    """One training step must update weights and strictly reduce loss on a small batch."""
    torch.manual_seed(42)
    model = UNet(
        in_channels=3, out_channels=3, base_channels=16, use_sigmoid=True, prior_reflectance=0.107
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.L1Loss()

    x = torch.rand(2, 3, 64, 64)
    target = torch.full((2, 3, 64, 64), 0.107)

    # Initial loss
    out0 = model(x)
    loss0 = criterion(out0, target).item()

    # Step
    optimizer.zero_grad()
    loss = criterion(out0, target)
    loss.backward()
    optimizer.step()

    # Post-step loss
    out1 = model(x)
    loss1 = criterion(out1, target).item()

    assert loss1 < loss0, f"Expected loss to decrease: loss0={loss0}, loss1={loss1}"


def test_render_comparison_figure_shared_stretch(tmp_path: Path):
    """render_comparison_figure must apply shared contrast stretch and save valid PNG."""
    import numpy as np

    from cloud_removal.infer import render_comparison_figure

    # Dark surface reflectance typical of winter Sentinel-2
    cloudy = np.full((3, 32, 32), 0.35, dtype=np.float32)
    pred = np.full((3, 32, 32), 0.10, dtype=np.float32)
    clear = np.linspace(0.06, 0.14, 3 * 32 * 32, dtype=np.float32).reshape(3, 32, 32)

    cloudy_copy = cloudy.copy()
    pred_copy = pred.copy()
    clear_copy = clear.copy()

    out_file = tmp_path / "stretched.png"
    result = render_comparison_figure(
        cloudy=cloudy,
        pred=pred,
        clear=clear,
        output_path=out_file,
        title_suffix="- Test Stretch",
        stretch_percentiles=(2.0, 98.0),
    )

    assert result.is_file()
    assert result.stat().st_size > 0
    # Inputs must not be mutated in place
    np.testing.assert_array_equal(cloudy, cloudy_copy)
    np.testing.assert_array_equal(pred, pred_copy)
    np.testing.assert_array_equal(clear, clear_copy)


def test_render_comparison_figure_options(tmp_path: Path):
    """render_comparison_figure must support unstretched mode and two-panel mode."""
    import numpy as np

    from cloud_removal.infer import render_comparison_figure

    cloudy = np.full((3, 32, 32), 0.20, dtype=np.float32)
    pred = np.full((3, 32, 32), 0.10, dtype=np.float32)

    # 1. Unstretched (raw) mode
    out_raw = tmp_path / "raw.png"
    res_raw = render_comparison_figure(
        cloudy=cloudy,
        pred=pred,
        clear=pred,
        output_path=out_raw,
        stretch_percentiles=None,
    )
    assert res_raw.is_file() and res_raw.stat().st_size > 0

    # 2. Two-panel mode (clear=None)
    out_2panel = tmp_path / "two_panel.png"
    res_2panel = render_comparison_figure(
        cloudy=cloudy,
        pred=pred,
        clear=None,
        output_path=out_2panel,
    )
    assert res_2panel.is_file() and res_2panel.stat().st_size > 0


def test_run_demo(tmp_path: Path):
    """run_demo must execute end-to-end and produce valid metrics and figure."""
    from cloud_removal.demo import run_demo

    ckpt_path = Path("checkpoints/compact/best_compact_unet.pt")
    manifest_path = Path("data/raw/sen12mscr/manifest.json")
    if not ckpt_path.is_file() or not manifest_path.is_file():
        pytest.skip("Compact checkpoint or dataset manifest not present")

    out_fig = tmp_path / "demo_test.png"
    report = run_demo(
        checkpoint_path=ckpt_path,
        manifest_path=manifest_path,
        split="val",
        sample_index=0,
        output_figure=out_fig,
        device="cpu",
    )

    assert out_fig.is_file()
    assert out_fig.stat().st_size > 0
    assert report["patch_id"] == "p101"
    assert "unet_prediction_mae" in report["metrics"]
    assert report["metrics"]["unet_prediction_mae"] < 0.01


def test_unet_output_quality_gate():
    """Quality gate: verify output shapes, finiteness, and valid [0, 1] range across model variants."""
    for base_ch in [8, 16, 32]:
        model = UNet(
            in_channels=3,
            out_channels=3,
            base_channels=base_ch,
            use_sigmoid=True,
            prior_reflectance=0.107,
        )
        model.eval()
        for batch_size, size in [(1, 32), (2, 64)]:
            x = torch.rand(batch_size, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (
                batch_size,
                3,
                size,
                size,
            ), f"Shape mismatch for base_channels={base_ch}"
            assert torch.all(
                torch.isfinite(out)
            ), f"Output contains non-finite values for base_channels={base_ch}"
            assert torch.all(
                out >= 0.0
            ), f"Output contains negative values for base_channels={base_ch}"
            assert torch.all(out <= 1.0), f"Output exceeds 1.0 for base_channels={base_ch}"
