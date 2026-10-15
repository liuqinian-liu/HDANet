"""Quick CPU checks for the integrated B4 and G1 operators and fold settings."""
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from model_hdanet.MAF import MAF
from model_hdanet.csh_boundary_loss import csh_boundary_loss
from model_hdanet.g1_spatial_gate import PriorSpatialGate


def main():
    root = Path(__file__).resolve().parent
    expected = {0: (200, 2, 0.001, True),
                1: (200, 2, 0.0025, False),
                2: (300, 4, 0.0025, False)}
    for fold, settings in expected.items():
        cfg = yaml.safe_load((root / f"config/B4-G1/fold{fold}.yaml").read_text())
        train = cfg["TRAIN"]
        assert (train["epochs"], train["batch_size"], train["base_lr"],
                train["remove_aspp"]) == settings
        assert (train["manual_seed"], train["shot"], train["split"]) == (2025, 1, fold)
        assert train["csh_boundary_weight"] == 0.1
        assert train["spatial_gate_enabled"] is True
        assert train["resume"] is None and train["weight"] is None
        for split in ("train", "val"):
            assert (root / cfg["DATA"][f"{split}_list"]).is_file()

    torch.manual_seed(5)
    feature = torch.randn(2, 16, 7, 7)
    prior = torch.randn(2, 2, 7, 7)
    gate = PriorSpatialGate(0.25)
    assert torch.equal(gate(feature, prior), feature)
    gate(feature, prior).square().mean().backward()
    assert gate.output.weight.grad.abs().sum() > 0

    base = MAF(16).eval()
    enhanced = deepcopy(base)
    enhanced.spatial_gate = PriorSpatialGate(0.25)
    query = torch.randn(2, 16, 7, 7)
    support = torch.randn(2, 16, 7, 7)
    with torch.no_grad():
        assert torch.equal(base(query, support, prior),
                           enhanced(query, support, prior))

    logits = torch.randn(2, 2, 16, 16, requires_grad=True)
    labels = torch.zeros(2, 16, 16, dtype=torch.long)
    labels[:, 4:12, 4:12] = 1
    labels[:, :2] = 255
    loss = csh_boundary_loss(logits, labels)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    ignored = csh_boundary_loss(logits, torch.full_like(labels, 255))
    assert torch.isfinite(ignored) and ignored == 0
    print("B4+G1 config, zero-init, gradient, and ignored-label checks passed")


if __name__ == "__main__":
    main()
