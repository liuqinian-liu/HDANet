"""G1: bounded PDM-guided AEM spatial gate from the B4+G1 experiments."""
import torch
from torch import nn


class PriorSpatialGate(nn.Module):
    """The zero-initialized residual leaves the original AEM output unchanged."""

    def __init__(self, maximum_change):
        super().__init__()
        self.maximum_change = float(maximum_change)
        self.context = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False), nn.ReLU(inplace=True))
        self.output = nn.Conv2d(16, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, feature, prior):
        adjustment = torch.tanh(self.output(self.context(prior)))
        return feature * (1.0 + self.maximum_change * adjustment)


def configure_spatial_gate(model, args):
    if not getattr(args, 'spatial_gate_enabled', False):
        return
    # Preserve the original parameter initialization and random sequence.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.manual_seed + 4103)
        for name, bound in (('maf_l', 0.15), ('maf_m', 0.25), ('maf_h', 0.15)):
            getattr(model, name).spatial_gate = PriorSpatialGate(bound)
