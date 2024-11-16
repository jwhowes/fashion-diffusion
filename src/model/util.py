import torch

from torch import nn
from collections import OrderedDict
from dataclasses import dataclass


class TimeConditionalLayerNorm(nn.Module):
    def __init__(self, d_t, d_model, *args, **kwargs):
        super(TimeConditionalLayerNorm, self).__init__()
        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

        self.norm = nn.LayerNorm(d_model, *args, elementwise_affine=False, **kwargs)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, 1, 1, -1)
        b = self.beta(t).view(B, 1, 1, -1)

        return g * self.norm(x) + b


@dataclass
class DiagonalGaussian(OrderedDict):
    mean: torch.FloatTensor
    log_var: torch.FloatTensor

    def sample(self):
        return torch.randn_like(self.mean) * (0.5 * self.log_var).exp() + self.mean

    @property
    def kl(self):
        return 0.5 * (
                self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var
        ).sum((1, 2, 3)).mean()
