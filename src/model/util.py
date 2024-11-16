import torch
import torch.nn.functional as F

from torch import nn
from collections import OrderedDict
from dataclasses import dataclass
from math import sqrt


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_size=None):
        super(SwiGLU, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.gate_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.hidden_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x):
        return self.out_proj(
            F.silu(self.gate_proj(x)) * self.hidden_proj(x)
        )


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, theta_base=10000):
        super(SinusoidalEmbedding, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1 / (
                theta_base ** (2 * torch.arange(d_model // 2) / d_model)
            ),
            persistent=False
        )

    def forward(self, x):
        x = x.view(-1, 1)
        B = x.shape[0]
        freqs = x * self.theta

        return torch.stack((
            torch.sin(freqs),
            torch.cos(freqs)
        ), dim=-1).view(B, -1)


class FiLM(nn.Module):
    def __init__(self, d_t, d_model, *args, **kwargs):
        super(FiLM, self).__init__()
        self.gamma = nn.Linear(d_t, d_model)
        self.beta = nn.Linear(d_t, d_model)

        self.norm = nn.LayerNorm(d_model, *args, elementwise_affine=False, **kwargs)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, 1, 1, -1)
        b = self.beta(t).view(B, 1, 1, -1)

        return g * self.norm(x) + b


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, d_context, n_heads):
        super(CrossAttentionBlock, self).__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_context, d_model, bias=False)
        self.W_v = nn.Linear(d_context, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_kv, attention_mask=None):
        """
        :param x_q: FloatTensor B x h x w x d_model
        :param x_kv: FloatTensor B x L x d_context
        :param attention_mask: FloatTensor (B x L | B x (h * w) x L)
        :return: FloatTensor B x d_model x h w
        """
        B, h, w, _ = x_q.shape
        L = x_kv.shape[1]

        q = self.W_q(x_q).view(B, h * w, self.n_heads, -1).transpose(1, 2)
        k = self.W_k(x_kv).view(B, L, self.n_heads, -1).transpose(1, 2)
        v = self.W_v(x_kv).view(B, L, self.n_heads, -1).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask.view(B, 1, -1, L)

        x = F.softmax(attn, dim=-1) @ v

        return self.W_o(
            x.transpose(1, 2).contiguous().view(B, h, w, -1)
        )


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
