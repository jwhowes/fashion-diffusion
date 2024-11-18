import torch

from torch import nn

from .util import Attention, FiLM, SwiGLU


class ConditionalTransformerDecoderBlock(nn.Module):
    def __init__(self, d_t, d_model, d_context, n_heads, attn_dropout=0.0, resid_dropout=0.0, norm_eps=1e-6):
        super(ConditionalTransformerDecoderBlock, self).__init__()

        self.self_attn = Attention(d_model, n_heads)
        self.self_attn_norm = FiLM(d_t, d_model, eps=norm_eps)

        self.cross_attn = Attention(d_model, n_heads, d_context=d_context)
        self.cross_attn_norm = FiLM(d_t, d_model, eps=norm_eps)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.ffn = SwiGLU(d_model)
        self.ffn_norm = FiLM(d_t, d_model, eps=norm_eps)
        self.ffn_dropout = nn.Dropout(resid_dropout)

    def forward(self, x, t, context, attention_mask=None):
        x = x + self.attn_dropout(self.self_attn(
            self.self_attn_norm(x, t)
        ))

        x = x + self.attn_dropout(self.cross_attn(
            self.cross_attn_norm(x, t), context, attention_mask
        ))

        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x, t)))
