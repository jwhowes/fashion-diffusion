import torch

from torch import nn

from .util import DiagonalGaussian, FiLM, CrossAttentionBlock


class GRN(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(GRN, self).__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, 1, 1, d_model))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, d_model))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + self.eps)

        return self.gamma * (x * Nx) + self.beta + x


class FiLMConvNeXtV2Block(nn.Module):
    def __init__(self, d_model, d_t, dropout=0.0, norm_eps=1e-6):
        super(FiLMConvNeXtV2Block, self).__init__()
        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)

        self.norm = FiLM(d_t, d_model, eps=norm_eps)

        self.pw_model = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            GRN(4 * d_model),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        residual = x

        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)

        x = self.pw_model(self.norm(x, t))
        x = x.permute(0, 3, 1, 2)

        return residual + self.dropout(x)


class ConvNeXtV2Block(nn.Module):
    def __init__(self, d_model, dropout=0.0, norm_eps=1e-6):
        super(ConvNeXtV2Block, self).__init__()

        self.dw_conv = nn.Conv2d(d_model, d_model, kernel_size=7, padding=3, groups=d_model)

        self.pw_model = nn.Sequential(
            nn.LayerNorm(d_model, eps=norm_eps),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            GRN(4 * d_model),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw_model(x)
        x = x.permute(0, 3, 1, 2)

        return residual + self.dropout(x)


class VAEEncoder(nn.Module):
    def __init__(
            self, in_channels, latent_channels,
            d_init=32, n_scales=4, blocks_per_scale=1, dropout=0.0
    ):
        super(VAEEncoder, self).__init__()
        scale = 1

        self.stem = nn.Conv2d(in_channels, d_init, kernel_size=3, padding=1)
        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(n_scales - 1):
            blocks = nn.ModuleList()
            for j in range(blocks_per_scale):
                blocks.append(ConvNeXtV2Block(d_init * scale, dropout=dropout))

            self.down_path.append(blocks)
            self.down_samples.append(nn.Conv2d(scale * d_init, 2 * scale * d_init, kernel_size=2, stride=2))
            scale *= 2
        
        self.mid_blocks = nn.ModuleList()
        for i in range(blocks_per_scale):
            self.mid_blocks.append(ConvNeXtV2Block(d_init * scale, dropout=dropout))

        self.head = nn.Conv2d(scale * d_init, 2 * latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for down_sample, down_blocks in zip(self.down_samples, self.down_path):
            for down_block in down_blocks:
                x = down_block(x)
            x = down_sample(x)

        for block in self.mid_blocks:
            x = block(x)

        mean, log_var = self.head(x).chunk(dim=1, chunks=2)

        return DiagonalGaussian(mean=mean, log_var=log_var)


class VAEDecoder(nn.Module):
    def __init__(
            self, in_channels, latent_channels,
            d_init=32, n_scales=4, blocks_per_scale=1, dropout=0.0
    ):
        super(VAEDecoder, self).__init__()
        scale = 2 ** n_scales
        
        self.stem = nn.Conv2d(latent_channels, scale * d_init, kernel_size=3, padding=1)
        self.mid_blocks = nn.ModuleList()
        for i in range(blocks_per_scale):
            self.mid_blocks.append(ConvNeXtV2Block(scale * d_init, dropout=dropout))

        self.up_path = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(n_scales - 1):
            self.up_samples.append(nn.ConvTranspose2d(scale * d_init, scale * d_init // 2, kernel_size=2, stride=2))
            scale //= 2
            blocks = nn.ModuleList()
            for j in range(blocks_per_scale):
                blocks.append(ConvNeXtV2Block(scale * d_init, dropout=dropout))

            self.up_path.append(blocks)

        self.head = nn.Conv2d(d_init, in_channels, kernel_size=1)

    def forward(self, z):
        z = self.stem(z)
        for block in self.mid_blocks:
            z = block(z)

        for up_sample, up_blocks in zip(self.up_samples, self.up_path):
            z = up_sample(z)
            for up_block in up_blocks:
                z = up_block(z)

        return self.head(z)


class VAE(nn.Module):
    def __init__(
            self, in_channels, latent_channels,
            d_init=32, n_scales=4, blocks_per_scale=1, dropout=0.0
    ):
        super(VAE, self).__init__()
        # Total blocks = 2 * n_scales * blocks_per_scale
        # Downscale size = 2 ** (n_scales - 1)
        self.encode = VAEEncoder(in_channels, latent_channels, d_init, n_scales, blocks_per_scale, dropout)
        self.decode = VAEDecoder(in_channels, latent_channels, d_init, n_scales, blocks_per_scale, dropout)

    def forward(self, x, return_dist=True):
        dist = self.encode(x)
        z = dist.sample()
        recon = self.decode(z)

        if return_dist:
            return recon, dist

        return recon


class ConditionalConvNeXtDiffuserBlock(nn.Module):
    def __init__(self, d_model, d_context, d_t, n_heads, resid_dropout=0.0, attn_dropout=0.0, norm_eps=1e-6):
        super(ConditionalConvNeXtDiffuserBlock, self).__init__()
        self.attn = CrossAttentionBlock(d_model, d_context, n_heads)
        self.attn_norm = FiLM(d_t, d_model, eps=norm_eps)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.ffn = FiLMConvNeXtV2Block(d_model, d_t, resid_dropout, norm_eps)

    def forward(self, x, t, context, attention_mask=None):
        x = x.permute(0, 2, 3, 1)
        x = x + self.attn_dropout(
            self.attn(
                self.attn_norm(x, t), context, attention_mask=attention_mask
            )
        )

        x = x.permute(0, 3, 1, 2)
        return self.ffn(x, t)


class ConditionalUNet(nn.Module):
    def __init__(
            self, in_channels, d_init, d_context, d_t, n_heads_init,
            n_scales=3, blocks_per_scale=1,
            resid_dropout=0.0, attn_dropout=0.0, norm_eps=1e-6
    ):
        super(ConditionalUNet, self).__init__()
        self.stem = nn.Conv2d(in_channels, d_init, kernel_size=7, padding=3)

        scale = 1
        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(n_scales - 1):
            blocks = nn.ModuleList()
            for j in range(blocks_per_scale):
                blocks.append(
                    ConditionalConvNeXtDiffuserBlock(
                        d_init * scale, d_context, d_t, n_heads_init * scale, resid_dropout, attn_dropout, norm_eps
                    )
                )
            self.down_path.append(blocks)
            self.down_samples.append(nn.Conv2d(d_init * scale, d_init * 2 * scale, kernel_size=2, stride=2))
            scale *= 2

        self.mid_blocks = nn.ModuleList()
        for i in range(blocks_per_scale):
            self.mid_blocks.append(
                ConditionalConvNeXtDiffuserBlock(
                    d_init * scale, d_context, d_t, n_heads_init * scale, resid_dropout, attn_dropout, norm_eps
                )
            )

        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_path = nn.ModuleList()
        for i in range(n_scales - 1):
            self.up_samples.append(nn.ConvTranspose2d(d_init * scale, d_init * scale // 2, kernel_size=2, stride=2))
            self.up_combines.append(nn.Conv2d(d_init * scale, d_init * scale // 2, kernel_size=7, padding=3))

            scale //= 2
            blocks = nn.ModuleList()
            for j in range(blocks_per_scale):
                blocks.append(
                    ConditionalConvNeXtDiffuserBlock(
                        d_init * scale, d_context, d_t, n_heads_init * scale, resid_dropout, attn_dropout, norm_eps
                    )
                )
            self.up_path.append(blocks)

        self.head = nn.Conv2d(d_init, in_channels, kernel_size=7, padding=3)

    def forward(self, x, t, context, attention_mask=None):
        x = self.stem(x)

        down_acts = []
        for down_blocks, down_sample in zip(self.down_path, self.down_samples):
            for block in down_blocks:
                x = block(x, t, context, attention_mask)
            down_acts.append(x)
            x = down_sample(x)

        for block in self.mid_blocks:
            x = block(x, t, context, attention_mask)

        for up_sample, up_combine, up_blocks, act in zip(self.up_samples, self.up_combines, self.up_path, down_acts[::-1]):
            x = up_sample(x)
            x = up_combine(
                torch.concat((
                    x,
                    act
                ), dim=1)
            )
            for block in up_blocks:
                x = block(x, t, context, attention_mask)

        return self.head(x)
