import torch

from torch import nn

from .util import DiagonalGaussian


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
            d_init=32, n_scales=3, blocks_per_scale=1, dropout=0.0
    ):
        super(VAEEncoder, self).__init__()
        scale = 1

        self.stem = nn.Conv2d(in_channels, d_init, kernel_size=3, padding=1)
        self.down_path = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(n_scales):
            blocks = nn.ModuleList()
            for j in range(blocks_per_scale):
                blocks.append(ConvNeXtV2Block(d_init * scale, dropout=dropout))

            self.down_path.append(blocks)
            self.down_samples.append(nn.Conv2d(scale * d_init, 2 * scale * d_init, kernel_size=2, stride=2))
            scale *= 2

        self.head = nn.Conv2d(scale * d_init, 2 * latent_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        for down_sample, down_blocks in zip(self.down_samples, self.down_path):
            for down_block in down_blocks:
                x = down_block(x)
            x = down_sample(x)

        mean, log_var = self.head(x).chunk(dim=1, chunks=2)

        return DiagonalGaussian(mean=mean, log_var=log_var)


class VAEDecoder(nn.Module):
    def __init__(
            self, in_channels, latent_channels,
            d_init=32, n_scales=3, blocks_per_scale=1, dropout=0.0
    ):
        super(VAEDecoder, self).__init__()
        scale = 2 ** n_scales

        stem = [nn.Conv2d(latent_channels, scale * d_init, kernel_size=3, padding=1)] + [
            ConvNeXtV2Block(scale * d_init, dropout=dropout) for _ in range(blocks_per_scale)
        ]
        self.stem = nn.Sequential(*stem)
        self.up_path = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(n_scales):
            self.up_samples.append(nn.ConvTranspose2d(scale * d_init, scale * d_init // 2, kernel_size=2, stride=2))
            scale //= 2
            blocks = nn.ModuleList()
            for j in range(blocks_per_scale):
                blocks.append(ConvNeXtV2Block(scale * d_init, dropout=dropout))

            self.up_path.append(blocks)

        self.head = nn.Conv2d(d_init, in_channels, kernel_size=1)

    def forward(self, z):
        z = self.stem(z)

        for up_sample, up_blocks in zip(self.up_samples, self.up_path):
            z = up_sample(z)
            for up_block in up_blocks:
                z = up_block(z)

        return self.head(z)


class VAE(nn.Module):
    def __init__(
            self, in_channels, latent_channels,
            d_init=32, n_scales=3, blocks_per_scale=1, dropout=0.0
    ):
        super(VAE, self).__init__()
        self.encode = VAEEncoder(in_channels, latent_channels, d_init, n_scales, blocks_per_scale, dropout)
        self.decode = VAEDecoder(in_channels, latent_channels, d_init, n_scales, blocks_per_scale, dropout)

    def forward(self, x, return_dist=True):
        dist = self.encode(x)
        z = dist.sample()
        recon = self.decode(z)

        if return_dist:
            return recon, dist

        return recon
