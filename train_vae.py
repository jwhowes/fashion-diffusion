import torch.nn.functional as F
import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.model import VAE, init_params, Classifier
from src.data import ImageDataset


def train(model, discriminator, dataloader, kl_weight=1e-6, adv_weight=0.5):
    num_epochs = 5
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=5e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    model, discriminator, opt, disc_opt, lr_scheduler = accelerator.prepare(
        model, discriminator, opt, disc_opt, lr_scheduler
    )

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, image in enumerate(dataloader):
            recon, dist = model(image)

            real_patches = (
                image
                .unfold(2, 16, 16)
                .unfold(3, 16, 16)
                .flatten(2, 3)
                .transpose(1, 2)
                .reshape(-1, 3, 16, 16)
            )
            fake_patches = (
                recon
                .unfold(2, 16, 16)
                .unfold(3, 16, 16)
                .flatten(2, 3)
                .transpose(1, 2)
                .reshape(-1, 3, 16, 16)
            )

            adv_pred = discriminator(fake_patches).squeeze()

            opt.zero_grad()
            recon_loss = F.mse_loss(recon, image)
            kl_loss = dist.kl
            adv_loss = F.binary_cross_entropy_with_logits(adv_pred, torch.ones_like(adv_pred))
            loss = recon_loss + kl_weight * kl_loss + adv_weight * adv_loss

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            lr_scheduler.step()

            real_pred = discriminator(real_patches).squeeze()
            fake_pred = discriminator(fake_patches.detach()).squeeze()

            disc_opt.zero_grad()
            fake_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            real_loss = F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred))
            disc_loss = fake_loss + real_loss
            accelerator.backward(disc_loss)
            accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)
            disc_opt.step()

            if i % 10 == 0:
                print(f"\t{i} / {len(dataloader)} iters."
                      f"\t\tRecon Loss: {recon_loss.item():.3f}"
                      f"\t\tKL Loss: {kl_loss.item():.3f}"
                      f"\t\tAdv Loss: {adv_loss.item():.3f}")

        torch.save(
            accelerator.get_state_dict(model, unwrap=True),
            f"checkpoints/checkpoint_{epoch + 1:03}.pt"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--adv-weight", type=float, default=0.5)
    parser.add_argument("--d-init", type=int, default=64)
    parser.add_argument("--n-scales", type=int, default=3)
    parser.add_argument("--blocks-per-scale", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = VAE(
        3, 4,
        d_init=args.d_init,
        n_scales=args.n_scales,
        blocks_per_scale=args.blocks_per_scale,
        dropout=args.dropout
    )
    init_params(model)

    discriminator = Classifier(
        3, 1,
        d_init=args.d_init // 2,
        n_scales=args.n_scales - 1,
        blocks_per_scale=1,
        dropout=0.0
    )
    init_params(discriminator)

    dataset = ImageDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    train(model,discriminator, dataloader, kl_weight=args.kl_weight, adv_weight=args.adv_weight)
