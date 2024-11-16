import torch.nn.functional as F
import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.model import VAE, init_params
from src.data import ImageDataset


def train(model, dataloader, kl_weight=1e-6):
    num_epochs = 5
    opt = torch.optim.Adam(model.parameters(), lr=5e-5)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    model, dataloader, opt, lr_scheduler = accelerator.prepare(model, dataloader, opt, lr_scheduler)

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, image in enumerate(dataloader):
            recon, dist = model(image)

            recon_loss = F.mse_loss(recon, image)
            kl_loss = dist.kl
            loss = recon_loss + kl_weight * kl_loss

            accelerator.backward(loss)
            opt.step()
            lr_scheduler.step()

            if i % 10 == 0:
                print(f"\t{i} / {len(dataloader)} iters."
                      f"\t\tRecon Loss: {recon_loss.item():.4f}"
                      f"\t\tKL Loss: {kl_loss.item():.4f}")

        torch.save(
            accelerator.get_state_dict(model, unwrap=True),
            f"checkpoints/checkpoint_{epoch + 1:03}.pt"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--d-init", type=int, default=32)
    parser.add_argument("--n-scales", type=int, default=3)
    parser.add_argument("--blocks-per-scale", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    model = VAE(
        3, 4,
        d_init=args.d_init,
        n_scales=args.n_scales,
        blocks_per_scale=args.blocks_per_scale,
        dropout=args.dropout
    )
    init_params(model)

    dataset = ImageDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )

    train(model, dataloader, kl_weight=args.kl_weight)
