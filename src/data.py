import torch

from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from random import random
from transformers import BatchEncoding


class ImageDataset(Dataset):
    def __init__(self, image_size=96):
        self.dataset = load_dataset("tomytjandra/h-and-m-fashion-caption", split="train")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]["image"]).clamp(-1.0, 1.0)


class ConditionalImageDataset(ImageDataset):
    def __init__(self, tokenizer, *args, p_uncond=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.p_uncond = p_uncond

    def collate(self, batch):
        image, text = zip(*batch)

        text_encoding = self.tokenizer(text, padding=True, return_tensors="pt")

        return BatchEncoding({
            "image": torch.stack(image, dim=0),
            "tokens": text_encoding["input_ids"],
            "attention_mask": torch.zeros(text_encoding["attention_mask"].shape).masked_fill(
                ~(text_encoding["attention_mask"].to(torch.bool)), float("-inf")
            )
        })

    def __getitem__(self, idx):
        data = self.dataset[idx]

        image = self.transform(data["image"]).clamp(-1.0, 1.0)

        if random() < self.p_uncond:
            text = ""
        else:
            text = data["text"]

        return image, text
