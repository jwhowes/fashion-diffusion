from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self):
        self.dataset = load_dataset("tomytjandra/h-and-m-fashion-caption", split="train")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((96, 96), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]["image"]).clamp(-1.0, 1.0)
