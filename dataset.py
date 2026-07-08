from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def get_emnist_dataloader(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        lambda img: torch.rot90(img, k=-1, dims=[1, 2]).flip(2) # Rotate images
    ])

    train_data = datasets.EMNIST(
        root="./data",
        split="byclass",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader