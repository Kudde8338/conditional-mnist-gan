from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloader(batch_size: int = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5,))
    ])

    train_data = datasets.MNIST(
        root="./data",
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