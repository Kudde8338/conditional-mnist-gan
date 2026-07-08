import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, feature_maps: int = 64, img_channels: int = 1, num_classes: int = 62):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, feature_maps*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps*4, feature_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps*2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, img_channels, kernel_size=4, stride=2, padding=3, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_map = self.label_emb(labels)
        label_map = label_map.view(label_map.size(0), -1, 1, 1)
        z = z.view(z.size(0), -1, 1, 1)
        x = torch.cat([z, label_map], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 64, img_channels: int = 1, num_classes: int = 62, img_size: int = 28):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)

        self.model = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_maps, kernel_size=4, stride=2, padding=3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps*2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*2, feature_maps*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps*4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps*4, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )
    
    def forward(self, img, labels):
        label_map = self.label_emb(labels)
        label_map = label_map.view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([img, label_map], dim=1)
        out = self.model(x)
        return out.view(-1, 1)