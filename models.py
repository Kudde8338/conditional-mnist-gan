import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_dim: int = 28*28, num_classes: int = 10):
        super().__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, img_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([z, label_input], dim=1)
        return self.model(x)
    
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28, num_classes: int = 10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(img_dim + num_classes, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        label_input = self.label_emb(labels)
        x = torch.cat([img, label_input], dim=1)
        return self.model(x)