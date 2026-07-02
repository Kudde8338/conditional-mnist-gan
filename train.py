from dataset import get_mnist_dataloader
from models import Generator, Discriminator
from utils import show_generated_image, show_loss_graph, save_model, load_model
from loops import training_loop
import torch
import torchvision
import torch.nn as nn
from pathlib import Path

train_loader = get_mnist_dataloader()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"You are currently running code on: {device}")
print(f"You are currently running PyTorch version: {torch.__version__}")
print(f"You are currently running TorchVision version: {torchvision.__version__}")

MODEL_FOLDER = Path("models/")

# --- Hyperparameters ---
latent_dim = 100
lr = 0.0002

# --- Model creation ---
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)

# --- Load existing weights ---
should_load = input("Do you wish to load a model (Y/n): ").strip().lower()
if should_load != "n":
    model_files = list(MODEL_FOLDER.glob("*.pth"))

    if not model_files:
        print("No Saved models found.")
    else:
        for number, name in enumerate(model_files):
            print(f"[{number}] | {name.name}")

        selected_index = int(input("Please select a model: "))
        load_model(G, D, model_files[selected_index], device)

# --- Initiate loss functions and optimizers ---
loss_fn = nn.BCELoss()
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# --- Train the model ---
should_save = "n"
should_train = input("Do you wish to train this model (Y/n): ").strip().lower()
if should_train != "n":
    epochs = int(input("How many epochs: "))
    G_loss_history, D_loss_history, loss_ratio_history = training_loop(G, D, G_optimizer, D_optimizer, loss_fn, epochs, latent_dim, train_loader, device)
    show_loss_graph(G_loss_history, D_loss_history, loss_ratio_history)
    should_save = input("Do you wish to save this model (Y/n): ").strip().lower()

# --- Save the model
if should_save != "n":
    model_name = input("What do you wish to name your model?\n")
    save_model(model_G=G, model_D=D, name=model_name, path=MODEL_FOLDER)


number_to_generate = int(input("Give me a number (0-9) to generate: "))
show_generated_image(G, latent_dim, device, number_to_generate)