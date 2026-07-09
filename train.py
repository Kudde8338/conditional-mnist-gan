from dataset import get_emnist_dataloader
from models import Generator, Discriminator
from utils import show_generated_image, show_loss_graph, save_model, load_model, char_to_label
from loops import training_loop

import torch
import torchvision
from pathlib import Path

train_loader = get_emnist_dataloader()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"You are currently running code on: {device}")
print(f"You are currently running PyTorch version: {torch.__version__}")
print(f"You are currently running TorchVision version: {torchvision.__version__}")

MODEL_FOLDER = Path("models/")

# --- Hyperparameters ---
latent_dim = 100
lr_G = 0.0002
lr_D = 0.0001

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
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(0.5, 0.999))

# --- Train the model ---
should_save = "n"
should_train = input("Do you wish to train this model (Y/n): ").strip().lower()
if should_train != "n":
    epochs = int(input("How many epochs: "))
    G_loss_history, D_loss_history = training_loop(G, D, G_optimizer, D_optimizer, epochs, latent_dim, train_loader, device)
    show_loss_graph(G_loss_history, D_loss_history)
    should_save = input("Do you wish to save this model (Y/n): ").strip().lower()

# --- Save the model
if should_save != "n":
    model_name = input("What do you wish to name your model?\n")
    save_model(model_G=G, model_D=D, name=model_name, path=MODEL_FOLDER)


character_to_generate = input("Give me a character to generate: ")
label = char_to_label(character_to_generate)
show_generated_image(G, latent_dim, device, label)