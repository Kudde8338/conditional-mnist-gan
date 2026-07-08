from utils import show_generated_image, char_to_label, load_model
from models import Generator, Discriminator
from pathlib import Path
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# The amount of random noise
latent_dim = 100

# Initiate models (D is reqired to load the models later on)
G = Generator(latent_dim=latent_dim).to(device)
D = Discriminator().to(device)

MODEL_FOLDER = Path("models/")
model_files = list(MODEL_FOLDER.glob("*.pth"))

if not model_files:
    print("No Saved models found.")
else:
    for number, name in enumerate(model_files):
        print(f"[{number}] | {name.name}")

    selected_index = int(input("Please select a model: "))
    load_model(G, D, model_files[selected_index], device)

while True:
    character_to_generate = input("Give me a character to generate: ")
    label = char_to_label(character_to_generate)
    show_generated_image(G, latent_dim, device, label, block=False)