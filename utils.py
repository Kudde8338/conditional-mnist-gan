import matplotlib.pyplot as plt
import torch
from pathlib import Path

def show_generated_image(model, latent_dim, device, number):
    model.eval()  # Put the model into evaluation mode
    
    with torch.inference_mode():
        z = torch.randn(1, latent_dim).to(device)
        fake_gen_labels = torch.tensor([number]).to(device)
        fake_image = model(z, fake_gen_labels)
    
    # Rezise the image to 28x28 pixels
    fake_image = fake_image.view(28, 28).cpu()
    
    # gå tillbaka från [-1, 1] till [0, 1] för korrekt visning
    fake_image = (fake_image + 1) / 2
    
    plt.imshow(fake_image, cmap="gray")
    plt.axis("off")
    plt.show()
    
    model.train()

def show_loss_graph(G_loss, D_loss, loss_ratio):
    plt.figure(figsize=(8, 5))
    plt.plot(D_loss, label="Discriminator loss")
    plt.plot(G_loss, label="Generator loss")
    plt.plot(loss_ratio, label="Loss ratio")
    plt.title("Loss curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

def save_model(model_G, model_D, name, path):
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    MODEL_SAVE_PATH = save_dir / f"{name}.pth"

    torch.save({
        "G_state_dict": model_G.state_dict(),
        "D_state_dict": model_D.state_dict(),
    }, MODEL_SAVE_PATH)

    print(f"Model successfully saved to: {MODEL_SAVE_PATH}")

def load_model(model_G, model_D, path, device):
    checkpoint = torch.load(path, map_location=device)
    model_G.load_state_dict(checkpoint["G_state_dict"])
    model_D.load_state_dict(checkpoint["D_state_dict"])