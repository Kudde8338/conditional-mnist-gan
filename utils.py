import matplotlib.pyplot as plt
import torch
from pathlib import Path
import string
import torch.autograd as autograd

def show_generated_image(model, latent_dim, device, number, z=None, block=True):
    model.eval()  # Put the model into evaluation mode
    
    with torch.inference_mode():
        if z is None:
            z = torch.randn(1, latent_dim).to(device)
        fake_gen_labels = torch.tensor([number]).to(device)
        fake_image = model(z, fake_gen_labels)
    
    # Rezise the image to 28x28 pixels
    fake_image = fake_image.view(28, 28).cpu()
    
    # gå tillbaka från [-1, 1] till [0, 1] för korrekt visning
    fake_image = (fake_image + 1) / 2
    
    plt.figure()
    plt.imshow(fake_image, cmap="gray")
    plt.axis("off")
    plt.show(block=block)
    
    model.train()

def show_loss_graph(G_loss, D_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(D_loss, label="Discriminator loss")
    plt.plot(G_loss, label="Generator loss")
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

def char_to_label(char: str = "1"):
    EMNIST_CLASSES = list(string.digits) + list(string.ascii_uppercase) + list(string.ascii_lowercase)
    return EMNIST_CLASSES.index(char)

def compute_gradient_penalty(D, real_images, fake_images, labels, device):
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolera mellan riktiga och falska bilder
    interpolates = (epsilon * real_images + (1 - epsilon) * fake_images).requires_grad_(True)
    D_interpolates = D(interpolates, labels)
    
    gradients = autograd.grad(
        outputs=D_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(D_interpolates).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty