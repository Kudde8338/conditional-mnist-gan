import torch
from utils import compute_gradient_penalty

def training_loop(G, D, G_optimizer, D_optimizer, epochs, latent_dim, train_loader, device, n_critic=5, lambda_gp=10):

    D_loss_history = []
    G_loss_history = []

    for epoch in range(epochs):
        G.train()
        D.train()

        G_epoch_loss = 0
        D_epoch_loss = 0

        for batch_idx, (real_images, labels) in enumerate(train_loader):
            labels = labels.to(device)
            batch_size = real_images.size(0) # Calcuate the batch size
            real_images = real_images.to(device)

            # ===========================
            # STEP 1: Train Discriminator
            # ===========================
            for _ in range(n_critic):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_images = G(z, labels)

                D_real = D(real_images, labels)
                D_fake = D(fake_images.detach(), labels)

                gradient_penalty = compute_gradient_penalty(D, real_images, fake_images.detach(), labels, device)

                wasserstein_estimate = D_real.mean() - D_fake.mean()

                loss_D = -torch.mean(D_real) + torch.mean(D_fake) + lambda_gp * gradient_penalty
                
                D_optimizer.zero_grad()
                loss_D.backward()
                D_optimizer.step()

                D_epoch_loss += loss_D.item()

            # =======================
            # STEP 2: Train Generator
            # =======================
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_gen_labels = torch.randint(0, 62, (batch_size,)).to(device)
            fake_images = G(z, fake_gen_labels)
            D_fake = D(fake_images, fake_gen_labels)

            loss_G = -torch.mean(D_fake)

            G_optimizer.zero_grad()
            loss_G.backward()
            G_optimizer.step()

            G_epoch_loss += loss_G.item()

        average_D_loss = D_epoch_loss / len(train_loader)
        average_G_loss = G_epoch_loss / len(train_loader)
    
        G_loss_history.append(average_G_loss)
        D_loss_history.append(average_D_loss)

        print(f"\rEpoch [{epoch+1}/{epochs}]  W-dist: {wasserstein_estimate.item():.4f} | GP: {gradient_penalty.item():.4f}", end="")
    print()
    return G_loss_history, D_loss_history
