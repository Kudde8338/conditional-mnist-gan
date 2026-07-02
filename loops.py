import torch

def training_loop(G, D, G_optimizer, D_optimizer, loss_fn, epochs, latent_dim, train_loader, device):

    D_loss_history = []
    G_loss_history = []
    loss_ratio_history = []

    for epoch in range(epochs):
        G.train()
        D.train()

        G_epoch_loss = 0
        D_epoch_loss = 0

        for batch_idx, (real_images, labels) in enumerate(train_loader):
            labels = labels.to(device)

            batch_size = real_images.size(0) # Calcuate the batch size
            real_images = real_images.view(batch_size, -1).to(device) # Flatten to [batch, 784]

            # Labels for loss
            real_labels = torch.ones(batch_size, 1).to(device)   # Real = 1
            fake_labels = torch.zeros(batch_size, 1).to(device)  # Fake = 0

            # Generate fake-images from noise
            fake_gen_labels = torch.randint(0, 10, (batch_size,)).to(device)
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = G(z, fake_gen_labels)

            # ===========================
            # STEP 1: Train Discriminator
            # ===========================

            # Forward pass on real images
            D_real_pred = D(real_images, labels)

            # Calculate the loss
            loss_D_real = loss_fn(D_real_pred, real_labels)

            # Optimizer zero grad
            D_optimizer.zero_grad()

            # Forward pass for fake images
            D_fake_pred = D(fake_images.detach(), fake_gen_labels)
        
            # Calculate the loss
            loss_D_fake = loss_fn(D_fake_pred, fake_labels)

            # Calculate the total loss
            loss_D = loss_D_real + loss_D_fake
            D_epoch_loss += loss_D.item()

            # Loss backwards
            loss_D.backward()

            # Step the optimizer
            D_optimizer.step()

            # =======================
            # STEP 2: Train Generator
            # =======================

            # Forward pass on the fake images
            D_pred = D(fake_images, fake_gen_labels)

            # Calculate the loss
            loss_G = loss_fn(D_pred, real_labels)
            G_epoch_loss += loss_G.item()

            # Optimizer zero grad
            G_optimizer.zero_grad()

            # Loss backwards
            loss_G.backward()

            # Step the optimizer
            G_optimizer.step()

        average_D_loss = D_epoch_loss / len(train_loader)
        average_G_loss = G_epoch_loss / len(train_loader)
        loss_ratio = average_G_loss/average_D_loss
    
        G_loss_history.append(average_G_loss)
        D_loss_history.append(average_D_loss)
        loss_ratio_history.append(loss_ratio)

        print(f"\rEpoch [{epoch+1}/{epochs}]  Loss D: {loss_D.item():.4f}  Loss G: {loss_G.item():.4f}  Loss ratio: {loss_ratio:.4f}", end="")

    return G_loss_history, D_loss_history, loss_ratio_history
