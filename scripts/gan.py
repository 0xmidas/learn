import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from modules.gan import Discriminator, Generator


def visualize_examples(G, latent_dim, epoch, device):
    G.eval()
    with torch.no_grad():
        z = torch.randn(16, latent_dim, device=device)
        generated_images = G(z)

        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            img = generated_images[i, 0].cpu()
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        plt.suptitle(f"Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"generated_epoch_{epoch}.png", dpi=100)
        plt.close()
        print(f"Saved generated_epoch_{epoch}.png")
    G.train()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Normalize to [-1, 1] to match tanh output
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2, drop_last=True
    )

    latent_dim = 128
    G = Generator(latent_dim=latent_dim).to(device)
    D = Discriminator().to(device)

    # Lower lr for D, higher for G to help G catch up
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))

    criterion = torch.nn.BCEWithLogitsLoss()

    # Fixed noise for visualization consistency
    fixed_noise = torch.randn(16, latent_dim, device=device)

    n_epochs = 64
    for epoch in range(n_epochs):
        d_losses = []
        g_losses = []

        for batch_idx, (real_images, _) in enumerate(trainloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Labels with one-sided smoothing (only real labels)
            real_labels = torch.ones(batch_size, 1, device=device) * 0.9
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Instance noise that decays over training
            noise_factor = max(0, 0.1 * (1 - epoch / n_epochs))
            if noise_factor > 0:
                real_images = real_images + noise_factor * torch.randn_like(real_images)

            # ============================
            # Train Discriminator
            # ============================
            d_optimizer.zero_grad()

            # Real images
            real_pred = D(real_images)
            d_loss_real = criterion(real_pred, real_labels)

            # Fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = G(z)
            fake_pred = D(fake_images.detach())
            d_loss_fake = criterion(fake_pred, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()

            # ============================
            # Train Generator (2x per D step to help G keep up)
            # ============================
            for _ in range(2):
                g_optimizer.zero_grad()

                z = torch.randn(batch_size, latent_dim, device=device)
                fake_images = G(z)
                fake_pred = D(fake_images)

                # G wants D to think fakes are real (use 1.0, not smoothed)
                g_loss = criterion(fake_pred, torch.ones(batch_size, 1, device=device))
                g_loss.backward()
                g_optimizer.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if batch_idx % 100 == 0:
                # Calculate D accuracy for monitoring
                with torch.no_grad():
                    real_acc = (torch.sigmoid(real_pred) > 0.5).float().mean()
                    fake_acc = (torch.sigmoid(fake_pred) < 0.5).float().mean()

                print(
                    f"[Epoch {epoch}/{n_epochs}] [Batch {batch_idx}] "
                    f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                    f"D_real: {real_acc:.2f} D_fake: {fake_acc:.2f}"
                )

        # Epoch summary
        print(
            f"Epoch {epoch} avg - D_loss: {sum(d_losses)/len(d_losses):.4f} "
            f"G_loss: {sum(g_losses)/len(g_losses):.4f}"
        )

        # Visualize every 2 epochs
        if epoch % 2 == 0:
            visualize_examples(G, latent_dim, epoch, device)

    # Save final model
    torch.save({
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
    }, 'gan_mnist.pt')
    print("Saved model to gan_mnist.pt")


if __name__ == "__main__":
    main()
