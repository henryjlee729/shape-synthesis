import torch
import torch.optim as optim
import numpy as np
import os
from vae import PointCloudVAE
from torch.utils.data import Dataset, DataLoader

# Load data
class ChairPointCloudDataset(Dataset):
    def __init__(self, points_dir, num_points=2048):
        self.clouds = []
        for filename in sorted(os.listdir(points_dir)):
            filepath = os.path.join(points_dir, filename)
            points = np.loadtxt(filepath)
            idx = np.random.choice(len(points), num_points, replace=len(points) < num_points)
            self.clouds.append(points[idx])
        self.clouds = np.array(self.clouds, dtype=np.float32)

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        points = self.clouds[idx].copy()

        centroid = np.mean(points, axis=0)
        points = points - centroid

        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale

        return torch.FloatTensor(points)

def chamfer_distance(pc1, pc2):
    # pc1, pc2: (batch, num_points, 3)
    dist = torch.cdist(pc1, pc2, p=2) ** 2
    min_dist_pc1 = dist.min(dim=2)[0]
    min_dist_pc2 = dist.min(dim=1)[0]
    return min_dist_pc1.mean() + min_dist_pc2.mean()

def vae_loss(recon, original, mu, logvar):
    # Reconstruction loss for unordered point clouds
    recon_loss = chamfer_distance(recon, original)

    # KL divergence: how far is the latent distribution from a standard normal
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return recon_loss + 0.0001 * kl_loss, recon_loss, kl_loss

# Training
def train():
    # Use gpu for training (works for mac mps and nvidia cuda)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = ChairPointCloudDataset("shapenet_data/PartAnnotation/03001627/points")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Loaded {len(dataset)} chair point clouds")

    # Create model and optimizer
    model = PointCloudVAE(num_points=2048, latent_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch in dataloader:
            batch = batch.to(device)

            # Forward pass
            recon, mu, logvar = model(batch)

            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss(recon, batch, mu, logvar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

        # Print progress every 10 epochs
        num_batches = len(dataloader)
        #if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | "
                f"Loss: {total_loss/num_batches:.7f} | "
                f"Recon: {total_recon/num_batches:.7f} | "
                f"KL: {total_kl/num_batches:.7f}")

    # Save the trained model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/vae_chair.pth")
    print("Model saved to checkpoints/vae_chair.pth")

if __name__ == "__main__":
    train()