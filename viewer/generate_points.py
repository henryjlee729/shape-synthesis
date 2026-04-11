import torch
import numpy as np
import os
from vae import PointCloudVAE

def generate_chairs(num_chairs=1):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load trained model
    model = PointCloudVAE(num_points=2048, latent_dim=128).to(device)
    model.load_state_dict(torch.load("checkpoints/vae_chair.pth", map_location=device))
    model.eval()

    if os.path.exists("test"):
        for f in os.listdir("test"):
            os.remove(os.path.join("test", f))

    # Generate chairs from random latent vectors
    os.makedirs("test", exist_ok=True)
    with torch.no_grad():
        for i in range(num_chairs):
            z = torch.randn(1, 128).to(device)
            points = model.decode(z).cpu().numpy()[0]  # (2048, 3)
            np.savetxt(f"test/chair_{i}.pts", points, fmt="%.6f")
            print(f"Saved chair_{i}.pts | range: [{points.min():.3f}, {points.max():.3f}]")

if __name__ == "__main__":
    generate_chairs()