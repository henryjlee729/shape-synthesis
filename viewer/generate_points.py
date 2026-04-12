import torch
import numpy as np
from pathlib import Path
from time import time
from vae import PointCloudVAE

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "vae_chair.pth"
OUTPUT_DIR = PROJECT_DIR / "test"

def generate_chairs(num_chairs=1):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PointCloudVAE(num_points=2048, latent_dim=128).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    with torch.no_grad():
        for i in range(num_chairs):
            z = torch.randn(1, 128, device=device)
            points = model.decode(z).cpu().numpy()[0]

            stamp = int(time() * 1000)
            out_path = OUTPUT_DIR / f"chair_{stamp}_{i}.pts"
            np.savetxt(out_path, points, fmt="%.6f")

            print(f"Saved {out_path}")
            saved_paths.append(str(out_path))

    return saved_paths[0] if num_chairs == 1 else saved_paths