import zipfile
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Unzip the file
with zipfile.ZipFile('shapenet.zip', 'r') as zip_ref:
    zip_ref.extractall('shapenet_data')

# Chair directory in ShapeNetPart
chair_dir = 'shapenet_data/PartAnnotation/03001627/points'

# Load all chair point clouds
chair_clouds = []
for filename in sorted(os.listdir(chair_dir)):
    filepath = os.path.join(chair_dir, filename)
    points = np.loadtxt(filepath)  # each file is (N, 3) xyz coordinates
    chair_clouds.append(points)

print(f"Found {len(chair_clouds)} chair point clouds")
print(f"First cloud shape: {chair_clouds[0].shape}")

# Normalize to 2048 points per cloud (some may have more or fewer)
def resample(points, n=2048):
    idx = np.random.choice(len(points), n, replace=len(points) < n)
    return points[idx]

chair_clouds = np.array([resample(pc) for pc in chair_clouds])  # (num_chairs, 2048, 3)
print(f"Resampled shape: {chair_clouds.shape}")

# PyTorch dataset
class ChairPointCloudDataset(Dataset):
    def __init__(self, point_clouds):
        self.points = torch.FloatTensor(point_clouds)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return self.points[idx]

dataset = ChairPointCloudDataset(chair_clouds)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Sanity check
for batch in dataloader:
    print("Batch shape:", batch.shape)  # (32, 2048, 3)
    break