import torch
import torch.nn as nn

class pointCloudVAE(nn.module):
    def __init__(self, num_points=2048, latent_dim=128):
        super.__init__()
        self.num_points = num_points
        self.latent_dim = latent_dim

        # Encoder: processes each point independently, then aggregates
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)