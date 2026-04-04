import torch
import torch.nn as nn

class PointCloudVAE(nn.Module):
    def __init__(self, num_points=2048, latent_dim=128):
        super().__init__()
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

        # Decoder: latent vector -> full point cloud
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1536),
            nn.ReLU(),
            nn.Linear(1536, num_points * 3),
        )

    def encode(self, x):
        # x shape: (batch, 2048, 3)
        features = self.encoder(x)          # (batch, 2048, 256)
        global_feat = features.max(dim=1)[0] # (batch, 256) — max pool over points
        mu = self.fc_mu(global_feat)
        logvar = self.fc_logvar(global_feat)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        # z shape: (batch, 128)
        out = self.decoder(z)                          # (batch, 2048*3)
        return out.view(-1, self.num_points, 3)        # (batch, 2048, 3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar