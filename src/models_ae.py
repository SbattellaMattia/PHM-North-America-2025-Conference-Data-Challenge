import torch
import torch.nn as nn

class TimeStepAE(nn.Module):
    """
    AE applied per time-step (shared MLP), then reconstruction per time-step.
    Input: [B, L, D]
    """
    def __init__(self, in_dim, latent_dim=48, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        # x: [B,L,D]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
