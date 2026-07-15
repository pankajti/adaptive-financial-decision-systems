"""Optional VAE latent-state extension."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def run_vae_projection(
    model_df: pd.DataFrame,
    feature_cols: list[str],
    latent_dim: int = 2,
    hidden_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[pd.DataFrame, object]:
    """Fit a simple VAE and append VAE latent columns.

    Returns a copy of model_df with VAE1..VAEk columns and the trained torch module.
    Requires torch. Keep this optional because PCA is the primary interpretable baseline.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for run_vae_projection.") from exc

    torch.manual_seed(seed)

    class VAE(nn.Module):
        def __init__(self, input_dim, latent_dim=2, hidden_dim=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.logvar(h)

        def reparameterize(self, mu, logvar):
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar

    X = SimpleImputer(strategy="median").fit_transform(model_df[feature_cols])
    X = StandardScaler().fit_transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)
    vae = VAE(X.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    for _ in range(epochs):
        for (batch,) in loader:
            recon, mu, logvar = vae(batch)
            recon_loss = nn.functional.mse_loss(recon, batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.01 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        z, _ = vae.encode(X_tensor)
    out = model_df.copy()
    for i in range(latent_dim):
        out[f"VAE{i+1}"] = z[:, i].numpy()
    return out, vae
