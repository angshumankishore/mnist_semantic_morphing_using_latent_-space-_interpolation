import torch
import torch.nn as nn
import numpy as np
import cv2

LATENT_DIM = 20
DEVICE = torch.device("cpu")


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU()
        )

        self.mu = nn.Linear(400, LATENT_DIM)
        self.logvar = nn.Linear(400, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# NOTE: image loading and preprocessing utilities have been moved to
# `image_utils.py` to keep the model code focused on architecture and
# inference. See `image_utils.py` for helpers that produce tensors
# ready to be passed into the VAE.
