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


def load_user_image(path_or_digit):
    """
    Accepts:
    - path to image OR
    - single digit character ('0'–'9')
    """

    # if digit, load from fixed directory
    if len(path_or_digit) == 1 and path_or_digit.isdigit():
        path = f"digits/{path_or_digit}.png"
    else:
        path = path_or_digit

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image path")

    img = 255 - img
    img = cv2.resize(img, (28, 28))
    img = img.astype(np.float32) / 255.0

    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    return img.to(DEVICE)
