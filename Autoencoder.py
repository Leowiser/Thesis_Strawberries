import torch
import torch.nn as nn

class HyperspectralCNN(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super(HyperspectralCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # Reduce size
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(64 * 28 * 49 * 28, latent_dim)  # Adapt this based on your input size
        self.fc_logvar = nn.Linear(64 * 28 * 49 * 28, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim, output_shape):
        super(VAE, self).__init__()
        self.encoder = HyperspectralCNN(in_channels, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, 64 * 28 * 49 * 28)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(self.decoder_input(z).view(-1, 64, 28, 49, 28))  # Reshape
        return x_recon, mu, logvar
    
    def vae_loss(x, x_recon, mu, logvar):
        recon_loss = nn.MSELoss()(x_recon, x)  # or nn.BCELoss()
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div