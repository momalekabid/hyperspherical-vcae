import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from power_spherical import PowerSpherical, HypersphericalUniform
from models.clifford import CliffordTorusDistribution, CliffordTorusUniform


class Encoder(nn.Module):
    def __init__(self, latent_dim, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        # alternative/old arch. below {}
        # self.conv1 = nn.Conv2d(in_channels, 32, 2, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        # self.conv4 = nn.Conv2d(128, 256, 6, stride=2, padding=3)
        # self.fc_mu = nn.Sequential(
        #     nn.Linear(512 * 2 * 2, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, latent_dim) ...
        # self.fc_logvar = nn.Sequential(
        #     nn.Linear(512 * 2 * 2, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1),
        #     nn.Softplus()
        # ))
        self.fc_logvar = nn.Linear(512 * 2 * 2, 1)
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        # vae sampling
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = F.softplus(logvar) + 1
        # print(logvar)
        # the `+ 1` can prevent collapsing behaviors
        # normalize mu 
        mu = F.normalize(mu, p=2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    """Generic decoder used by all latent distributions.

    Note: For the Clifford torus distribution the latent samples live in a
    complex-valued space of size ``2 * latent_dim`` (real + imaginary parts).
    @param latent:dim, out_channels
    """

    def __init__(self, latent_dim, out_channels=1):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)
        # self.fc = nn.Sequential(
        #     nn.Linear(latent_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512 * 2 * 2)
        # )
        # self.conv_trans0 = nn.ConvTranspose2d(256, 128, 6, stride=2, padding=3)
        # self.conv_trans1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        # self.conv_trans2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        # self.conv_trans3 = nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1)
        self.conv_trans1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(64, out_channels, 3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 2, 2)
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = F.relu(self.conv_trans3(x))
        x = self.conv_trans4(x)
        x = F.tanh(x)
        return x


class VAE(nn.Module):
    """Convolutional VAE that supports Gaussian and various hyperspherical distributions 
    latent distributions.
    """

    def __init__(self, latent_dim, in_channels=1, distribution="powerspherical"):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim, in_channels)

        if distribution == "clifford":
            self.pre_fc = nn.Linear(latent_dim * 4, latent_dim * 2)
            dec_in_dim = latent_dim * 2
        else:
            self.pre_fc = nn.Identity()
            dec_in_dim = latent_dim

        self.decoder = Decoder(dec_in_dim, in_channels)
        self.latent_dim = latent_dim
        self.distribution = distribution
        
    def reparameterize(self, mu, logvar):
        if self.distribution == "powerspherical":
            ## we already normalize mu to unit vector in encoder
            # else: print(f"diff between unit norm and mu: {mu.norm(dim=-1)}")
            q_z = PowerSpherical(mu, logvar.squeeze(-1)) # logvar: (B,1) -> (B,)
            p_z = HypersphericalUniform(self.latent_dim - 1, validate_args=False)
            z = q_z.rsample()
            return z, q_z, p_z
        elif self.distribution == "clifford":
            concentration = logvar.expand(-1, mu.size(-1))
            q_z = CliffordTorusDistribution(mu, concentration)
            p_z = CliffordTorusUniform(self.latent_dim, device=mu.device)

            z_complex = q_z.rsample()  # shape (..., 2*latent_dim)
            # convert to real tensor with shape (B, 4*latent_dim)
            z_real = torch.view_as_real(z_complex).reshape(z_complex.size(0), -1).float()
            z = self.pre_fc(z_real)
            return z, q_z, p_z
        else:  # gaussian
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std, None, None
            
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z, q_z, p_z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, q_z, p_z

    def compute_loss(self, x, recon_x, mu, logvar, q_z, p_z, beta, gamma):
        # reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        if self.distribution in ["powerspherical", "clifford"]:
            kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum()
        elif self.distribution == "gaussian":
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
            
        # unitary constraint
        fft_magnitudes = torch.abs(torch.fft.fftshift(torch.fft.fft(mu)))
        target = torch.ones_like(fft_magnitudes) / torch.sqrt(torch.tensor(mu.size(-1), dtype=torch.float32))
        unitary_loss = F.mse_loss(fft_magnitudes, target, reduction='sum')
        unitary_loss = F.mse_loss(fft_magnitudes, torch.ones_like(fft_magnitudes), reduction='sum')
        total_loss = recon_loss + beta * kl_loss + gamma * unitary_loss
        return total_loss, recon_loss, kl_loss, unitary_loss

