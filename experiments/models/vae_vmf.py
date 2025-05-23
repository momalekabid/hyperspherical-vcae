import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../vmf'))
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
# we keep the vmf and powerspherical imports into different files for simplicity, as they share hypersphericalUniform
# mainly to avoid complications when running code in notebooks
class ModelVAE(nn.Module):
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution="normal", device="cpu"):
        super(ModelVAE, self).__init__()
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution

        self.encoder = nn.Sequential(
            nn.Linear(784, h_dim * 2), nn.ReLU(), nn.Linear(h_dim * 2, h_dim), nn.ReLU()
        )

        if self.distribution == "normal":
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        elif self.distribution == "vmf":
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_scale = nn.Linear(h_dim, 1)

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, 784),
        )

        self.to(device)

    def encode(self, x):
        x = self.encoder(x)
        if self.distribution == "normal":
            return self.fc_mean(x), F.softplus(self.fc_var(x))
        elif self.distribution == "vmf":
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            z_scale = F.softplus(self.fc_scale(x)) + 1
            return z_mean, z_scale

    def reparameterize(self, z_mean, z_var_or_scale):
        if self.distribution == "normal":
            q_z = torch.distributions.normal.Normal(z_mean, z_var_or_scale)
            p_z = torch.distributions.normal.Normal(
                torch.zeros_like(z_mean), torch.ones_like(z_var_or_scale)
            )
        elif self.distribution == "vmf":
            q_z = VonMisesFisher(z_mean, z_var_or_scale)
            p_z = HypersphericalUniform(self.z_dim - 1, validate_args=False)
        return q_z, p_z

    def forward(self, x):
        z_mean, z_var_or_scale = self.encode(x.view(-1, 784))
        q_z, p_z = self.reparameterize(z_mean, z_var_or_scale)
        z = q_z.rsample()
        x_ = self.decoder(z)
        return (z_mean, z_var_or_scale), (q_z, p_z), z, x_


def compute_loss(model, x):
    _, (q_z, p_z), _, x_ = model(x)
    loss_recon = F.binary_cross_entropy_with_logits(
        x_, x.view(-1, 784), reduction="sum"
    ) / x.size(0)
    loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    return loss_recon + loss_KL