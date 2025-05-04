import torch
import torch.nn as nn
import torch.nn.functional as F
from .clifford import CliffordTorusDistribution, CliffordTorusUniform


class ModelVAE(nn.Module):
    """ Defined architecture as taken from experiments [] from
    (Davidson et al. ) https://arxiv.org/abs/1804.00891"""

    def __init__(self, h_dim: int, z_dim: int, device: str | torch.device = "cpu"):
        super().__init__()
        self.z_dim = z_dim

        
        self.encoder = nn.Sequential(
            nn.Linear(784, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_concentration = nn.Linear(h_dim, 1)

        # decoder network (note: Clifford sample is complex length 2*z_dim)
        # we convert to real (4*z_dim) then project back to z_dim real vector
        self.pre_fc = nn.Linear(4 * z_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim * 2),
            nn.ReLU(),
            nn.Linear(h_dim * 2, 784),
        )

        self.to(device)

    def encode(self, x: torch.Tensor):
        h = self.encoder(x.view(-1, 784))
        mu = self.fc_mu(h)
        concentration = F.softplus(self.fc_concentration(h)) + 1
        return mu, concentration

    def reparameterize(self, mu: torch.Tensor, concentration: torch.Tensor):
        q_z = CliffordTorusDistribution(mu, concentration.expand_as(mu))
        p_z = CliffordTorusUniform(self.z_dim, device=mu.device)
        z_c = q_z.rsample()  # complex (B, 2*z_dim)
        z_real = torch.view_as_real(z_c).reshape(z_c.size(0), -1).float()
        z = self.pre_fc(z_real)
        return z, q_z, p_z

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        mu, kappa = self.encode(x)
        z, q_z, p_z = self.reparameterize(mu, kappa)
        x_recon = self.decoder(z)
        return (mu, kappa), (q_z, p_z), z, x_recon


def compute_loss(model: "ModelVAE", x: torch.Tensor, beta: float = 1.0):
    (mu, kappa), (q_z, p_z), _, x_recon = model(x)
    recon_loss = F.binary_cross_entropy_with_logits(
        x_recon, x.view(-1, 784), reduction="sum"
    ) / x.size(0)
    kl = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
    return recon_loss + beta * kl 