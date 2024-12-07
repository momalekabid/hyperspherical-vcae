import torch
import torch.nn as nn
from torch.distributions import VonMises, Distribution, constraints
from torch.distributions.utils import broadcast_all
import math
import numpy as np
from scipy.special import i0, i1
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from torch.distributions.kl import register_kl, kl_divergence

class CliffordTorusDistribution(Distribution):
    """
    Distribution over points on a Clifford torus using independent von Mises distributions.
    """
    arg_constraints = {
        "loc": constraints.real,
        "concentration": constraints.positive
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, concentration, validate_args=None):
        self.loc, self.concentration = broadcast_all(loc, concentration)
        batch_shape = self.loc.shape[:-1]
        self.orig_dim = self.loc.shape[-1]
        event_shape = torch.Size([2 * self.orig_dim])
        super().__init__(batch_shape, event_shape, validate_args)
        self.device = loc.device
        self.dtype = loc.dtype

    def _von_mises_entropy(self, kappa):
        """
        Compute entropy of von Mises distribution manually.
        H = log(2π I₀(κ)) - κ E[cos(x - μ)]
        where E[cos(x - μ)] = I₁(κ)/I₀(κ)
        """
        kappa_np = kappa.detach().cpu().numpy()
        
        # compute I₀(κ) and I₁(κ)
        i0_k = i0(kappa_np)
        i1_k = i1(kappa_np)
        
        i0_k = torch.from_numpy(i0_k).to(kappa.device)
        i1_k = torch.from_numpy(i1_k).to(kappa.device)
        
        # compute E[cos(x - μ)] = I₁(κ)/I₀(κ)
        expected_cos = i1_k / i0_k
        
        # compute entropy
        entropy = torch.log(2 * math.pi * i0_k) - kappa * expected_cos
        
        return entropy

    def rsample(self, sample_shape=torch.Size()):
        loc = self.loc
        concentration = self.concentration
        # sample from mu 
        shape = self._extended_shape(sample_shape)
        theta_collection = torch.zeros(
            *shape[:-1],  # all but last dimension
            self.orig_dim,  # original dimension
            dtype=torch.float32,
            device=self.loc.device
        )
        # theta_collection = torch.zeros(
        #     self.loc.shape,
        #     dtype=torch.float32,
        #     device=self.loc.device
        # ) 
        for i in range(self.orig_dim):
            von_mises = VonMises(
                loc=loc[..., i],
                concentration=concentration[..., i]
            )
            theta = von_mises.sample(sample_shape) 
            theta_collection[..., i] = theta

        n = self.orig_dim * 2
        theta_s = torch.zeros((*theta_collection.shape[:-1], n), device=theta_collection.device) 
        # ensure DC component (k=0), should be 0
        try:
            assert [theta_s[i, 0] == 0 for i in range(theta_s.shape[0])]
        except Exception as e:
            print(f"DC component is not 0: {e}")
        
        # print(theta_s)
        # fill positive frequencies (k > 0)
        theta_s[..., 1:self.orig_dim] = theta_collection[..., 1:]
        # fill negative frequencies to ensure conjugate symmetry
        # For k from 1 to N/2-1: F(N-k) = F(k)*
        theta_s[..., -self.orig_dim+1:] = -torch.flip(theta_collection[..., 1:], dims=(-1,))
        # print(theta_s[..., n//2])
        try:
            assert [theta_s[i, n//2] == 0 for i in range(theta_s.shape[0])]
        except Exception as e:
            print(f"error at verification of conj symmetry: {e}")
            print(theta_s)
        # assert [theta_s[i, n//2] == 0 for i in range(theta_s.shape[0])]
        
        # convert to complex exponentials
        samples_complex = torch.exp(1j * theta_s)

        # orthonormal scales by 1/sqrt(N) instead of 1/n, which means it's a unitary transform too?
        result = torch.fft.ifft(samples_complex, dim=-1, norm="ortho") 
        # print(f"result shape: {result.shape}")

        return result.real

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        
        freq_domain = torch.fft.fft(value, dim=-1, norm="ortho")
        first_half = freq_domain[..., :self.orig_dim]
       
        angles = torch.angle(first_half)
        log_prob = 0.
        
        for i in range(self.orig_dim):
            von_mises = VonMises(
                loc=self.loc[..., i],
                concentration=self.concentration[..., i]
            )
            log_prob = log_prob + von_mises.log_prob(angles[..., i])
        
        return log_prob

    def entropy(self):
        """
        Compute entropy as sum of individual von Mises entropies.
        Since the distribution is a product of independent von Mises distributions,
        the total entropy is the sum of individual entropies.
        """
        total_entropy = 0.
        for i in range(self.orig_dim):
            kappa = self.concentration[..., i]
            total_entropy = total_entropy + self._von_mises_entropy(kappa)
        return total_entropy

class CliffordTorusEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.W_mu = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        mu = self.W_mu(x)
        concentration = torch.exp(self.W_k(x))
        return CliffordTorusDistribution(mu, concentration)

class CliffordTorusUniform(torch.distributions.Distribution):
    """
    Uniform distribution on the Clifford torus.
    """
    arg_constraints = {
        "dim": torch.distributions.constraints.positive_integer,
    }
    support = torch.distributions.constraints.real
    has_rsample = True

    def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=None):
        self.dim = dim if isinstance(dim, torch.Tensor) else torch.tensor(dim, device=device)
        super().__init__(validate_args=validate_args)
        self.device = device
        self.dtype = dtype
        
    def rsample(self, sample_shape=torch.Size()):
        # Sample uniform angles
        shape = sample_shape + torch.Size([self.dim])
        angles = torch.rand(shape, device=self.device) * 2 * math.pi
        
        # Convert to complex exponentials
        samples_complex = torch.zeros(
            sample_shape + torch.Size([2 * self.dim]), 
            dtype=torch.complex64,
            device=self.device
        )
        samples_complex[..., :self.dim] = torch.exp(1j * angles)
        # Fill conjugate symmetric values
        samples_complex[..., self.dim:] = torch.conj(
            torch.flip(samples_complex[..., :self.dim], dims=(-1,))
        )
        
        # Take inverse FFT
        result = torch.fft.ifft(samples_complex, dim=-1, norm="ortho")
        return result

    def log_prob(self, value):
        # Uniform probability on the torus
        return -torch.ones_like(value[..., 0]) * self.log_normalizer()
    
    def log_normalizer(self):
        # Log of surface area of torus = log((2π)^dim)
        return self.dim * math.log(2 * math.pi)

    def entropy(self):
        return self.log_normalizer()


# def test_clifford_torus_properties():
#     """Test suite to verify properties of the Clifford torus distribution"""
    
#     # Test 1: Verify distribution integrates to 1 and sampling works
#     dim = 10  # This gives us a 4D embedding
#     loc = torch.zeros(dim)
#     concentration = torch.ones(dim) * 5.0
#     dist = CliffordTorusDistribution(loc, concentration)
    
#     # Generate samples and visualize
#     n_samples = 1000
#     samples = dist.rsample((n_samples,))
    
#     # Plot circle embeddings
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     ax1.scatter(samples[:, 0].real, samples[:, 1].real, alpha=0.5)
#     ax1.set_title('First Circle Embedding')
#     ax1.set_aspect('equal')
#     ax2.scatter(samples[:, 2].real, samples[:, 3].real, alpha=0.5)
#     ax2.set_title('Second Circle Embedding')
#     ax2.set_aspect('equal')
#     plt.show()
    
#     # Test 2: Check entropy estimation via MC
#     entropy_mc = -dist.log_prob(samples).mean()
#     entropy_analytical = dist.entropy()
#     print("MC Entropy:", entropy_mc.item())
#     print("Analytical Entropy:", entropy_analytical.item())
#     print("Close?", torch.isclose(entropy_mc, entropy_analytical, atol=1e-2))
    
#     # Test 3: Check KL divergence with uniform
#     prior = CliffordTorusUniform(dim)
#     kl_mc = (dist.log_prob(samples) - prior.log_prob(samples)).mean()
#     kl_analytical = torch.distributions.kl.kl_divergence(dist, prior)
#     print("\nKL divergence (MC):", kl_mc.item())
#     print("KL divergence (Analytical):", kl_analytical.item())
#     print("Close?", torch.isclose(kl_mc, kl_analytical, atol=1e-2))
    
#     # Test 4: Check gradients
#     sample = dist.rsample()
#     grad_loc = torch.autograd.grad(sample.sum(), dist.loc, retain_graph=True)[0]
#     grad_concentration = torch.autograd.grad(sample.sum(), dist.concentration)[0]
#     print("\nGradient w.r.t loc:", grad_loc)
#     print("Gradient w.r.t concentration:", grad_concentration)


def test_clifford_torus_properties():
    """
    Test suite to verify and visualize properties of the Clifford torus distribution
    """
    # create a simple distribution instance
    
    dim = 2  # This will give us a 4D embedding that we can project
    loc = torch.zeros(dim)
    concentration = torch.ones(dim)
    dist = CliffordTorusDistribution(loc, concentration)
    
    # generate samples
    n_samples = 1000
    samples = dist.rsample((n_samples,))
    
    # plot circle embeddings
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(samples[:, 0].real, samples[:, 1].real, alpha=0.5)
    ax1.set_title('First Circle Embedding')
    ax1.set_aspect('equal')
    ax2.scatter(samples[:, 2].real, samples[:, 3].real, alpha=0.5)
    ax2.set_title('Second Circle Embedding')
    ax2.set_aspect('equal')
    plt.show() 
    # 1. test that samples live on unit circle in each plane
    def plot_angle_distributions(samples):
        """
        Plot angle distributions for both circles
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Compute angles for first circle
        angles1 = torch.atan2(samples[:, 1].real, samples[:, 0].real)
        ax1.hist(angles1.numpy(), bins=50, density=True)
        ax1.set_title('Angle Distribution (First Circle)')
        
        # Compute angles for second circle
        angles2 = torch.atan2(samples[:, 3].real, samples[:, 2].real)
        ax2.hist(angles2.numpy(), bins=50, density=True)
        ax2.set_title('Angle Distribution (Second Circle)')
        
        plt.show()

    # 5. test for independence of circles
    def plot_correlation():
        angles1 = torch.atan2(samples[:, 1].real, samples[:, 0].real)
        angles2 = torch.atan2(samples[:, 3].real, samples[:, 2].real)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(angles1, angles2, alpha=0.5)
        plt.xlabel('Angle on first circle')
        plt.ylabel('Angle on second circle')
        plt.title('Independence of Circle Parameters')
        plt.show()

    samples = dist.rsample((n_samples,))
    print("Testing Clifford Torus Properties...")
    plot_angle_distributions(samples)
    plot_correlation()

@torch.distributions.kl.register_kl(CliffordTorusDistribution, CliffordTorusUniform)
def _kl_clifford_uniform(p, q):
    """
    Compute KL(p||q) between a Clifford torus distribution p and uniform distribution q.
    KL divergence is the sum of KL divergences for each von Mises component.
    """
    return -p.entropy() + q.entropy()

if __name__ == "__main__":
 
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 64 
    d = 4 
    
    encoder = CliffordTorusEncoder(d, d).to(device)
    x = torch.randn(batch_size, d, device=device)
    distribution = encoder(x)
    
    samples = distribution.rsample()
    # print(samples.real)
    # print(samples)
    # print(samples.imag)
    # magnitude of imaginary components summed
    imaginary = [torch.sum(samples[..., i].imag) for i in range(samples.shape[-1])]
    imaginary = torch.stack(imaginary)
    imaginary = torch.sum(imaginary)
    imaginary = imaginary.item()
    # sum all imag comp 
    # print("Sample shape:", samples.shape)
    print("Sum of imaginary components:", imaginary)
    print("Is real:", torch.allclose(samples.imag, torch.zeros_like(samples.imag), rtol=1e-6, atol=1e-6))
    
    log_prob = distribution.log_prob(samples)
    print("Log probability shape:", log_prob.shape)
    
    entropy = distribution.entropy()
    print("Entropy shape:", entropy.shape)
    test_clifford_torus_properties()
    prior = CliffordTorusUniform(d, device=device)
    posterior = distribution  # Your encoder output
    
    kl_div = torch.distributions.kl.kl_divergence(posterior, prior)
    print("KL divergence shape:", kl_div.shape)
    print("Mean KL divergence:", kl_div.mean().item())

