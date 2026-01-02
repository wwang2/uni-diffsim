#!/usr/bin/env python3
"""
Contrastive Divergence via Differentiable Simulation

This script demonstrates how differentiable simulation enables efficient
contrastive divergence (CD) training of energy-based models.

Theory:
-------
Energy-based models define p(x|θ) ∝ exp(-E(x;θ)). Maximum likelihood training
requires computing:

    ∇_θ log p(x_data|θ) = -∇_θ E(x_data;θ) + E_p[∇_θ E(x;θ)]

The second term (partition function gradient) requires sampling from the model.
Contrastive Divergence approximates this by running k steps of MCMC from data:

    ∇_θ^CD ≈ ∇_θ E(x_data;θ) - ∇_θ E(x_model;θ)

where x_model = MCMC_k(x_data).

Differentiable Simulation Advantage:
------------------------------------
1. **CD-k via Langevin dynamics**: Instead of Gibbs sampling, we use
   differentiable Langevin dynamics for the negative phase. This allows
   gradients to flow through the sampling process.

2. **Gradient methods comparison**:
   - BPTT: Backprop through k Langevin steps (memory-intensive but low variance)
   - REINFORCE: Score function estimator (memory-efficient, higher variance)
   - Persistent CD: Maintain chains across updates (better mixing)

3. **Variance reduction**: The framework's gradient estimators provide
   natural variance reduction techniques.

Experiments:
------------
1. Learning a 1D mixture of Gaussians
2. Learning a 2D double-well potential
3. Comparison of CD-k for different k values
4. BPTT vs REINFORCE for CD gradients

Author: uni-diffsim demo
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uni_diffsim import OverdampedLangevin, BAOAB
from uni_diffsim.plotting import apply_style, COLORS, LW, MS

apply_style()


# =============================================================================
# Learnable Energy-Based Models
# =============================================================================

class GaussianMixtureEBM(nn.Module):
    """Energy-based model parameterized as mixture of Gaussians.
    
    E(x; θ) = -log Σ_k w_k exp(-0.5 * (x - μ_k)² / σ_k²)
    
    This is a simple EBM where we learn the means and weights of a GMM.
    """
    
    def __init__(self, n_components: int = 2, dim: int = 1):
        super().__init__()
        self.n_components = n_components
        self.dim = dim
        
        # Initialize means spread out
        self.means = nn.Parameter(torch.randn(n_components, dim) * 2)
        # Log-weights for numerical stability (softmax to get weights)
        self.log_weights = nn.Parameter(torch.zeros(n_components))
        # Log-variances
        self.log_vars = nn.Parameter(torch.zeros(n_components))
    
    @property
    def weights(self):
        return torch.softmax(self.log_weights, dim=0)
    
    @property
    def variances(self):
        return torch.exp(self.log_vars)
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy E(x). x: (..., dim) -> (...)"""
        # x: (..., dim), means: (K, dim)
        # Compute log-sum-exp of weighted Gaussians
        
        # Expand for broadcasting: x -> (..., 1, dim), means -> (K, dim)
        x_exp = x.unsqueeze(-2)  # (..., 1, dim)
        
        # Squared distances: (..., K)
        sq_dist = ((x_exp - self.means) ** 2).sum(-1)  # (..., K)
        
        # Log of Gaussian components (unnormalized)
        log_components = -0.5 * sq_dist / self.variances  # (..., K)
        
        # Add log-weights
        log_weighted = log_components + self.log_weights  # (..., K)
        
        # Energy = -log Σ exp(log_weighted) = -logsumexp
        energy = -torch.logsumexp(log_weighted, dim=-1)
        
        return energy
    
    def force(self, x: torch.Tensor) -> torch.Tensor:
        """Compute force = -∇E(x)."""
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            e = self.energy(x_req)
            grad = torch.autograd.grad(e.sum(), x_req)[0]
        return -grad
    
    def sample_model(self, n_samples: int, device='cpu') -> torch.Tensor:
        """Sample from the model using the current parameters."""
        with torch.no_grad():
            # Sample component indices
            indices = torch.multinomial(self.weights, n_samples, replacement=True)
            # Sample from selected Gaussians
            samples = torch.randn(n_samples, self.dim, device=device)
            samples = samples * torch.sqrt(self.variances[indices]).unsqueeze(-1)
            samples = samples + self.means[indices]
        return samples


class NeuralEBM(nn.Module):
    """Neural network energy-based model.
    
    E(x; θ) = f_θ(x) where f is a neural network.
    """
    
    def __init__(self, dim: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) -> (...)"""
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.dim)
        e_flat = self.net(x_flat).squeeze(-1)
        return e_flat.reshape(shape)
    
    def force(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            e = self.energy(x_req)
            grad = torch.autograd.grad(e.sum(), x_req)[0]
        return -grad


# =============================================================================
# Contrastive Divergence Training
# =============================================================================

def cd_loss_bptt(model: nn.Module, data: torch.Tensor, 
                 integrator: OverdampedLangevin, dt: float, 
                 cd_steps: int) -> torch.Tensor:
    """Contrastive Divergence loss with BPTT through Langevin dynamics.
    
    Loss = E[E(x_data)] - E[E(x_model)]
    
    where x_model is obtained by running cd_steps of Langevin from x_data.
    Gradients flow through the Langevin chain via backprop.
    
    Args:
        model: Energy-based model with .energy() and .force() methods
        data: Data samples (n_samples, dim)
        integrator: Langevin integrator
        dt: Time step
        cd_steps: Number of MCMC steps for negative phase
    
    Returns:
        Scalar loss value
    """
    # Positive phase: energy at data
    e_data = model.energy(data).mean()
    
    # Negative phase: run Langevin from data
    x_model = integrator.run(
        data, model.force, dt=dt, n_steps=cd_steps, final_only=True
    )[0]  # Shape: (n_samples, dim)
    
    e_model = model.energy(x_model).mean()
    
    # CD loss: minimize data energy, maximize model sample energy
    # This pushes probability mass toward data
    loss = e_data - e_model
    
    return loss


def cd_loss_reinforce(model: nn.Module, data: torch.Tensor,
                      integrator: OverdampedLangevin, dt: float,
                      cd_steps: int, kT: float = 1.0) -> torch.Tensor:
    """Contrastive Divergence with REINFORCE gradient for negative phase.
    
    The positive phase gradient is straightforward: ∇_θ E(x_data; θ)
    
    For the negative phase, we use the REINFORCE identity:
        ∇_θ E_p[E(x;θ)] = E_p[E · ∇_θ log p] + E_p[∇_θ E]
                        = -β E_p[(E - ⟨E⟩) · ∇_θ E] + E_p[∇_θ E]
    
    This avoids backprop through dynamics but has higher variance.
    """
    # Positive phase (direct gradient)
    e_data = model.energy(data).mean()
    
    # Negative phase: sample without gradients
    with torch.no_grad():
        x_model = integrator.run(
            data, model.force, dt=dt, n_steps=cd_steps, final_only=True
        )[0]
    
    # Compute energy at model samples
    e_model = model.energy(x_model)
    e_model_mean = e_model.mean()
    
    # REINFORCE surrogate loss for the negative phase
    # The gradient of this equals the REINFORCE gradient
    beta = 1.0 / kT
    e_centered = (e_model - e_model_mean).detach()
    
    # Surrogate: when differentiated, gives -β * Cov(E, ∇E) + ⟨∇E⟩
    # This is the score function gradient for E_p[E]
    surrogate = e_model_mean - beta * (e_centered * e_model).mean()
    
    # Total loss
    loss = e_data - surrogate
    
    return loss


def persistent_cd_loss(model: nn.Module, data: torch.Tensor,
                       persistent_chains: torch.Tensor,
                       integrator: OverdampedLangevin, dt: float,
                       cd_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Persistent Contrastive Divergence (PCD).
    
    Instead of initializing chains from data each time, maintain
    persistent chains that continue across gradient updates.
    This improves mixing and reduces bias from short chains.
    
    Returns:
        (loss, updated_chains)
    """
    # Positive phase
    e_data = model.energy(data).mean()
    
    # Negative phase: continue from persistent chains
    with torch.no_grad():
        new_chains = integrator.run(
            persistent_chains, model.force, dt=dt, n_steps=cd_steps, final_only=True
        )[0]
    
    e_model = model.energy(new_chains).mean()
    
    loss = e_data - e_model
    
    return loss, new_chains.detach()


# =============================================================================
# Experiment 1: Learning 1D Mixture of Gaussians
# =============================================================================

def experiment_1d_gmm():
    """Learn a 1D mixture of Gaussians from samples."""
    print("\n" + "="*60)
    print("Experiment 1: Learning 1D GMM with Contrastive Divergence")
    print("="*60)
    
    torch.manual_seed(42)
    
    # True distribution: mixture of 2 Gaussians
    true_means = torch.tensor([[-2.0], [2.0]])
    true_vars = torch.tensor([0.5, 0.5])
    true_weights = torch.tensor([0.4, 0.6])
    
    def sample_true(n):
        indices = torch.multinomial(true_weights, n, replacement=True)
        samples = torch.randn(n, 1) * torch.sqrt(true_vars[indices]).unsqueeze(-1)
        samples = samples + true_means[indices]
        return samples
    
    # Model
    model = GaussianMixtureEBM(n_components=2, dim=1)
    
    # Initialize far from true
    with torch.no_grad():
        model.means.data = torch.tensor([[0.0], [0.5]])
        model.log_weights.data = torch.tensor([0.0, 0.0])
        model.log_vars.data = torch.tensor([0.0, 0.0])
    
    # Training settings
    kT = 1.0
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    
    n_epochs = 300
    batch_size = 256
    dt = 0.05
    
    # Compare different CD-k values
    cd_steps_list = [1, 5, 20]
    results = {k: {'losses': [], 'means': [], 'weights': []} for k in cd_steps_list}
    
    for cd_steps in cd_steps_list:
        print(f"\n--- CD-{cd_steps} ---")
        
        # Reset model
        model_k = GaussianMixtureEBM(n_components=2, dim=1)
        with torch.no_grad():
            model_k.means.data = torch.tensor([[0.0], [0.5]])
            model_k.log_weights.data = torch.tensor([0.0, 0.0])
            model_k.log_vars.data = torch.tensor([0.0, 0.0])
        
        optimizer_k = torch.optim.Adam(model_k.parameters(), lr=0.05)
        
        for epoch in range(n_epochs):
            data = sample_true(batch_size)
            
            optimizer_k.zero_grad()
            loss = cd_loss_bptt(model_k, data, integrator, dt, cd_steps)
            loss.backward()
            optimizer_k.step()
            
            results[cd_steps]['losses'].append(loss.item())
            results[cd_steps]['means'].append(model_k.means.detach().clone())
            results[cd_steps]['weights'].append(model_k.weights.detach().clone())
            
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: loss={loss.item():.4f}, "
                      f"means={model_k.means.data.squeeze().tolist()}")
    
    return results, true_means, true_weights, sample_true


# =============================================================================
# Experiment 2: BPTT vs REINFORCE for CD
# =============================================================================

def experiment_bptt_vs_reinforce():
    """Compare BPTT and REINFORCE gradients for CD training."""
    print("\n" + "="*60)
    print("Experiment 2: BPTT vs REINFORCE for CD Gradients")
    print("="*60)
    
    torch.manual_seed(123)
    
    # True distribution
    true_means = torch.tensor([[-1.5], [1.5]])
    true_vars = torch.tensor([0.3, 0.3])
    true_weights = torch.tensor([0.5, 0.5])
    
    def sample_true(n):
        indices = torch.multinomial(true_weights, n, replacement=True)
        samples = torch.randn(n, 1) * torch.sqrt(true_vars[indices]).unsqueeze(-1)
        samples = samples + true_means[indices]
        return samples
    
    kT = 1.0
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    
    n_epochs = 200
    batch_size = 256
    dt = 0.05
    cd_steps = 10
    
    results = {
        'bptt': {'losses': [], 'means': []},
        'reinforce': {'losses': [], 'means': []},
    }
    
    # BPTT training
    print("\n--- BPTT ---")
    model_bptt = GaussianMixtureEBM(n_components=2, dim=1)
    with torch.no_grad():
        model_bptt.means.data = torch.tensor([[0.0], [0.5]])
    optimizer_bptt = torch.optim.Adam(model_bptt.parameters(), lr=0.03)
    
    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        data = sample_true(batch_size)
        
        optimizer_bptt.zero_grad()
        loss = cd_loss_bptt(model_bptt, data, integrator, dt, cd_steps)
        loss.backward()
        optimizer_bptt.step()
        
        results['bptt']['losses'].append(loss.item())
        results['bptt']['means'].append(model_bptt.means.detach().clone())
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    # REINFORCE training
    print("\n--- REINFORCE ---")
    model_rf = GaussianMixtureEBM(n_components=2, dim=1)
    with torch.no_grad():
        model_rf.means.data = torch.tensor([[0.0], [0.5]])
    optimizer_rf = torch.optim.Adam(model_rf.parameters(), lr=0.03)
    
    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        data = sample_true(batch_size)
        
        optimizer_rf.zero_grad()
        loss = cd_loss_reinforce(model_rf, data, integrator, dt, cd_steps, kT)
        loss.backward()
        optimizer_rf.step()
        
        results['reinforce']['losses'].append(loss.item())
        results['reinforce']['means'].append(model_rf.means.detach().clone())
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    return results, true_means


# =============================================================================
# Experiment 3: Persistent CD vs Standard CD
# =============================================================================

def experiment_persistent_cd():
    """Compare standard CD with Persistent CD."""
    print("\n" + "="*60)
    print("Experiment 3: Standard CD vs Persistent CD")
    print("="*60)
    
    torch.manual_seed(456)
    
    # Bimodal target
    true_means = torch.tensor([[-2.0], [2.0]])
    true_vars = torch.tensor([0.4, 0.4])
    true_weights = torch.tensor([0.5, 0.5])
    
    def sample_true(n):
        indices = torch.multinomial(true_weights, n, replacement=True)
        samples = torch.randn(n, 1) * torch.sqrt(true_vars[indices]).unsqueeze(-1)
        samples = samples + true_means[indices]
        return samples
    
    kT = 1.0
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    
    n_epochs = 250
    batch_size = 256
    n_persistent = 512  # Number of persistent chains
    dt = 0.05
    cd_steps = 5  # Fewer steps since PCD chains are already mixed
    
    results = {
        'standard': {'losses': [], 'means': []},
        'persistent': {'losses': [], 'means': []},
    }
    
    # Standard CD
    print("\n--- Standard CD-5 ---")
    model_std = GaussianMixtureEBM(n_components=2, dim=1)
    with torch.no_grad():
        model_std.means.data = torch.tensor([[0.0], [0.5]])
    optimizer_std = torch.optim.Adam(model_std.parameters(), lr=0.04)
    
    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        data = sample_true(batch_size)
        
        optimizer_std.zero_grad()
        loss = cd_loss_bptt(model_std, data, integrator, dt, cd_steps)
        loss.backward()
        optimizer_std.step()
        
        results['standard']['losses'].append(loss.item())
        results['standard']['means'].append(model_std.means.detach().clone())
    
    # Persistent CD
    print("\n--- Persistent CD-5 ---")
    model_pcd = GaussianMixtureEBM(n_components=2, dim=1)
    with torch.no_grad():
        model_pcd.means.data = torch.tensor([[0.0], [0.5]])
    optimizer_pcd = torch.optim.Adam(model_pcd.parameters(), lr=0.04)
    
    # Initialize persistent chains from prior
    persistent_chains = torch.randn(n_persistent, 1) * 2
    
    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        data = sample_true(batch_size)
        
        optimizer_pcd.zero_grad()
        loss, persistent_chains = persistent_cd_loss(
            model_pcd, data, persistent_chains, integrator, dt, cd_steps
        )
        loss.backward()
        optimizer_pcd.step()
        
        results['persistent']['losses'].append(loss.item())
        results['persistent']['means'].append(model_pcd.means.detach().clone())
        
        if epoch % 80 == 0:
            print(f"  Epoch {epoch}: std_loss={results['standard']['losses'][epoch]:.4f}, "
                  f"pcd_loss={loss.item():.4f}")
    
    return results, true_means


# =============================================================================
# Experiment 4: Neural EBM on 2D Data
# =============================================================================

def experiment_neural_ebm_2d():
    """Train a neural EBM on 2D data using CD."""
    print("\n" + "="*60)
    print("Experiment 4: Neural EBM on 2D Data")
    print("="*60)
    
    torch.manual_seed(789)
    
    # Target: 2D mixture (spiral-like)
    def sample_true(n):
        # Two clusters
        n1 = n // 2
        n2 = n - n1
        
        # Cluster 1
        theta1 = torch.randn(n1) * 0.3 + 0.5
        r1 = 1.5 + torch.randn(n1) * 0.2
        x1 = torch.stack([r1 * torch.cos(theta1 * 2 * np.pi), 
                          r1 * torch.sin(theta1 * 2 * np.pi)], dim=-1)
        
        # Cluster 2
        theta2 = torch.randn(n2) * 0.3 + 0.0
        r2 = 1.5 + torch.randn(n2) * 0.2
        x2 = torch.stack([r2 * torch.cos(theta2 * 2 * np.pi), 
                          r2 * torch.sin(theta2 * 2 * np.pi)], dim=-1)
        
        return torch.cat([x1, x2], dim=0)
    
    # Model
    model = NeuralEBM(dim=2, hidden_dim=64)
    
    kT = 0.5
    integrator = OverdampedLangevin(gamma=1.0, kT=kT)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    n_epochs = 400
    batch_size = 256
    dt = 0.02
    cd_steps = 20
    
    losses = []
    samples_history = []
    
    for epoch in range(n_epochs):
        data = sample_true(batch_size)
        
        optimizer.zero_grad()
        loss = cd_loss_bptt(model, data, integrator, dt, cd_steps)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            # Generate samples for visualization
            with torch.no_grad():
                init = torch.randn(500, 2) * 2
                samples = integrator.run(init, model.force, dt=dt, n_steps=200, final_only=True)[0]
                samples_history.append((epoch, samples.clone()))
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")
    
    # Final samples
    with torch.no_grad():
        init = torch.randn(1000, 2) * 2
        final_samples = integrator.run(init, model.force, dt=dt, n_steps=500, final_only=True)[0]
    
    return losses, samples_history, final_samples, sample_true


# =============================================================================
# Visualization
# =============================================================================

def plot_results(exp1, exp2, exp3, exp4, save_path):
    """Create comprehensive visualization."""
    
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # --- Row 1: Experiment 1 - CD-k comparison ---
    results_1, true_means_1, true_weights_1, sample_true_1 = exp1
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    for k, color in zip([1, 5, 20], [COLORS['red'], COLORS['blue'], COLORS['green']]):
        ax1.plot(results_1[k]['losses'], label=f'CD-{k}', color=color, lw=LW)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CD Loss')
    ax1.set_title('(a) CD-k Loss Comparison')
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, len(results_1[1]['losses']))
    
    # Mean trajectories
    ax2 = fig.add_subplot(gs[0, 1])
    for k, color in zip([1, 5, 20], [COLORS['red'], COLORS['blue'], COLORS['green']]):
        means = torch.stack(results_1[k]['means'])  # (epochs, 2, 1)
        ax2.plot(means[:, 0, 0].numpy(), label=f'CD-{k} μ₁', color=color, lw=LW, ls='-')
        ax2.plot(means[:, 1, 0].numpy(), color=color, lw=LW, ls='--')
    ax2.axhline(true_means_1[0, 0].item(), color=COLORS['gray'], ls=':', lw=1.5, label='True')
    ax2.axhline(true_means_1[1, 0].item(), color=COLORS['gray'], ls=':', lw=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learned Means')
    ax2.set_title('(b) Mean Parameter Evolution')
    ax2.legend(loc='upper right', fontsize=8)
    
    # Final learned distribution comparison
    ax3 = fig.add_subplot(gs[0, 2:])
    x_plot = torch.linspace(-5, 5, 200).unsqueeze(-1)
    
    # True density (approximate via samples histogram)
    true_samples = sample_true_1(5000)
    ax3.hist(true_samples.numpy(), bins=50, density=True, alpha=0.3, 
             color=COLORS['gray'], label='True data')
    
    # Learned densities for each CD-k
    # Note: We need to compute p(x) ∝ exp(-E(x)) and normalize
    for k, color in zip([1, 5, 20], [COLORS['red'], COLORS['blue'], COLORS['green']]):
        # Reconstruct final model
        model_k = GaussianMixtureEBM(n_components=2, dim=1)
        with torch.no_grad():
            model_k.means.data = results_1[k]['means'][-1]
            model_k.log_weights.data = torch.log(results_1[k]['weights'][-1] + 1e-8)
        
        with torch.no_grad():
            e = model_k.energy(x_plot)
            log_p = -e
            p = torch.exp(log_p - log_p.max())
            p = p / (p.sum() * (x_plot[1] - x_plot[0]))
        ax3.plot(x_plot.numpy(), p.numpy(), color=color, lw=LW, label=f'CD-{k}')
    
    ax3.set_xlabel('x')
    ax3.set_ylabel('Density')
    ax3.set_title('(c) Learned Distributions')
    ax3.legend(loc='upper right')
    ax3.set_xlim(-5, 5)
    
    # --- Row 2: Experiment 2 - BPTT vs REINFORCE ---
    results_2, true_means_2 = exp2
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(results_2['bptt']['losses'], label='BPTT', color=COLORS['bptt'], lw=LW)
    ax4.plot(results_2['reinforce']['losses'], label='REINFORCE', color=COLORS['reinforce'], lw=LW)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('CD Loss')
    ax4.set_title('(d) BPTT vs REINFORCE')
    ax4.legend()
    
    ax5 = fig.add_subplot(gs[1, 1])
    means_bptt = torch.stack(results_2['bptt']['means'])
    means_rf = torch.stack(results_2['reinforce']['means'])
    ax5.plot(means_bptt[:, 0, 0].numpy(), color=COLORS['bptt'], lw=LW, label='BPTT μ₁')
    ax5.plot(means_bptt[:, 1, 0].numpy(), color=COLORS['bptt'], lw=LW, ls='--')
    ax5.plot(means_rf[:, 0, 0].numpy(), color=COLORS['reinforce'], lw=LW, label='RF μ₁')
    ax5.plot(means_rf[:, 1, 0].numpy(), color=COLORS['reinforce'], lw=LW, ls='--')
    ax5.axhline(true_means_2[0, 0].item(), color=COLORS['gray'], ls=':', lw=1.5)
    ax5.axhline(true_means_2[1, 0].item(), color=COLORS['gray'], ls=':', lw=1.5)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Learned Means')
    ax5.set_title('(e) Parameter Convergence')
    ax5.legend(loc='upper right', fontsize=8)
    
    # --- Row 2 continued: Experiment 3 - Standard vs Persistent CD ---
    results_3, true_means_3 = exp3
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(results_3['standard']['losses'], label='Standard CD', color=COLORS['orange'], lw=LW)
    ax6.plot(results_3['persistent']['losses'], label='Persistent CD', color=COLORS['purple'], lw=LW)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('CD Loss')
    ax6.set_title('(f) Standard vs Persistent CD')
    ax6.legend()
    
    ax7 = fig.add_subplot(gs[1, 3])
    means_std = torch.stack(results_3['standard']['means'])
    means_pcd = torch.stack(results_3['persistent']['means'])
    ax7.plot(means_std[:, 0, 0].numpy(), color=COLORS['orange'], lw=LW, label='Std μ₁')
    ax7.plot(means_std[:, 1, 0].numpy(), color=COLORS['orange'], lw=LW, ls='--')
    ax7.plot(means_pcd[:, 0, 0].numpy(), color=COLORS['purple'], lw=LW, label='PCD μ₁')
    ax7.plot(means_pcd[:, 1, 0].numpy(), color=COLORS['purple'], lw=LW, ls='--')
    ax7.axhline(true_means_3[0, 0].item(), color=COLORS['gray'], ls=':', lw=1.5)
    ax7.axhline(true_means_3[1, 0].item(), color=COLORS['gray'], ls=':', lw=1.5)
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Learned Means')
    ax7.set_title('(g) PCD Convergence')
    ax7.legend(loc='upper right', fontsize=8)
    
    # --- Row 3: Experiment 4 - 2D Neural EBM ---
    losses_4, samples_hist, final_samples, sample_true_4 = exp4
    
    ax8 = fig.add_subplot(gs[2, 0])
    ax8.plot(losses_4, color=COLORS['neural'], lw=LW)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('CD Loss')
    ax8.set_title('(h) Neural EBM Training')
    
    # Evolution of samples
    for idx, (ax_idx, title) in enumerate([(gs[2, 1], '(i) Epoch 0'), 
                                            (gs[2, 2], '(j) Epoch 100')]):
        ax = fig.add_subplot(ax_idx)
        
        # True data
        true_data = sample_true_4(500)
        ax.scatter(true_data[:, 0].numpy(), true_data[:, 1].numpy(), 
                   alpha=0.3, s=10, c=COLORS['gray'], label='Data')
        
        # Model samples
        if idx < len(samples_hist):
            epoch, samples = samples_hist[idx]
            ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(),
                       alpha=0.5, s=10, c=COLORS['neural'], label='Model')
        
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(title)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_aspect('equal')
    
    # Final samples
    ax11 = fig.add_subplot(gs[2, 3])
    true_data = sample_true_4(500)
    ax11.scatter(true_data[:, 0].numpy(), true_data[:, 1].numpy(),
                 alpha=0.3, s=10, c=COLORS['gray'], label='Data')
    ax11.scatter(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(),
                 alpha=0.5, s=10, c=COLORS['neural'], label='Model')
    ax11.set_xlabel('x₁')
    ax11.set_ylabel('x₂')
    ax11.set_title('(k) Final Samples')
    ax11.set_xlim(-4, 4)
    ax11.set_ylim(-4, 4)
    ax11.legend(loc='upper right', fontsize=8)
    ax11.set_aspect('equal')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to {save_path}")
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("Contrastive Divergence via Differentiable Simulation")
    print("="*60)
    print("""
This demo shows how differentiable Langevin dynamics enables
efficient training of energy-based models using Contrastive Divergence.

Key insights:
1. CD-k with larger k gives better gradients but costs more
2. BPTT through dynamics has lower variance than REINFORCE
3. Persistent CD improves mixing without longer chains
4. Neural EBMs can learn complex 2D distributions
""")
    
    # Run experiments
    exp1 = experiment_1d_gmm()
    exp2 = experiment_bptt_vs_reinforce()
    exp3 = experiment_persistent_cd()
    exp4 = experiment_neural_ebm_2d()
    
    # Plot results
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    save_path = os.path.join(assets_dir, "demo_contrastive_divergence.png")
    
    plot_results(exp1, exp2, exp3, exp4, save_path)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("""
Contrastive Divergence successfully trained EBMs by:
- Using differentiable Langevin dynamics for the negative phase
- Comparing BPTT (low variance) vs REINFORCE (memory efficient)
- Demonstrating Persistent CD for improved mixing
- Training neural EBMs on 2D distributions

The differentiable simulation framework makes CD training natural:
- Gradients flow through MCMC steps via automatic differentiation
- No need for separate gradient estimators
- Seamless integration with PyTorch optimizers
""")


if __name__ == "__main__":
    main()

