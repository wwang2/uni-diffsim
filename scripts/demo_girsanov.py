#!/usr/bin/env python3
"""
Demo: Girsanov Estimator — Behavior, Variance, and Practical Guidelines

This script provides a comprehensive exploration of the Girsanov path-reweighting
gradient estimator, demonstrating:

1. **Theoretical Foundation**: How the Girsanov formula works for Langevin dynamics
2. **Variance vs Trajectory Length**: The exponential growth of variance with T
3. **Observable Complexity**: Simple vs path-dependent observables
4. **Effective Sample Size**: How importance sampling efficiency degrades
5. **Comparison with BPTT**: When Girsanov agrees vs diverges from ground truth
6. **Practical Guidelines**: When to use Girsanov vs alternatives

═══════════════════════════════════════════════════════════════════════════════
THEORETICAL BACKGROUND
═══════════════════════════════════════════════════════════════════════════════

For overdamped Langevin dynamics:
    dx = F_θ(x)dt + σdW

The path probability density (Girsanov theorem) is:
    p(τ|θ) ∝ exp[ (1/σ²) ∫₀ᵀ F_θ(x_t) · dx_t - (1/2σ²) ∫₀ᵀ |F_θ(x_t)|² dt ]

Taking the gradient w.r.t. θ gives the log path score:
    ∇_θ log p(τ|θ) = (1/σ²) ∫₀ᵀ ∇_θF_θ(x_t) · (dx_t - F_θ(x_t)dt)

This enables the score function (REINFORCE-style) estimator in path space:
    ∇_θ ⟨O⟩ = ⟨O · ∇_θ log p(τ|θ)⟩

Key insight: The integrand (dx_t - F_θ(x_t)dt) ≈ σdW_t is the noise increment.
Over T steps, these accumulate, causing variance to grow with trajectory length.

═══════════════════════════════════════════════════════════════════════════════
WHEN TO USE GIRSANOV
═══════════════════════════════════════════════════════════════════════════════

✓ USE GIRSANOV WHEN:
  • Trajectories are short (T < relaxation time)
  • Observable depends only on final state or simple time averages
  • BPTT memory is prohibitive
  • Problem is non-equilibrium (REINFORCE doesn't apply)

✗ AVOID GIRSANOV WHEN:
  • Trajectories are long (variance explodes)
  • Observable is path-dependent (FPT, max, integrals over rare events)
  • High accuracy is required
  • System is high-dimensional (variance compounds across dimensions)

═══════════════════════════════════════════════════════════════════════════════

Author: uni-diffsim
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uni_diffsim import OverdampedLangevin, Harmonic, DoubleWell, AsymmetricDoubleWell
from uni_diffsim.gradient_estimators import GirsanovEstimator, ReinforceEstimator
from uni_diffsim.plotting import apply_style, COLORS, LW, MS

apply_style()


def soft_indicator(x, threshold=0.0, sigma=0.1):
    """Differentiable approximation to indicator x > threshold."""
    return torch.sigmoid((x - threshold) / sigma)


def soft_fpt(traj, threshold=0.0, sigma=0.1, dt=0.01):
    """Differentiable approximation to first passage time.
    
    Computes expected hitting time using survival probability:
        τ = ∫₀^∞ P(not crossed by time t) dt ≈ Σ_t S(t) * dt
    
    Args:
        traj: (n_steps, dim) or (n_steps, n_walkers, dim)
        threshold: Crossing threshold
        sigma: Softness of indicator
        dt: Time step
    
    Returns:
        First passage time estimate per trajectory
    """
    x = traj[..., 0]  # [T] or [T, N]
    crossed = soft_indicator(x, threshold, sigma)
    log_survival = torch.cumsum(torch.log(1 - crossed + 1e-8), dim=0)
    survival = torch.exp(log_survival)
    return survival.sum(dim=0) * dt  # Sum over time


def time_integral(traj, dt=0.01):
    """Time integral of position: ∫₀ᵀ x(t) dt."""
    return traj[..., 0].sum(dim=0) * dt


# =============================================================================
# Experiment 1: Variance vs Trajectory Length (Harmonic)
# =============================================================================

def experiment_variance_vs_length():
    """
    Show how Girsanov variance grows with trajectory length.
    
    For a harmonic potential U(x) = k*x²/2, the equilibrium distribution is
    Gaussian with variance σ² = kT/k. The gradient d⟨x²⟩/dk = -kT/k².
    
    We compare BPTT (exact) vs Girsanov across trajectory lengths.
    """
    print("\n" + "="*70)
    print("[1/6] Variance vs Trajectory Length (Harmonic Potential)")
    print("="*70)
    print("Theory: d⟨x²⟩/dk = -kT/k² = -1.0 for k=kT=1")
    
    kT = 1.0
    k = 1.0
    dt = 0.01
    n_walkers = 300
    n_trials = 25
    
    # Test different trajectory lengths
    n_steps_list = [10, 25, 50, 100, 200, 500, 1000, 2000]
    
    results = {
        'n_steps': n_steps_list,
        'T': [n * dt for n in n_steps_list],
        'bptt_mean': [], 'bptt_std': [],
        'girsanov_mean': [], 'girsanov_std': [],
        'theory': -kT / k**2,
    }
    
    print(f"\n{'T (s)':<10} {'BPTT':<20} {'Girsanov':<20} {'Gir/BPTT Ratio':<15}")
    print("-" * 65)
    
    for n_steps in n_steps_list:
        bptt_grads = []
        gir_grads = []
        
        for trial in range(n_trials):
            torch.manual_seed(trial * 100 + n_steps)
            x0 = torch.randn(n_walkers, 1)
            
            # BPTT (ground truth)
            potential_bptt = Harmonic(k=k)
            integrator = OverdampedLangevin(gamma=1.0, kT=kT)
            traj = integrator.run(x0, potential_bptt.force, dt=dt, n_steps=n_steps)
            
            # Observable: time-averaged ⟨x²⟩
            var = (traj**2).mean()
            grad_bptt = torch.autograd.grad(var, potential_bptt.k)[0]
            bptt_grads.append(grad_bptt.item())
            
            # Girsanov
            potential_gir = Harmonic(k=k)
            estimator = GirsanovEstimator(potential_gir, sigma=np.sqrt(2*kT))
            traj_gir = traj.detach().transpose(0, 1)  # (n_walkers, n_steps, 1)
            
            # Observable: time-averaged ⟨x²⟩ per trajectory
            obs_gir = lambda t: (t**2).mean(dim=(1, 2))  # (n_walkers,)
            grads_gir = estimator.estimate_gradient(traj_gir, observable=obs_gir, dt=dt)
            gir_grads.append(grads_gir['k'].item())
        
        bptt_mean, bptt_std = np.mean(bptt_grads), np.std(bptt_grads)
        gir_mean, gir_std = np.mean(gir_grads), np.std(gir_grads)
        
        results['bptt_mean'].append(bptt_mean)
        results['bptt_std'].append(bptt_std)
        results['girsanov_mean'].append(gir_mean)
        results['girsanov_std'].append(gir_std)
        
        ratio = gir_mean / bptt_mean if abs(bptt_mean) > 1e-6 else np.nan
        print(f"{n_steps*dt:<10.2f} {bptt_mean:>7.3f} ± {bptt_std:<8.3f} "
              f"{gir_mean:>7.3f} ± {gir_std:<8.3f} {ratio:>10.2f}x")
    
    return results


# =============================================================================
# Experiment 2: Simple vs Path-Dependent Observables
# =============================================================================

def experiment_observable_types():
    """
    Compare Girsanov performance on different observable types.
    
    Simple observables (final state, averages) should work better than
    path-dependent observables (first passage time, time integrals).
    """
    print("\n" + "="*70)
    print("[2/6] Simple vs Path-Dependent Observables (Double Well)")
    print("="*70)
    
    kT = 0.5
    dt = 0.01
    n_steps = 300
    n_walkers = 400
    n_trials = 20
    
    barrier_values = np.linspace(0.5, 2.0, 7)
    
    results = {
        'barrier': barrier_values,
        # Observable 1: Final position P(x_T > 0)
        'final_bptt_mean': [], 'final_bptt_std': [],
        'final_gir_mean': [], 'final_gir_std': [],
        # Observable 2: Time-averaged position ⟨x⟩_T
        'avg_bptt_mean': [], 'avg_bptt_std': [],
        'avg_gir_mean': [], 'avg_gir_std': [],
        # Observable 3: First passage time (path-dependent)
        'fpt_bptt_mean': [], 'fpt_bptt_std': [],
        'fpt_gir_mean': [], 'fpt_gir_std': [],
    }
    
    print(f"\n{'Barrier':<10} {'Final Pos (BPTT/Gir)':<25} {'Time Avg (BPTT/Gir)':<25} {'FPT (BPTT/Gir)':<25}")
    print("-" * 85)
    
    for b in barrier_values:
        final_bptt, final_gir = [], []
        avg_bptt, avg_gir = [], []
        fpt_bptt, fpt_gir = [], []
        
        for trial in range(n_trials):
            torch.manual_seed(trial * 100 + int(b * 10))
            x0 = torch.full((n_walkers, 1), -1.0)  # Start in left well
            
            # === Run simulation ===
            potential = DoubleWell(barrier_height=b)
            integrator = OverdampedLangevin(gamma=1.0, kT=kT)
            traj = integrator.run(x0, potential.force, dt=dt, n_steps=n_steps)
            
            # === Observable 1: Final position P(x_T > 0) ===
            x_final = traj[-1, :, 0]
            p_right = soft_indicator(x_final, 0.0, 0.1).mean()
            grad = torch.autograd.grad(p_right, potential.barrier_height)[0]
            final_bptt.append(grad.item())
            
            potential_gir = DoubleWell(barrier_height=b)
            estimator = GirsanovEstimator(potential_gir, sigma=np.sqrt(2*kT))
            traj_gir = traj.detach().transpose(0, 1)
            obs_final = lambda t: soft_indicator(t[:, -1, 0], 0.0, 0.1)
            grads = estimator.estimate_gradient(traj_gir, observable=obs_final, dt=dt)
            final_gir.append(grads['barrier_height'].item())
            
            # === Observable 2: Time-averaged position ===
            potential2 = DoubleWell(barrier_height=b)
            traj2 = integrator.run(x0.clone(), potential2.force, dt=dt, n_steps=n_steps)
            mean_x = traj2.mean()
            grad_avg = torch.autograd.grad(mean_x, potential2.barrier_height)[0]
            avg_bptt.append(grad_avg.item())
            
            potential_gir2 = DoubleWell(barrier_height=b)
            estimator2 = GirsanovEstimator(potential_gir2, sigma=np.sqrt(2*kT))
            traj_gir2 = traj2.detach().transpose(0, 1)
            obs_avg = lambda t: t.mean(dim=(1, 2))
            grads2 = estimator2.estimate_gradient(traj_gir2, observable=obs_avg, dt=dt)
            avg_gir.append(grads2['barrier_height'].item())
            
            # === Observable 3: First passage time ===
            potential3 = DoubleWell(barrier_height=b)
            traj3 = integrator.run(x0.clone(), potential3.force, dt=dt, n_steps=n_steps)
            fpt_val = soft_fpt(traj3, threshold=0.0, sigma=0.1, dt=dt).mean()
            grad_fpt = torch.autograd.grad(fpt_val, potential3.barrier_height)[0]
            fpt_bptt.append(grad_fpt.item())
            
            potential_gir3 = DoubleWell(barrier_height=b)
            estimator3 = GirsanovEstimator(potential_gir3, sigma=np.sqrt(2*kT))
            traj_gir3 = traj3.detach().transpose(0, 1)
            obs_fpt = lambda t: soft_fpt(t.transpose(0, 1), threshold=0.0, sigma=0.1, dt=dt)
            grads3 = estimator3.estimate_gradient(traj_gir3, observable=obs_fpt, dt=dt)
            fpt_gir.append(grads3['barrier_height'].item())
        
        # Store results
        results['final_bptt_mean'].append(np.mean(final_bptt))
        results['final_bptt_std'].append(np.std(final_bptt))
        results['final_gir_mean'].append(np.mean(final_gir))
        results['final_gir_std'].append(np.std(final_gir))
        
        results['avg_bptt_mean'].append(np.mean(avg_bptt))
        results['avg_bptt_std'].append(np.std(avg_bptt))
        results['avg_gir_mean'].append(np.mean(avg_gir))
        results['avg_gir_std'].append(np.std(avg_gir))
        
        results['fpt_bptt_mean'].append(np.mean(fpt_bptt))
        results['fpt_bptt_std'].append(np.std(fpt_bptt))
        results['fpt_gir_mean'].append(np.mean(fpt_gir))
        results['fpt_gir_std'].append(np.std(fpt_gir))
        
        print(f"{b:<10.1f} {np.mean(final_bptt):>6.3f} / {np.mean(final_gir):<12.3f} "
              f"{np.mean(avg_bptt):>6.3f} / {np.mean(avg_gir):<12.3f} "
              f"{np.mean(fpt_bptt):>6.3f} / {np.mean(fpt_gir):<12.3f}")
    
    return results


# =============================================================================
# Experiment 3: Log Score Distribution Analysis
# =============================================================================

def experiment_score_distribution():
    """
    Visualize how log path score distribution changes with trajectory length.
    
    The log score ∇_θ log p(τ|θ) is a sum of noise-weighted force gradients.
    As T increases, this sum has growing variance (central limit theorem),
    but the tails become heavier, causing importance sampling to fail.
    """
    print("\n" + "="*70)
    print("[3/6] Log Score Distribution Analysis")
    print("="*70)
    
    kT = 1.0
    dt = 0.01
    n_walkers = 2000
    
    results = {
        'short': {'n_steps': 50, 'T': 0.5, 'scores': None},
        'medium': {'n_steps': 200, 'T': 2.0, 'scores': None},
        'long': {'n_steps': 1000, 'T': 10.0, 'scores': None},
        'very_long': {'n_steps': 5000, 'T': 50.0, 'scores': None},
    }
    
    torch.manual_seed(42)
    
    print(f"\n{'Length':<12} {'T (s)':<8} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10} {'Kurtosis':<10}")
    print("-" * 70)
    
    for key, cfg in results.items():
        n_steps = cfg['n_steps']
        x0 = torch.randn(n_walkers, 1)
        
        potential = Harmonic(k=1.0)
        integrator = OverdampedLangevin(gamma=1.0, kT=kT)
        traj = integrator.run(x0, potential.force, dt=dt, n_steps=n_steps)
        
        # Compute log scores
        estimator = GirsanovEstimator(potential, sigma=np.sqrt(2*kT))
        traj_for_score = traj  # (n_steps, n_walkers, 1)
        log_scores = estimator.compute_log_path_score(traj_for_score, dt)
        
        scores_np = log_scores.detach().numpy()
        results[key]['scores'] = scores_np
        
        # Compute kurtosis (excess kurtosis: normal = 0)
        kurtosis = ((scores_np - scores_np.mean())**4).mean() / (scores_np.std()**4) - 3
        
        print(f"{key:<12} {cfg['T']:<8.1f} {scores_np.mean():<10.2f} {scores_np.std():<10.2f} "
              f"{scores_np.min():<10.1f} {scores_np.max():<10.1f} {kurtosis:<10.2f}")
    
    return results


# =============================================================================
# Experiment 4: Effective Sample Size Decay
# =============================================================================

def experiment_effective_samples():
    """
    Show how effective sample size (ESS) decreases with trajectory length.
    
    ESS measures how many "effective" independent samples we have after
    importance weighting. When ESS << N, the estimate is dominated by
    a few trajectories with extreme weights.
    
    ESS = (Σ w_i)² / Σ w_i² = 1 / Σ w̃_i²  where w̃ are normalized weights
    """
    print("\n" + "="*70)
    print("[4/6] Effective Sample Size Decay")
    print("="*70)
    
    kT = 1.0
    dt = 0.01
    n_walkers = 1000
    
    n_steps_list = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    
    results = {
        'n_steps': n_steps_list,
        'T': [n * dt for n in n_steps_list],
        'ess': [],
        'ess_ratio': [],
        'max_weight_ratio': [],  # Max weight / mean weight
    }
    
    torch.manual_seed(42)
    
    print(f"\n{'T (s)':<10} {'ESS':<15} {'ESS/N':<12} {'Max/Mean Weight':<15}")
    print("-" * 55)
    
    for n_steps in n_steps_list:
        x0 = torch.randn(n_walkers, 1)
        
        potential = Harmonic(k=1.0)
        integrator = OverdampedLangevin(gamma=1.0, kT=kT)
        traj = integrator.run(x0, potential.force, dt=dt, n_steps=n_steps)
        
        # Compute log scores
        estimator = GirsanovEstimator(potential, sigma=np.sqrt(2*kT))
        log_scores = estimator.compute_log_path_score(traj, dt)
        
        # Compute importance weights (self-normalized)
        log_weights = log_scores - log_scores.max()  # Numerical stability
        weights = torch.exp(log_weights)
        weights_norm = weights / weights.sum()
        
        # Effective sample size: ESS = 1 / Σ w̃²
        ess = 1.0 / (weights_norm**2).sum().item()
        ess_ratio = ess / n_walkers
        
        # Max weight ratio
        max_weight_ratio = (weights.max() / weights.mean()).item()
        
        results['ess'].append(ess)
        results['ess_ratio'].append(ess_ratio)
        results['max_weight_ratio'].append(max_weight_ratio)
        
        print(f"{n_steps*dt:<10.2f} {ess:<15.1f} {ess_ratio:<12.1%} {max_weight_ratio:<15.1f}")
    
    return results


# =============================================================================
# Experiment 5: Girsanov vs REINFORCE for Equilibrium
# =============================================================================

def experiment_girsanov_vs_reinforce():
    """
    Compare Girsanov and REINFORCE for equilibrium observables.
    
    For equilibrium, REINFORCE uses:
        ∇_θ ⟨O⟩ = -β Cov(O, ∇_θU)
    
    This should have lower variance than Girsanov for long trajectories
    because it doesn't accumulate path noise.
    """
    print("\n" + "="*70)
    print("[5/6] Girsanov vs REINFORCE (Equilibrium Observable)")
    print("="*70)
    
    kT = 1.0
    beta = 1.0 / kT
    dt = 0.01
    n_walkers = 300
    n_trials = 20
    burn_in_ratio = 0.2  # Discard first 20% as burn-in
    
    n_steps_list = [100, 200, 500, 1000, 2000]
    
    results = {
        'n_steps': n_steps_list,
        'T': [n * dt for n in n_steps_list],
        'bptt_mean': [], 'bptt_std': [],
        'girsanov_mean': [], 'girsanov_std': [],
        'reinforce_mean': [], 'reinforce_std': [],
        'theory': -kT,  # d⟨x²⟩/dk = -kT/k² = -1 for k=1
    }
    
    print(f"\n{'T (s)':<8} {'BPTT':<18} {'Girsanov':<18} {'REINFORCE':<18}")
    print("-" * 65)
    
    for n_steps in n_steps_list:
        bptt_grads, gir_grads, rf_grads = [], [], []
        burn_in = int(n_steps * burn_in_ratio)
        
        for trial in range(n_trials):
            torch.manual_seed(trial * 100 + n_steps)
            x0 = torch.randn(n_walkers, 1)
            
            # === BPTT ===
            potential_bptt = Harmonic(k=1.0)
            integrator = OverdampedLangevin(gamma=1.0, kT=kT)
            traj = integrator.run(x0, potential_bptt.force, dt=dt, n_steps=n_steps)
            
            samples = traj[burn_in:]
            var = (samples**2).mean()
            grad_bptt = torch.autograd.grad(var, potential_bptt.k)[0]
            bptt_grads.append(grad_bptt.item())
            
            # === Girsanov ===
            potential_gir = Harmonic(k=1.0)
            estimator_gir = GirsanovEstimator(potential_gir, sigma=np.sqrt(2*kT))
            traj_gir = traj.detach().transpose(0, 1)
            obs_gir = lambda t: (t[:, burn_in:]**2).mean(dim=(1, 2))
            grads_gir = estimator_gir.estimate_gradient(traj_gir, observable=obs_gir, dt=dt)
            gir_grads.append(grads_gir['k'].item())
            
            # === REINFORCE ===
            samples_flat = samples.reshape(-1, 1).detach()
            potential_rf = Harmonic(k=1.0)
            estimator_rf = ReinforceEstimator(potential_rf, beta=beta)
            obs_rf = lambda x: (x**2).sum(dim=-1)
            grads_rf = estimator_rf.estimate_gradient(samples_flat, observable=obs_rf)
            rf_grads.append(grads_rf['k'].item())
        
        results['bptt_mean'].append(np.mean(bptt_grads))
        results['bptt_std'].append(np.std(bptt_grads))
        results['girsanov_mean'].append(np.mean(gir_grads))
        results['girsanov_std'].append(np.std(gir_grads))
        results['reinforce_mean'].append(np.mean(rf_grads))
        results['reinforce_std'].append(np.std(rf_grads))
        
        print(f"{n_steps*dt:<8.1f} {np.mean(bptt_grads):>6.3f} ± {np.std(bptt_grads):<7.3f} "
              f"{np.mean(gir_grads):>6.3f} ± {np.std(gir_grads):<7.3f} "
              f"{np.mean(rf_grads):>6.3f} ± {np.std(rf_grads):<7.3f}")
    
    return results


# =============================================================================
# Experiment 6: Optimization with Girsanov
# =============================================================================

def experiment_optimization():
    """
    Test whether Girsanov gradients can be used for optimization.
    
    Task: Find the spring constant k such that ⟨x²⟩ = target_var.
    Compare optimization trajectories using BPTT vs Girsanov gradients.
    """
    print("\n" + "="*70)
    print("[6/6] Optimization with Girsanov Gradients")
    print("="*70)
    
    kT = 1.0
    dt = 0.01
    n_steps = 100  # Short trajectory for Girsanov to work
    n_walkers = 200
    n_epochs = 40
    target_var = 0.5  # Target ⟨x²⟩ = 0.5, so optimal k = kT/0.5 = 2.0
    
    results = {
        'epochs': list(range(n_epochs)),
        'bptt_loss': [], 'bptt_k': [],
        'girsanov_loss': [], 'girsanov_k': [],
        'target_k': kT / target_var,
    }
    
    print(f"\nTarget: ⟨x²⟩ = {target_var}, optimal k = {kT/target_var:.2f}")
    print(f"Starting k = 0.5, trajectory length T = {n_steps*dt:.1f}s")
    print(f"\n{'Epoch':<8} {'BPTT k':<12} {'BPTT Loss':<12} {'Gir k':<12} {'Gir Loss':<12}")
    print("-" * 55)
    
    # BPTT optimization
    k_bptt = 0.5
    lr_bptt = 0.3
    
    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        x0 = torch.randn(n_walkers, 1)
        
        k_tensor = torch.tensor([k_bptt], requires_grad=True)
        def force_fn(x):
            return -k_tensor * x
        
        integrator = OverdampedLangevin(gamma=1.0, kT=kT)
        traj = integrator.run(x0, force_fn, dt=dt, n_steps=n_steps)
        
        var = (traj**2).mean()
        loss = (var - target_var)**2
        loss.backward()
        
        results['bptt_loss'].append(loss.item())
        results['bptt_k'].append(k_bptt)
        
        k_bptt = max(0.1, k_bptt - lr_bptt * k_tensor.grad.item())
    
    # Girsanov optimization
    k_gir = 0.5
    lr_gir = 0.05  # Lower learning rate due to noisy gradients
    
    for epoch in range(n_epochs):
        torch.manual_seed(epoch)
        x0 = torch.randn(n_walkers, 1)
        
        with torch.no_grad():
            potential_sim = Harmonic(k=k_gir)
            integrator = OverdampedLangevin(gamma=1.0, kT=kT)
            traj = integrator.run(x0, potential_sim.force, dt=dt, n_steps=n_steps)
        
        var = (traj**2).mean().item()
        loss = (var - target_var)**2
        
        # Girsanov gradient
        potential_gir = Harmonic(k=k_gir)
        estimator = GirsanovEstimator(potential_gir, sigma=np.sqrt(2*kT))
        traj_gir = traj.transpose(0, 1)
        obs_gir = lambda t: (t**2).mean(dim=(1, 2))
        grads = estimator.estimate_gradient(traj_gir, observable=obs_gir, dt=dt)
        
        # Chain rule: d(loss)/dk = 2*(var - target) * d(var)/dk
        grad_loss = 2 * (var - target_var) * grads['k'].item()
        
        results['girsanov_loss'].append(loss)
        results['girsanov_k'].append(k_gir)
        
        k_gir = max(0.1, k_gir - lr_gir * grad_loss)
        
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"{epoch:<8} {results['bptt_k'][epoch]:<12.3f} {results['bptt_loss'][epoch]:<12.4f} "
                  f"{results['girsanov_k'][epoch]:<12.3f} {results['girsanov_loss'][epoch]:<12.4f}")
    
    print(f"\nFinal: BPTT k = {results['bptt_k'][-1]:.3f}, Girsanov k = {results['girsanov_k'][-1]:.3f}, "
          f"Target k = {results['target_k']:.3f}")
    
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(exp1, exp2, exp3, exp4, exp5, exp6, save_path):
    """Create comprehensive 3×2 visualization of Girsanov behavior."""
    
    fig = plt.figure(figsize=(14, 16))
    gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Color scheme
    c_bptt = COLORS['bptt']
    c_gir = COLORS['girsanov']
    c_rf = COLORS['reinforce']
    c_theory = COLORS['theory']
    
    # =========================================================================
    # Panel A: Variance vs Trajectory Length
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    T = np.array(exp1['T'])
    
    ax1.errorbar(T, exp1['bptt_mean'], yerr=exp1['bptt_std'],
                 fmt='o-', color=c_bptt, lw=LW, ms=MS, capsize=3,
                 label='BPTT (exact)', markeredgecolor='white', markeredgewidth=1)
    ax1.errorbar(T, exp1['girsanov_mean'], yerr=exp1['girsanov_std'],
                 fmt='s--', color=c_gir, lw=LW, ms=MS, capsize=3,
                 label='Girsanov', markeredgecolor='white', markeredgewidth=1)
    ax1.axhline(exp1['theory'], color=c_theory, ls=':', lw=2, label=f"Theory: {exp1['theory']:.2f}")
    
    ax1.set_xlabel('Trajectory Length T (s)')
    ax1.set_ylabel(r'$\partial \langle x^2 \rangle / \partial k$')
    ax1.set_title('A. Gradient Estimate vs Trajectory Length\n(Harmonic: theory = -1.0)', fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_xscale('log')
    
    # Annotate bias growth
    ax1.annotate('Bias & variance\ngrow with T', xy=(10, exp1['girsanov_mean'][-2]),
                 xytext=(1.5, -5), fontsize=9, color=c_gir,
                 arrowprops=dict(arrowstyle='->', color=c_gir, lw=1.5))
    
    # =========================================================================
    # Panel B: Observable Types - Final Position
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    barrier = exp2['barrier']
    
    ax2.errorbar(barrier, exp2['final_bptt_mean'], yerr=exp2['final_bptt_std'],
                 fmt='o-', color=c_bptt, lw=LW, ms=MS, capsize=3,
                 label='BPTT', markeredgecolor='white', markeredgewidth=1)
    ax2.errorbar(barrier, exp2['final_gir_mean'], yerr=exp2['final_gir_std'],
                 fmt='s--', color=c_gir, lw=LW, ms=MS, capsize=3,
                 label='Girsanov', markeredgecolor='white', markeredgewidth=1, alpha=0.8)
    
    ax2.set_xlabel('Barrier Height')
    ax2.set_ylabel(r'$\partial P(x_T > 0) / \partial a$')
    ax2.set_title('B. Final Position Observable\n(Simple: Girsanov tracks trend)', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
    
    # =========================================================================
    # Panel C: Observable Types - FPT (path-dependent)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    ax3.errorbar(barrier, exp2['fpt_bptt_mean'], yerr=exp2['fpt_bptt_std'],
                 fmt='o-', color=c_bptt, lw=LW, ms=MS, capsize=3,
                 label='BPTT', markeredgecolor='white', markeredgewidth=1)
    ax3.errorbar(barrier, exp2['fpt_gir_mean'], yerr=exp2['fpt_gir_std'],
                 fmt='s--', color=c_gir, lw=LW, ms=MS, capsize=3,
                 label='Girsanov', markeredgecolor='white', markeredgewidth=1, alpha=0.8)
    
    ax3.set_xlabel('Barrier Height')
    ax3.set_ylabel(r'$\partial \langle \tau \rangle / \partial a$')
    ax3.set_title('C. First Passage Time Observable\n(Path-dependent: Girsanov fails)', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    
    # Highlight the failure
    ax3.fill_between(barrier, 
                     np.array(exp2['fpt_gir_mean']) - np.array(exp2['fpt_gir_std']),
                     np.array(exp2['fpt_gir_mean']) + np.array(exp2['fpt_gir_std']),
                     color=c_gir, alpha=0.15)
    ax3.annotate('Wrong sign\n& magnitude!', xy=(1.5, np.mean(exp2['fpt_gir_mean'])),
                 xytext=(0.7, -5), fontsize=9, color=COLORS['error'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['error'], lw=1.5))
    
    # =========================================================================
    # Panel D: Log Score Distribution
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    colors_dist = ['#A3BE8C', '#5E81AC', '#D08770', '#BF616A']
    labels = [f"T={exp3[k]['T']:.0f}s" for k in ['short', 'medium', 'long', 'very_long']]
    
    for i, (key, color, label) in enumerate(zip(['short', 'medium', 'long', 'very_long'], colors_dist, labels)):
        scores = exp3[key]['scores']
        ax4.hist(scores, bins=50, alpha=0.4, color=color, label=label, density=True)
        ax4.hist(scores, bins=50, histtype='step', color=color, lw=1.5, density=True)
    
    ax4.set_xlabel('Log Path Score')
    ax4.set_ylabel('Density')
    ax4.set_title('D. Distribution of Log Path Scores\n(Wider tails → variance explosion)', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.set_xlim(-15, 8)
    
    # =========================================================================
    # Panel E: Effective Sample Size
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    T_ess = np.array(exp4['T'])
    
    ax5.semilogy(T_ess, exp4['ess_ratio'], 'o-', color=c_gir, lw=LW, ms=MS,
                 markeredgecolor='white', markeredgewidth=1)
    ax5.fill_between(T_ess, 0.001, exp4['ess_ratio'], color=c_gir, alpha=0.2)
    
    ax5.set_xlabel('Trajectory Length T (s)')
    ax5.set_ylabel('ESS / N (log scale)')
    ax5.set_title('E. Effective Sample Size Ratio\n(Importance sampling efficiency)', fontweight='bold')
    ax5.set_xscale('log')
    ax5.set_ylim(0.01, 1.5)
    
    # Threshold lines
    ax5.axhline(0.5, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax5.axhline(0.1, color=COLORS['error'], ls='--', lw=1.5, alpha=0.7)
    ax5.text(0.12, 0.55, '50% efficiency', fontsize=8, color='gray')
    ax5.text(0.12, 0.12, '10% efficiency', fontsize=8, color=COLORS['error'])
    
    # =========================================================================
    # Panel F: Girsanov vs REINFORCE
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    T_cmp = np.array(exp5['T'])
    
    ax6.errorbar(T_cmp, exp5['bptt_mean'], yerr=exp5['bptt_std'],
                 fmt='o-', color=c_bptt, lw=LW, ms=MS, capsize=3,
                 label='BPTT', markeredgecolor='white', markeredgewidth=1)
    ax6.errorbar(T_cmp, exp5['girsanov_mean'], yerr=exp5['girsanov_std'],
                 fmt='s--', color=c_gir, lw=LW, ms=MS, capsize=3,
                 label='Girsanov', markeredgecolor='white', markeredgewidth=1)
    ax6.errorbar(T_cmp, exp5['reinforce_mean'], yerr=exp5['reinforce_std'],
                 fmt='^:', color=c_rf, lw=LW, ms=MS, capsize=3,
                 label='REINFORCE', markeredgecolor='white', markeredgewidth=1)
    ax6.axhline(exp5['theory'], color=c_theory, ls=':', lw=2, alpha=0.7)
    
    ax6.set_xlabel('Trajectory Length T (s)')
    ax6.set_ylabel(r'$\partial \langle x^2 \rangle / \partial k$')
    ax6.set_title('F. Girsanov vs REINFORCE (Equilibrium)\n(REINFORCE has bounded variance)', fontweight='bold')
    ax6.legend(loc='lower left', fontsize=9)
    ax6.set_xscale('log')
    
    # =========================================================================
    # Panel G: Optimization Trajectories
    # =========================================================================
    ax7 = fig.add_subplot(gs[3, 0])
    
    epochs = exp6['epochs']
    
    ax7.plot(epochs, exp6['bptt_k'], 'o-', color=c_bptt, lw=LW, ms=4,
             label='BPTT', markeredgecolor='white', markeredgewidth=0.5)
    ax7.plot(epochs, exp6['girsanov_k'], 's--', color=c_gir, lw=LW, ms=4,
             label='Girsanov', markeredgecolor='white', markeredgewidth=0.5)
    ax7.axhline(exp6['target_k'], color=c_theory, ls=':', lw=2, label=f"Target k = {exp6['target_k']:.1f}")
    
    ax7.set_xlabel('Optimization Epoch')
    ax7.set_ylabel('Spring Constant k')
    ax7.set_title('G. Parameter Optimization\n(Both converge, Girsanov slower)', fontweight='bold')
    ax7.legend(loc='lower right', fontsize=9)
    ax7.set_ylim(0, 2.5)
    
    # =========================================================================
    # Panel H: Summary Box
    # =========================================================================
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.axis('off')
    
    summary_text = """
    Girsanov Path-Reweighting: Summary
    ══════════════════════════════════════════════════════════════

    MATHEMATICAL BASIS:
      For dx = F_θ(x)dt + σdW, the path probability gradient is:
      ∇_θ log p(τ|θ) = (1/σ²) ∫₀ᵀ ∇_θF · (dx - F·dt)
      
      This enables: ∇_θ ⟨O⟩ = ⟨O · ∇_θ log p(τ|θ)⟩

    ✓ WORKS WELL FOR:
      • Short trajectories (T < relaxation time)
      • Simple observables (final state, time averages)
      • Non-equilibrium problems (where REINFORCE fails)
      • When BPTT memory is prohibitive

    ✗ STRUGGLES WITH:
      • Long trajectories (variance grows ~ exp(T))
      • Path-dependent observables (FPT, extrema, rare events)
      • High-dimensional systems
      • When high accuracy is required

    PRACTICAL GUIDELINES:
      1. Monitor ESS: if ESS/N < 10%, results unreliable
      2. For equilibrium: prefer REINFORCE (bounded variance)
      3. For non-equilibrium: use short T or switch to BPTT
      4. Girsanov can optimize, but needs lower learning rates
    """
    
    ax8.text(0.02, 0.98, summary_text, transform=ax8.transAxes,
             fontsize=9, family='monospace', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F4F8', 
                      edgecolor='#D8DEE9', alpha=0.95))
    
    plt.suptitle('Girsanov Path-Reweighting Gradient Estimator: Comprehensive Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#FAFBFC')
    print(f"\n[+] Saved figure to {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Girsanov Estimator: Comprehensive Behavior Analysis")
    print("=" * 70)
    print("""
This demo explores the Girsanov path-reweighting gradient estimator:
  1. Variance growth with trajectory length
  2. Simple vs path-dependent observables  
  3. Log score distribution analysis
  4. Effective sample size decay
  5. Comparison with REINFORCE for equilibrium
  6. Optimization performance
""")
    
    t0 = time.time()
    
    # Run experiments
    exp1 = experiment_variance_vs_length()
    exp2 = experiment_observable_types()
    exp3 = experiment_score_distribution()
    exp4 = experiment_effective_samples()
    exp5 = experiment_girsanov_vs_reinforce()
    exp6 = experiment_optimization()
    
    print(f"\nTotal runtime: {time.time() - t0:.1f} seconds")
    
    # Create output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    assets_dir = os.path.join(project_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Plot
    save_path = os.path.join(assets_dir, "demo_girsanov.png")
    plot_results(exp1, exp2, exp3, exp4, exp5, exp6, save_path)
    
    print("\n" + "=" * 70)
    print("Key Takeaways")
    print("=" * 70)
    print("""
1. VARIANCE GROWTH: Girsanov variance grows with trajectory length T.
   For T > relaxation time, estimates become unreliable.

2. OBSERVABLE COMPLEXITY: Simple observables (final state) work better
   than path-dependent ones (FPT). Path-dependent observables can give
   completely wrong gradients (wrong sign and magnitude).

3. EFFECTIVE SAMPLES: ESS drops rapidly with T. When ESS/N < 10%,
   the estimate is dominated by a few extreme trajectories.

4. VS REINFORCE: For equilibrium observables, REINFORCE has bounded
   variance while Girsanov variance grows. Prefer REINFORCE when applicable.

5. OPTIMIZATION: Girsanov can be used for optimization, but requires
   lower learning rates due to gradient noise.

6. WHEN TO USE: Girsanov is most useful for non-equilibrium problems
   with short trajectories where BPTT memory is prohibitive.
""")
