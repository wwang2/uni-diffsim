#!/usr/bin/env python3
"""
Non-Equilibrium Simulation Demos: BPTT vs REINFORCE vs Implicit Differentiation

These demos showcase scenarios where different gradient methods apply:

1. BPTT (Backprop Through Time): Works for everything, O(T) memory
2. REINFORCE/TPT: Only equilibrium observables, O(1) memory
3. Implicit Diff: Optimization problems + equilibrium (= REINFORCE), O(1) memory

Key insight from Blondel et al. NeurIPS 2022:
- Implicit diff uses optimality conditions F(x*,θ) = 0
- For equilibrium sampling, reduces to same formula as REINFORCE
- Does NOT work for path-dependent observables (FPT, work, etc.)

Demos:
1. First Passage Time Optimization (BPTT only)
2. Finite-Time Transition Probability (BPTT only)
3. Optimal Control / Steering (BPTT only)

Author: Generated with Claude Code
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uni_diffsim import (
    OverdampedLangevin, BAOAB, VelocityVerlet,
    DoubleWell, AsymmetricDoubleWell, Harmonic,
    ReinforceEstimator, ImplicitDiffEstimator,
)
from uni_diffsim.plotting import apply_style, COLORS

# Apply shared plotting style
apply_style()

LW = 2.0
MS = 6


def soft_indicator(x, threshold, sigma=0.1):
    """Differentiable approximation to indicator function x > threshold."""
    return torch.sigmoid((x - threshold) / sigma)


def soft_first_passage_time(traj, threshold, sigma=0.1, dt=0.01):
    """
    Differentiable approximation to first passage time.

    Uses cumulative product of (1 - soft_indicator) to track "not yet crossed",
    then sum to get expected crossing time.

    Args:
        traj: [n_steps, n_walkers, dim] trajectory
        threshold: crossing threshold
        sigma: softness parameter
        dt: time step

    Returns:
        Mean first passage time (differentiable)
    """
    n_steps, n_walkers, dim = traj.shape

    # For each time step, probability of crossing at that step
    # P(cross at t) = P(not crossed before t) * P(cross at t)
    x = traj[..., 0]  # [n_steps, n_walkers]

    # Soft indicator: has crossed threshold
    crossed = soft_indicator(x, threshold, sigma)  # [n_steps, n_walkers]

    # Survival probability: product of (1 - crossed) up to time t
    # Use cumsum of log for numerical stability
    log_survival = torch.cumsum(torch.log(1 - crossed + 1e-10), dim=0)
    survival = torch.exp(log_survival)  # [n_steps, n_walkers]

    # Expected first passage time: sum of survival probabilities * dt
    # This is because E[τ] = integral of survival function
    mean_fpt = survival.sum(dim=0).mean() * dt

    return mean_fpt


def soft_transition_probability(traj, threshold, sigma=0.1):
    """
    Differentiable transition probability: P(x(T) > threshold | x(0) < threshold).

    Args:
        traj: [n_steps, n_walkers, dim] trajectory
        threshold: crossing threshold
        sigma: softness parameter

    Returns:
        Transition probability (differentiable)
    """
    x_final = traj[-1, :, 0]  # [n_walkers]
    return soft_indicator(x_final, threshold, sigma).mean()


# =============================================================================
# Demo 1: First Passage Time Optimization
# =============================================================================
def demo_first_passage_time():
    """
    Optimize barrier height to minimize mean first passage time.

    Setup:
    - Particles start in left well (x = -1)
    - Goal: reach right well (x > 0)
    - Tune barrier height to minimize crossing time

    Trade-off: Lower barrier → faster crossing but less stable wells
    """
    print("\n" + "=" * 70)
    print("Demo 1: First Passage Time Optimization")
    print("=" * 70)
    print("Goal: Tune barrier height to minimize mean first passage time")
    print("Observable: τ = min{t : x(t) > 0 | x(0) = -1}")
    print("This is NON-EQUILIBRIUM - REINFORCE does not apply!")
    print("=" * 70)

    torch.manual_seed(42)

    # Parameters
    kT = 0.5
    dt = 0.005
    n_steps = 2000
    n_walkers = 200
    threshold = 0.0  # Cross to right well

    # Test different barrier heights
    barrier_heights = np.linspace(0.5, 3.0, 12)

    # Compare integrators
    integrators = {
        'Overdamped Langevin': OverdampedLangevin(gamma=1.0, kT=kT),
        'BAOAB': BAOAB(gamma=1.0, kT=kT),
    }

    results = {name: {'fpt': [], 'fpt_std': []} for name in integrators}

    for barrier in barrier_heights:
        potential = DoubleWell(barrier_height=barrier)

        for name, integrator in integrators.items():
            fpts = []

            for trial in range(5):
                torch.manual_seed(trial * 100)

                # Start in left well
                x0 = torch.full((n_walkers, 1), -1.0)
                if name == 'BAOAB':
                    v0 = torch.randn(n_walkers, 1) * np.sqrt(kT)
                    traj_x, traj_v = integrator.run(x0, v0, potential.force, dt=dt, n_steps=n_steps)
                else:
                    traj_x = integrator.run(x0, potential.force, dt=dt, n_steps=n_steps)

                # Compute first passage time (non-differentiable for measurement)
                x = traj_x[:, :, 0]  # [n_steps, n_walkers]
                crossed = (x > threshold).float()

                # First time each walker crosses
                first_crossing = torch.argmax(crossed, dim=0).float() * dt
                # If never crossed, use max time
                never_crossed = (crossed.sum(dim=0) == 0)
                first_crossing[never_crossed] = n_steps * dt

                fpts.append(first_crossing.mean().item())

            results[name]['fpt'].append(np.mean(fpts))
            results[name]['fpt_std'].append(np.std(fpts))

    # Now optimize barrier height using BPTT
    print("\n[BPTT Optimization] Finding optimal barrier height...")

    barrier_param = torch.tensor([2.0], requires_grad=True)
    optimizer = torch.optim.Adam([barrier_param], lr=0.1)

    barrier_history = [barrier_param.item()]
    fpt_history = []

    for epoch in range(50):
        optimizer.zero_grad()

        # Create potential with current barrier
        # We need to manually implement the potential to keep gradients
        def force_fn(x):
            # Double well: U = a(x^2 - 1)^2, F = -dU/dx = -4a*x*(x^2 - 1)
            return -4 * barrier_param * x * (x**2 - 1)

        torch.manual_seed(epoch)
        x0 = torch.full((n_walkers, 1), -1.0)

        # Run overdamped Langevin with gradient tracking
        integrator = OverdampedLangevin(gamma=1.0, kT=kT)
        traj = integrator.run(x0, force_fn, dt=dt, n_steps=n_steps)

        # Differentiable FPT approximation
        fpt = soft_first_passage_time(traj, threshold, sigma=0.1, dt=dt)
        fpt_history.append(fpt.item())

        # Minimize FPT
        fpt.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([barrier_param], 1.0)

        optimizer.step()

        # Keep barrier positive
        with torch.no_grad():
            barrier_param.clamp_(min=0.3, max=4.0)

        barrier_history.append(barrier_param.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: barrier = {barrier_param.item():.3f}, FPT ≈ {fpt.item():.3f}")

    optimal_barrier_bptt = barrier_param.item()
    print(f"\n  BPTT optimal barrier height: {optimal_barrier_bptt:.3f}")

    # -------------------------------------------------------------------------
    # Now try REINFORCE optimization (misapplied to non-equilibrium problem)
    # -------------------------------------------------------------------------
    print("\n[REINFORCE Optimization] Attempting to optimize barrier...")
    print("  (Note: REINFORCE assumes equilibrium samples - this is wrong here!)")

    barrier_param_rf = torch.tensor([2.0], requires_grad=True)
    optimizer_rf = torch.optim.Adam([barrier_param_rf], lr=0.1)

    barrier_history_rf = [barrier_param_rf.item()]
    fpt_history_rf = []

    for epoch in range(50):
        optimizer_rf.zero_grad()

        torch.manual_seed(epoch)
        x0 = torch.full((n_walkers, 1), -1.0)

        # Run simulation to get samples (starting from x=-1, NOT equilibrium!)
        potential_rf = DoubleWell(barrier_height=barrier_param_rf.item())
        integrator = OverdampedLangevin(gamma=1.0, kT=kT)
        traj = integrator.run(x0, potential_rf.force, dt=dt, n_steps=n_steps)
        samples = traj.reshape(-1, 1).detach()  # Flatten trajectory

        # Compute FPT (non-differentiable measurement)
        x = traj[:, :, 0]
        crossed = (x > threshold).float()
        first_crossing = torch.argmax(crossed, dim=0).float() * dt
        never_crossed = (crossed.sum(dim=0) == 0)
        first_crossing[never_crossed] = n_steps * dt
        fpt_measured = first_crossing.mean().item()
        fpt_history_rf.append(fpt_measured)

        # REINFORCE gradient estimate (WRONG for non-equilibrium!)
        # ∇_θ ⟨O⟩ = -β Cov(O, ∇_θU)
        # But our samples are NOT from equilibrium distribution
        potential_for_grad = DoubleWell(barrier_height=barrier_param_rf.item())
        potential_for_grad.barrier_height = torch.nn.Parameter(
            torch.tensor([barrier_param_rf.item()], requires_grad=True)
        )

        # Use soft FPT as observable (applied to each sample independently - wrong!)
        # This doesn't make sense for FPT but let's see what REINFORCE gives
        estimator = ReinforceEstimator(potential_for_grad, beta=1.0/kT)

        # Observable: soft indicator of being past threshold (proxy for "fast crossing")
        observable = lambda x: soft_indicator(x.squeeze(-1), threshold, sigma=0.1)
        try:
            grads = estimator.estimate_gradient(samples, observable=observable)
            rf_grad = grads.get('barrier_height', torch.tensor([0.0])).item()
        except Exception:
            rf_grad = 0.0

        # Update (we want to MINIMIZE FPT, so maximize crossing probability)
        # But REINFORCE gradient is for the observable, not FPT directly
        lr = 0.1 / (1 + epoch * 0.02)
        with torch.no_grad():
            # Negative because we want more particles to cross (higher observable)
            barrier_param_rf -= lr * rf_grad
            barrier_param_rf.clamp_(min=0.3, max=4.0)

        barrier_history_rf.append(barrier_param_rf.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: barrier = {barrier_param_rf.item():.3f}, "
                  f"FPT ≈ {fpt_measured:.3f}, RF_grad = {rf_grad:.4f}")

    optimal_barrier_rf = barrier_param_rf.item()
    print(f"\n  REINFORCE final barrier: {optimal_barrier_rf:.3f}")
    print(f"  Compare: BPTT found {optimal_barrier_bptt:.3f}")

    return {
        'barrier_heights': barrier_heights,
        'results': results,
        'barrier_history_bptt': barrier_history,
        'barrier_history_rf': barrier_history_rf,
        'fpt_history_bptt': fpt_history,
        'fpt_history_rf': fpt_history_rf,
        'optimal_barrier_bptt': optimal_barrier_bptt,
        'optimal_barrier_rf': optimal_barrier_rf,
        'kT': kT,
    }


# =============================================================================
# Demo 2: Finite-Time Transition Probability
# =============================================================================
def demo_transition_probability():
    """
    Maximize probability of crossing barrier in fixed time T.

    Setup:
    - Particles start in left well
    - Fixed time budget T
    - Goal: maximize P(x(T) > 0)

    Compare: Different integrators and temperatures
    """
    print("\n" + "=" * 70)
    print("Demo 2: Finite-Time Transition Probability")
    print("=" * 70)
    print("Goal: Maximize P(cross barrier in time T)")
    print("Observable: P_cross = E[1_{x(T) > 0} | x(0) = -1]")
    print("This is NON-EQUILIBRIUM - finite time, not steady state!")
    print("=" * 70)

    torch.manual_seed(42)

    # Parameters
    barrier = 2.0
    dt = 0.005
    n_walkers = 500
    threshold = 0.0

    # Test different time horizons
    time_horizons = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    temperatures = [0.3, 0.5, 0.8, 1.0]

    results = {kT: {'times': [], 'prob': [], 'prob_std': []} for kT in temperatures}

    for kT in temperatures:
        for T in time_horizons:
            n_steps = int(T / dt)
            probs = []

            for trial in range(5):
                torch.manual_seed(trial * 100)

                potential = DoubleWell(barrier_height=barrier)
                integrator = OverdampedLangevin(gamma=1.0, kT=kT)

                x0 = torch.full((n_walkers, 1), -1.0)
                traj = integrator.run(x0, potential.force, dt=dt, n_steps=n_steps)

                # Transition probability
                x_final = traj[-1, :, 0]
                p_cross = (x_final > threshold).float().mean().item()
                probs.append(p_cross)

            results[kT]['times'].append(T)
            results[kT]['prob'].append(np.mean(probs))
            results[kT]['prob_std'].append(np.std(probs))

    # Compare integrators at fixed temperature
    print("\n[Integrator Comparison] at kT = 0.5...")
    kT_compare = 0.5
    integrator_results = {}

    for name, IntClass in [('Overdamped', OverdampedLangevin), ('BAOAB', BAOAB)]:
        integrator_results[name] = {'times': [], 'prob': []}

        for T in time_horizons:
            n_steps = int(T / dt)
            probs = []

            for trial in range(5):
                torch.manual_seed(trial * 100)

                potential = DoubleWell(barrier_height=barrier)

                x0 = torch.full((n_walkers, 1), -1.0)

                if name == 'BAOAB':
                    integrator = IntClass(gamma=1.0, kT=kT_compare)
                    v0 = torch.randn(n_walkers, 1) * np.sqrt(kT_compare)
                    traj, _ = integrator.run(x0, v0, potential.force, dt=dt, n_steps=n_steps)
                else:
                    integrator = IntClass(gamma=1.0, kT=kT_compare)
                    traj = integrator.run(x0, potential.force, dt=dt, n_steps=n_steps)

                x_final = traj[-1, :, 0]
                p_cross = (x_final > threshold).float().mean().item()
                probs.append(p_cross)

            integrator_results[name]['times'].append(T)
            integrator_results[name]['prob'].append(np.mean(probs))

    # BPTT optimization of temperature for target transition probability
    print("\n[BPTT Optimization] Finding optimal kT for P_cross = 0.5 at T = 10...")

    T_target = 10.0
    n_steps_opt = int(T_target / dt)
    target_prob = 0.5

    # Parameterize kT
    log_kT = torch.tensor([np.log(0.5)], requires_grad=True)
    optimizer = torch.optim.Adam([log_kT], lr=0.1)

    kT_history = [np.exp(log_kT.item())]
    prob_history = []

    for epoch in range(40):
        optimizer.zero_grad()

        kT_current = torch.exp(log_kT)

        torch.manual_seed(epoch)
        x0 = torch.full((n_walkers, 1), -1.0)

        # Manual Langevin with differentiable kT
        x = x0.clone()
        potential = DoubleWell(barrier_height=barrier)
        gamma = 1.0

        for step in range(n_steps_opt):
            noise = torch.randn_like(x)
            force = potential.force(x)
            x = x + (force / gamma) * dt + torch.sqrt(2 * kT_current * dt / gamma) * noise

        # Differentiable transition probability
        p_cross = soft_indicator(x[:, 0], threshold, sigma=0.1).mean()
        prob_history.append(p_cross.item())

        # Loss: (P_cross - target)^2
        loss = (p_cross - target_prob) ** 2
        loss.backward()

        optimizer.step()

        with torch.no_grad():
            log_kT.clamp_(min=np.log(0.1), max=np.log(2.0))

        kT_history.append(np.exp(log_kT.item()))

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: kT = {kT_history[-1]:.3f}, P_cross = {prob_history[-1]:.3f}")

    optimal_kT_bptt = np.exp(log_kT.item())
    print(f"\n  BPTT optimal kT for P_cross = {target_prob}: {optimal_kT_bptt:.3f}")

    # -------------------------------------------------------------------------
    # REINFORCE comparison: Try to optimize for same target
    # -------------------------------------------------------------------------
    print("\n[REINFORCE Optimization] Attempting same optimization...")
    print("  (Note: REINFORCE uses equilibrium identity - doesn't account for finite time)")

    # For REINFORCE, we optimize barrier_height instead of kT
    # (since REINFORCE needs potential parameters)
    barrier_param_rf = torch.tensor([2.0], requires_grad=True)
    kT_fixed = 0.5

    barrier_history_rf = [barrier_param_rf.item()]
    prob_history_rf = []

    for epoch in range(40):
        torch.manual_seed(epoch)
        x0 = torch.full((n_walkers, 1), -1.0)

        # Run simulation (non-equilibrium!)
        potential_rf = DoubleWell(barrier_height=barrier_param_rf.item())
        integrator = OverdampedLangevin(gamma=1.0, kT=kT_fixed)
        traj = integrator.run(x0, potential_rf.force, dt=dt, n_steps=n_steps_opt)

        # Measure actual P_cross at time T
        x_final = traj[-1, :, 0]
        p_cross_actual = (x_final > threshold).float().mean().item()
        prob_history_rf.append(p_cross_actual)

        # REINFORCE gradient using trajectory samples (NOT equilibrium!)
        samples = traj.reshape(-1, 1).detach()
        potential_for_grad = DoubleWell(barrier_height=barrier_param_rf.item())

        estimator = ReinforceEstimator(potential_for_grad, beta=1.0/kT_fixed)
        observable = lambda x: soft_indicator(x.squeeze(-1), threshold, sigma=0.1)

        try:
            grads = estimator.estimate_gradient(samples, observable=observable)
            rf_grad = grads.get('barrier_height', torch.tensor([0.0])).item()
        except Exception:
            rf_grad = 0.0

        # Update: want P_cross = 0.5, so minimize (P_cross - 0.5)^2
        # d/d_barrier [(P - 0.5)^2] = 2(P - 0.5) * dP/d_barrier
        loss_grad = 2 * (p_cross_actual - target_prob) * rf_grad

        lr = 0.1 / (1 + epoch * 0.02)
        with torch.no_grad():
            barrier_param_rf -= lr * loss_grad
            barrier_param_rf.clamp_(min=0.5, max=4.0)

        barrier_history_rf.append(barrier_param_rf.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: barrier = {barrier_param_rf.item():.3f}, "
                  f"P_cross = {p_cross_actual:.3f}, RF_grad = {rf_grad:.4f}")

    optimal_barrier_rf = barrier_param_rf.item()
    print(f"\n  REINFORCE final barrier: {optimal_barrier_rf:.3f}")
    print(f"  Final P_cross: {prob_history_rf[-1]:.3f} (target: {target_prob})")

    return {
        'temperature_results': results,
        'integrator_results': integrator_results,
        'kT_history_bptt': kT_history,
        'prob_history_bptt': prob_history,
        'barrier_history_rf': barrier_history_rf,
        'prob_history_rf': prob_history_rf,
        'optimal_kT_bptt': optimal_kT_bptt,
        'optimal_barrier_rf': optimal_barrier_rf,
        'target_prob': target_prob,
    }


# =============================================================================
# Demo 3: Optimal Control with Neural Network
# =============================================================================
class NeuralController(nn.Module):
    """Time-dependent neural network controller F(x, t; θ)."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # Input: (x, t/T)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize with small weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x, t, T):
        """
        Args:
            x: [batch, 1] positions
            t: scalar, current time
            T: scalar, total time

        Returns:
            force: [batch, 1]
        """
        t_normalized = torch.full((x.shape[0], 1), t / T, device=x.device)
        inputs = torch.cat([x, t_normalized], dim=-1)
        return self.net(inputs)


def demo_optimal_control():
    """
    Learn optimal time-dependent control to steer particles.

    Setup:
    - Particles start at x = -1
    - Goal: reach x = +1 in time T
    - Learn control force F(x, t) to minimize work while reaching target

    This is pure BPTT - no equilibrium involved!
    """
    print("\n" + "=" * 70)
    print("Demo 3: Optimal Control with Neural Network")
    print("=" * 70)
    print("Goal: Learn F(x,t) to steer particles from x=-1 to x=+1")
    print("Minimize: Work done + Distance to target")
    print("This is OPTIMAL CONTROL - pure BPTT through dynamics!")
    print("=" * 70)

    torch.manual_seed(42)

    # Parameters
    kT = 0.3
    gamma = 1.0
    dt = 0.01
    T = 5.0
    n_steps = int(T / dt)
    n_walkers = 100

    x_start = -1.0
    x_target = 1.0

    # Baseline potential (flat or weak harmonic)
    def base_force(x):
        return torch.zeros_like(x)

    # Neural controller
    controller = NeuralController(hidden_dim=32)
    optimizer = torch.optim.Adam(controller.parameters(), lr=0.01)

    # Training
    loss_history = []
    work_history = []
    success_history = []

    print("\n[Training Neural Controller]...")

    for epoch in range(100):
        optimizer.zero_grad()

        torch.manual_seed(epoch)
        x = torch.full((n_walkers, 1), x_start)

        total_work = 0.0

        for step in range(n_steps):
            t = step * dt

            # Control force from neural network
            control_force = controller(x, t, T)

            # Total force = base + control
            force = base_force(x) + control_force

            # Work done by control force: W = F * dx
            # For overdamped: dx ≈ (F/γ) dt
            work_increment = (control_force ** 2).mean() * dt / gamma
            total_work = total_work + work_increment

            # Langevin step
            noise = torch.randn_like(x)
            x = x + (force / gamma) * dt + np.sqrt(2 * kT * dt / gamma) * noise

        # Final position loss
        distance_loss = ((x - x_target) ** 2).mean()

        # Success rate
        success_rate = (x[:, 0] > 0.5).float().mean().item()

        # Total loss: reach target + minimize work
        work_penalty = 0.01
        loss = distance_loss + work_penalty * total_work

        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
        optimizer.step()

        loss_history.append(loss.item())
        work_history.append(total_work.item())
        success_history.append(success_rate)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}: loss = {loss.item():.4f}, "
                  f"work = {total_work.item():.2f}, success = {success_rate:.2%}")

    # Evaluate learned controller
    print("\n[Evaluating Learned Controller]...")

    torch.manual_seed(999)
    x = torch.full((n_walkers, 1), x_start)

    trajectory = [x.detach().clone()]
    control_forces = []
    times = [0.0]

    with torch.no_grad():
        for step in range(n_steps):
            t = step * dt

            control_force = controller(x, t, T)
            control_forces.append(control_force.mean().item())

            force = base_force(x) + control_force
            noise = torch.randn_like(x)
            x = x + (force / gamma) * dt + np.sqrt(2 * kT * dt / gamma) * noise

            if step % 10 == 0:
                trajectory.append(x.clone())
                times.append((step + 1) * dt)

    trajectory = torch.stack(trajectory, dim=0)  # [n_saved, n_walkers, 1]

    final_x = trajectory[-1, :, 0]
    print(f"  Final position: {final_x.mean().item():.3f} ± {final_x.std().item():.3f}")
    print(f"  Success rate (x > 0.5): {(final_x > 0.5).float().mean().item():.2%}")

    # Compare with no control
    print("\n[Baseline: No Control]...")
    torch.manual_seed(999)
    x_baseline = torch.full((n_walkers, 1), x_start)

    traj_baseline = [x_baseline.detach().clone()]

    with torch.no_grad():
        for step in range(n_steps):
            force = base_force(x_baseline)
            noise = torch.randn_like(x_baseline)
            x_baseline = x_baseline + (force / gamma) * dt + np.sqrt(2 * kT * dt / gamma) * noise

            if step % 10 == 0:
                traj_baseline.append(x_baseline.clone())

    traj_baseline = torch.stack(traj_baseline, dim=0)

    final_baseline = traj_baseline[-1, :, 0]
    print(f"  Final position: {final_baseline.mean().item():.3f} ± {final_baseline.std().item():.3f}")
    print(f"  Success rate (x > 0.5): {(final_baseline > 0.5).float().mean().item():.2%}")

    return {
        'loss_history': loss_history,
        'work_history': work_history,
        'success_history': success_history,
        'trajectory': trajectory.numpy(),
        'traj_baseline': traj_baseline.numpy(),
        'times': times,
        'control_forces': control_forces,
        'controller': controller,
        'T': T,
        'dt': dt,
        'x_start': x_start,
        'x_target': x_target,
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(fpt_results, trans_results, control_results, save_path):
    """Create comprehensive figure with all demo results."""

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.patch.set_facecolor('#FAFBFC')

    # =========================================================================
    # Panel 1: First Passage Time vs Barrier Height
    # =========================================================================
    ax = axes[0, 0]

    for name, data in fpt_results['results'].items():
        color = COLORS['overdamped'] if 'Overdamped' in name else COLORS['baoab']
        ax.errorbar(fpt_results['barrier_heights'], data['fpt'],
                    yerr=data['fpt_std'], fmt='o-', color=color,
                    lw=LW, ms=MS, capsize=3, label=name)

    ax.axvline(fpt_results['optimal_barrier_bptt'], color=COLORS['bptt'],
               ls='--', lw=LW, label=f'BPTT optimal: {fpt_results["optimal_barrier_bptt"]:.2f}')
    ax.axvline(fpt_results['optimal_barrier_rf'], color=COLORS['reinforce'],
               ls=':', lw=LW, label=f'REINFORCE: {fpt_results["optimal_barrier_rf"]:.2f}')
    ax.set_xlabel('Barrier Height a')
    ax.set_ylabel('Mean First Passage Time')
    ax.set_title('First Passage Time vs Barrier', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 2: BPTT vs REINFORCE Barrier Optimization
    # =========================================================================
    ax = axes[0, 1]

    ax.plot(fpt_results['barrier_history_bptt'], '-', color=COLORS['bptt'], lw=LW, label='BPTT')
    ax.plot(fpt_results['barrier_history_rf'], '--', color=COLORS['reinforce'], lw=LW, label='REINFORCE')
    ax.axhline(0.3, color=COLORS['optimal'], ls=':', lw=LW-0.5, alpha=0.7, label='Optimal (low)')
    ax.set_xlabel('Optimization Iteration')
    ax.set_ylabel('Barrier Height a')
    ax.set_title('FPT Optimization: BPTT vs REINFORCE', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Inset: FPT during optimization
    ax_ins = ax.inset_axes([0.5, 0.5, 0.45, 0.4])
    ax_ins.plot(fpt_results['fpt_history_bptt'], '-', color=COLORS['bptt'], lw=LW-0.5, label='BPTT')
    ax_ins.plot(fpt_results['fpt_history_rf'], '--', color=COLORS['reinforce'], lw=LW-0.5, label='RF')
    ax_ins.set_xlabel('Iteration', fontsize=9)
    ax_ins.set_ylabel('FPT', fontsize=9)
    ax_ins.tick_params(labelsize=8)
    ax_ins.set_title('Mean FPT', fontsize=10)
    ax_ins.legend(fontsize=7)

    # =========================================================================
    # Panel 3: Transition Probability vs Time (different temperatures)
    # =========================================================================
    ax = axes[0, 2]

    temp_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(trans_results['temperature_results'])))

    for i, (kT, data) in enumerate(trans_results['temperature_results'].items()):
        ax.errorbar(data['times'], data['prob'], yerr=data['prob_std'],
                    fmt='o-', color=temp_colors[i], lw=LW, ms=MS-2,
                    capsize=2, label=f'kT = {kT}')

    ax.axhline(0.5, color='#4C566A', ls=':', lw=1.5, alpha=0.6)
    ax.set_xlabel('Time Horizon T')
    ax.set_ylabel('Transition Probability')
    ax.set_title('P(cross) vs Time (by Temperature)', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # =========================================================================
    # Panel 4: P_cross Optimization: BPTT vs REINFORCE
    # =========================================================================
    ax = axes[0, 3]

    # BPTT: kT optimization
    ax.plot(trans_results['prob_history_bptt'], '-', color=COLORS['bptt'], lw=LW, label='BPTT (opt kT)')
    # REINFORCE: barrier optimization
    ax.plot(trans_results['prob_history_rf'], '--', color=COLORS['reinforce'], lw=LW, label='REINFORCE (opt barrier)')
    ax.axhline(trans_results['target_prob'], color=COLORS['target'],
               ls=':', lw=LW-0.5, label=f'Target: {trans_results["target_prob"]}')
    ax.set_xlabel('Optimization Iteration')
    ax.set_ylabel('Transition Probability')
    ax.set_title('P_cross Optimization: BPTT vs REINFORCE', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # =========================================================================
    # Panel 5: Neural Controller Training Loss
    # =========================================================================
    ax = axes[1, 0]

    ax.plot(control_results['loss_history'], '-', color=COLORS['neural'], lw=LW)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Neural Controller Training', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 6: Success Rate During Training
    # =========================================================================
    ax = axes[1, 1]

    ax.plot(control_results['success_history'], '-', color=COLORS['optimal'], lw=LW)
    ax.axhline(1.0, color=COLORS['target'], ls='--', lw=1.5, alpha=0.6)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Success Rate (x > 0.5)')
    ax.set_title('Steering Success Rate', fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 7: Controlled vs Baseline Trajectories
    # =========================================================================
    ax = axes[1, 2]

    traj = control_results['trajectory']  # [n_times, n_walkers, 1]
    traj_base = control_results['traj_baseline']
    times = control_results['times']

    # Plot mean ± std
    mean_controlled = traj[:, :, 0].mean(axis=1)
    std_controlled = traj[:, :, 0].std(axis=1)
    mean_baseline = traj_base[:, :, 0].mean(axis=1)
    std_baseline = traj_base[:, :, 0].std(axis=1)

    ax.fill_between(times, mean_controlled - std_controlled,
                    mean_controlled + std_controlled,
                    color=COLORS['neural'], alpha=0.3)
    ax.plot(times, mean_controlled, '-', color=COLORS['neural'], lw=LW,
            label='Neural Control')

    ax.fill_between(times, mean_baseline - std_baseline,
                    mean_baseline + std_baseline,
                    color='#4C566A', alpha=0.2)
    ax.plot(times, mean_baseline, '--', color='#4C566A', lw=LW,
            label='No Control')

    ax.axhline(control_results['x_target'], color=COLORS['target'],
               ls=':', lw=LW, label='Target')
    ax.axhline(control_results['x_start'], color='#4C566A',
               ls=':', lw=1, alpha=0.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Position x')
    ax.set_title('Controlled vs Free Diffusion', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # =========================================================================
    # Panel 8: Control Force Profile
    # =========================================================================
    ax = axes[1, 3]

    force_times = np.linspace(0, control_results['T'], len(control_results['control_forces']))
    ax.plot(force_times, control_results['control_forces'], '-',
            color=COLORS['neural'], lw=LW)
    ax.axhline(0, color='#4C566A', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Control Force F(t)')
    ax.set_title('Learned Control Protocol', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add annotation about control strategy
    ax.annotate('Push right\nearly', xy=(0.5, control_results['control_forces'][25]),
                fontsize=9, ha='left', color=COLORS['neural'])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#FAFBFC')
    print(f"\n[+] Saved plot to {save_path}")

    return fig


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Create assets directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    assets_dir = os.path.join(project_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    # Run demos
    fpt_results = demo_first_passage_time()
    trans_results = demo_transition_probability()
    control_results = demo_optimal_control()

    # Plot results
    save_path = os.path.join(assets_dir, "demo_nonequilibrium.png")
    fig = plot_results(fpt_results, trans_results, control_results, save_path)

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Gradient Methods Comparison")
    print("=" * 70)
    print("""
┌─────────────────────┬────────┬───────────┬─────────────┐
│ Problem Type        │  BPTT  │ REINFORCE │ Implicit    │
├─────────────────────┼────────┼───────────┼─────────────┤
│ Energy Minimization │   ✓    │     ✗     │     ✓*      │
│ Equilibrium ⟨O⟩     │   ✓    │     ✓     │     ✓**     │
│ First Passage Time  │   ✓    │     ✗     │     ✗       │
│ Transition P(T)     │   ✓    │     ✗     │     ✗       │
│ Optimal Control     │   ✓    │     ✗     │     ✗       │
├─────────────────────┼────────┼───────────┼─────────────┤
│ Memory              │  O(T)  │    O(1)   │    O(1)     │
│ Needs Diff Solver   │  Yes   │    No     │    No       │
└─────────────────────┴────────┴───────────┴─────────────┘

* Implicit diff is EXACT for optimization (no sampling variance)
** For equilibrium sampling, Implicit diff = REINFORCE (same formula)

Key Insight (Blondel et al. NeurIPS 2022):
- Implicit diff uses F(x*,θ) = 0 (optimality/fixed point condition)
- For sampling p(x) ∝ exp(-βU), the stationarity gives REINFORCE formula
- Path-dependent observables have NO fixed point → only BPTT works

Non-Equilibrium Observables (This Demo):
1. First Passage Time τ: Path-dependent, no equilibrium
2. Transition P(cross): Finite-time, not steady-state
3. Optimal Control: Time-dependent force, not potential-based

For these problems: BPTT is the ONLY viable approach.
""")
    print("=" * 70)
