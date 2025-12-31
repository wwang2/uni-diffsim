#!/usr/bin/env python3
"""
Comparison: Forward Sensitivity vs BPTT vs Adjoint

This script benchmarks three gradient computation methods:
1. BPTT (Backpropagation Through Time): Standard PyTorch backward().
2. Adjoint: Checkpointed Discrete Adjoint (memory efficient).
3. Forward Sensitivity: Forward Mode AD via Jacobian-Vector Products (O(P) memory).

Metrics:
- Execution Time (Forward + Backward) vs Trajectory Length
- Peak Memory Usage vs Trajectory Length
- Gradient Accuracy (relative to BPTT)

The benchmark uses a simple harmonic oscillator system to keep dynamics fast,
focusing on the overhead of the gradient methods themselves.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
import tracemalloc
import gc
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uni_diffsim import (
    OverdampedLangevin, Harmonic,
    CheckpointedNoseHoover, DiscreteAdjointNoseHoover,
    ForwardSensitivityEstimator
)
from uni_diffsim.integrators import NoseHoover
from uni_diffsim.plotting import apply_style, COLORS as BASE_COLORS

# Apply shared plotting style
apply_style()

# Extend color palette for this script
COLORS = {
    'BPTT': BASE_COLORS['bptt'],
    'Adjoint': BASE_COLORS['adjoint'],
    'Forward': BASE_COLORS['forward'],
    'Theory': BASE_COLORS['theory'],
}

def benchmark_methods(traj_lengths: List[int], n_trials: int = 5):
    """Run benchmark for time, memory, and accuracy."""
    
    kT = 1.0
    k_spring = 1.0
    gamma = 1.0
    dt = 0.01
    dim = 1
    # For forward mode, we typically have few parameters. 
    # Let's say we optimize k and maybe gamma.
    # To mimic a real scenario, we might have more parameters or larger state.
    # Forward mode scales with # params. BPTT/Adjoint scale with state size (mostly).
    # Let's stick to 1D harmonic but 2 parameters (k, kT) for now.
    
    # Storage for results
    results = {
        'lengths': traj_lengths,
        'BPTT': {'time': [], 'mem': [], 'grad': []},
        'Adjoint': {'time': [], 'mem': [], 'grad': []},
        'Forward': {'time': [], 'mem': [], 'grad': []}
    }
    
    # We will compute gradient of Loss = 0.5 * (x_final - target)^2
    target = 1.0
    
    print(f"{'Steps':<10} | {'Method':<10} | {'Time (s)':<10} | {'Mem (MB)':<10}")
    print("-" * 50)

    for n_steps in traj_lengths:
        for method in ['BPTT', 'Adjoint', 'Forward']:
            times = []
            mems = []
            grads = []
            
            # --- Warmup Run ---
            # Run once to initialize caches, functional transforms, or compilation
            # This ensures we measure "steady state" performance.
            torch.manual_seed(42)
            x0_w = torch.zeros(10, dim) # Smaller batch for warmup
            v0_w = torch.randn(10, dim)
            
            if method == 'BPTT':
                integrator = NoseHoover(kT=kT, mass=1.0, Q=1.0)
                force_fn = lambda x: -k_spring * x
                traj_x, _ = integrator.run(x0_w, v0_w, force_fn, dt, 10)
                loss = 0.5 * ((traj_x[-1] - target)**2).mean()
                loss.backward()
            elif method == 'Adjoint':
                integrator = CheckpointedNoseHoover(kT=kT, mass=1.0, Q=1.0)
                force_fn = lambda x: -k_spring * x
                traj_x, _ = integrator.run(x0_w, v0_w, force_fn, dt, 10, final_only=True)
                loss = 0.5 * ((traj_x[0] - target)**2).mean()
                loss.backward()
            elif method == 'Forward':
                integrator = NoseHoover(kT=kT, mass=1.0, Q=1.0)
                estimator = ForwardSensitivityEstimator(integrator, param_names=['kT'])
                force_fn = lambda x: -k_spring * x
                _ = estimator.forward_sensitivity(x0_w, v0_w, force_fn, dt, 10, final_only=True)
            
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            for trial in range(n_trials):
                # Setup
                torch.manual_seed(42 + trial)
                x0 = torch.zeros(100, dim) # Batch of 100 walkers
                v0 = torch.randn(100, dim)
                
                # Start tracking
                tracemalloc.start()
                start_time = time.perf_counter()
                
                # --- Method Execution ---
                if method == 'BPTT':
                    # Use NoseHoover for direct comparison with Adjoint
                    # BPTT stores full trajectory for backprop (O(T) memory)
                    integrator = NoseHoover(kT=kT, mass=1.0, Q=1.0)
                    force_fn = lambda x: -k_spring * x
                    # Note: BPTT needs full trajectory stored for backward pass
                    # even if we only care about final state loss
                    traj_x, traj_v = integrator.run(x0, v0, force_fn, dt, n_steps)
                    
                    loss = 0.5 * ((traj_x[-1] - target)**2).mean()
                    loss.backward()
                    
                    grad_val = integrator.kT.grad.item()
                    
                elif method == 'Adjoint':
                    # Checkpointed Adjoint with final_only=True for O(√T) memory
                    integrator = CheckpointedNoseHoover(kT=kT, mass=1.0, Q=1.0)
                    force_fn = lambda x: -k_spring * x
                    # Use final_only=True to only store checkpoints, not full trajectory
                    traj_x, traj_v = integrator.run(x0, v0, force_fn, dt, n_steps, final_only=True)
                    
                    # traj_x has shape (1, batch, dim) when final_only=True
                    loss = 0.5 * ((traj_x[0] - target)**2).mean()
                    loss.backward()
                    
                    grad_val = integrator.kT.grad.item()
                    
                elif method == 'Forward':
                    # Forward Sensitivity with final_only=True for O(1) memory
                    integrator = NoseHoover(kT=kT, mass=1.0, Q=1.0)
                    estimator = ForwardSensitivityEstimator(integrator, param_names=['kT'])
                    
                    force_fn = lambda x: -k_spring * x
                    # Use final_only=True for memory efficiency
                    (traj_x, traj_v), jac_output = estimator.forward_sensitivity(
                        x0, v0, force_fn, dt, n_steps, final_only=True
                    )
                    
                    # With final_only=True, traj_x has shape (1, batch, dim)
                    jacs_x = jac_output[0]
                    
                    dx_dkT = jacs_x['kT'][0]  # (batch, dim) when final_only=True
                    x_T = traj_x[0]           # (batch, dim)
                    
                    # Chain rule: dL/dkT = mean( (x_i - target) * dx_i/dkT )
                    dL_dkT = ((x_T - target) * dx_dkT).mean()
                    
                    grad_val = dL_dkT.item()

                end_time = time.perf_counter()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
                times.append(end_time - start_time)
                mems.append(peak / 1024 / 1024) # MB
                grads.append(grad_val)
            
            # Record stats
            results[method]['time'].append(np.mean(times))
            results[method]['mem'].append(np.mean(mems))
            results[method]['grad'].append(np.mean(grads))
            
            print(f"{n_steps:<10} | {method:<10} | {np.mean(times):<10.4f} | {np.mean(mems):<10.2f}")

    return results

def plot_results(results, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    lengths = results['lengths']
    
    # 1. Time Scaling
    ax = axes[0]
    for method in ['BPTT', 'Adjoint', 'Forward']:
        ax.plot(lengths, results[method]['time'], 'o-', label=method, color=COLORS[method])
    ax.set_xlabel('Trajectory Length (steps)')
    ax.set_ylabel('Time (s) [Steady State]')
    ax.set_title('Computational Time (Lower is Better)', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 2. Memory Scaling
    ax = axes[1]
    for method in ['BPTT', 'Adjoint', 'Forward']:
        ax.plot(lengths, results[method]['mem'], 'o-', label=method, color=COLORS[method])
    ax.set_xlabel('Trajectory Length (steps)')
    ax.set_ylabel('Peak Memory (MB) [Steady State]')
    ax.set_title('Memory Usage (Lower is Better)', fontweight='bold')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # 3. Gradient Consistency (Bar Plot)
    ax = axes[2]
    # We'll plot the gradients for the longest trajectory to compare values
    x = np.arange(len(lengths))
    width = 0.25
    
    for i, method in enumerate(['BPTT', 'Adjoint', 'Forward']):
        ax.bar(x + (i-1)*width, results[method]['grad'], width, label=method, color=COLORS[method], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lengths])
    ax.set_xlabel('Trajectory Length (steps)')
    ax.set_ylabel('Gradient Estimate (dL/dkT)')
    ax.set_title('Gradient Consistency (Higher matches BPTT)', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    lengths = [100, 200, 500, 1000] 
    results = benchmark_methods(lengths, n_trials=3)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    assets_dir = os.path.join(project_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    save_path = os.path.join(assets_dir, "benchmark_forward_vs_adjoint.png")
    plot_results(results, save_path)

