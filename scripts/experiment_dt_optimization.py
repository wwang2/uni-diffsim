
import torch
import torch.nn as nn
import torch.optim as optim
from uni_diffsim.integrators import NoseHoover
from uni_diffsim.gradient_estimators import CheckpointedNoseHoover
from uni_diffsim.potentials import Harmonic

def run_experiment():
    print("==================================================")
    print("Experiment: Optimizing Time Step (dt) via Gradient Descent")
    print("==================================================\n")

    device = torch.device("cpu")
    torch.manual_seed(42)

    # --- Experiment Setup ---
    # We want to find the 'dt' that evolves the system to a specific target state
    # after a fixed number of steps.

    n_steps = 50
    dt_true = 0.05
    x0 = torch.tensor([[1.0, 0.0]], device=device) # Start at (1,0)
    v0 = torch.tensor([[0.0, 1.0]], device=device) # Initial velocity

    # Potential: Harmonic Oscillator k=1.0
    potential = Harmonic(k=1.0)

    print(f"Goal: Recover dt_true = {dt_true:.4f} starting from a different value.")
    print(f"System: Harmonic Oscillator, Nose-Hoover Dynamics, {n_steps} steps.\n")

    # --- Generate Target ---
    print("Generating target state with true dt...")
    integrator_ref = NoseHoover(kT=1.0, mass=1.0, Q=1.0)
    with torch.no_grad():
        traj_target, _ = integrator_ref.run(x0, v0, potential.force, dt=dt_true, n_steps=n_steps)
        x_target = traj_target[-1]

    print(f"Target final position: {x_target.numpy()[0]}\n")

    # --- Method 1: Optimization using Standard BPTT ---
    print("--- Method 1: Standard BPTT (NoseHoover) ---")

    # Initialize learnable dt
    dt_learn = torch.tensor(0.02, device=device, requires_grad=True)
    optimizer = optim.Adam([dt_learn], lr=0.005)

    integrator_bptt = NoseHoover(kT=1.0, mass=1.0, Q=1.0)

    for epoch in range(1, 51):
        optimizer.zero_grad()

        # Forward pass (BPTT tracks gradients through the graph automatically)
        traj, _ = integrator_bptt.run(x0, v0, potential.force, dt=dt_learn, n_steps=n_steps)
        x_final = traj[-1]

        # Loss: Squared Euclidean distance to target
        loss = ((x_final - x_target) ** 2).sum()

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}: Loss = {loss.item():.6f}, dt = {dt_learn.item():.5f}, grad = {dt_learn.grad.item():.5f}")

    print(f"Result BPTT: dt converged to {dt_learn.item():.5f} (Target: {dt_true})\n")

    # --- Method 2: Optimization using Checkpointed Adjoint ---
    print("--- Method 2: Checkpointed Adjoint (CheckpointedNoseHoover) ---")
    print("Note: This uses O(sqrt(T)) memory instead of O(T)!\n")

    # Initialize learnable dt
    dt_learn_adj = torch.tensor(0.02, device=device, requires_grad=True)
    optimizer_adj = optim.Adam([dt_learn_adj], lr=0.005)

    # Checkpointed Integrator
    integrator_adj = CheckpointedNoseHoover(kT=1.0, mass=1.0, Q=1.0, checkpoint_segments=10)

    for epoch in range(1, 51):
        optimizer_adj.zero_grad()

        # Run forward (final_only=True returns just the last state for loss calc)
        # Note: In CheckpointedNoseHoover, if we want to backprop through the final state,
        # we typically use final_only=False or handle the sparse output.
        # But wait, CheckpointedNoseHoover.run returns tuple(trajs).
        # If final_only=False, it returns checkpoints.
        # The custom function backward handles gradients.

        # Let's use final_only=False to get full trajectory behavior (though checkpoints used internally)
        # Actually, let's use final_only=True to demonstrate memory efficiency if supported.
        # run() returns (traj_x, traj_v). If final_only=True, shapes are (1, ...)

        traj_x, _ = integrator_adj.run(x0, v0, potential.force, dt=dt_learn_adj, n_steps=n_steps, final_only=True)
        x_final = traj_x[-1] # Shape (1, 2) -> (2)

        loss = ((x_final - x_target) ** 2).sum()

        loss.backward()
        optimizer_adj.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}: Loss = {loss.item():.6f}, dt = {dt_learn_adj.item():.5f}, grad = {dt_learn_adj.grad.item():.5f}")

    print(f"Result Adjoint: dt converged to {dt_learn_adj.item():.5f} (Target: {dt_true})")
    print("==================================================")

if __name__ == "__main__":
    run_experiment()
