"""Ghost Gradient Estimator (Least Squares Shadowing).

This module implements the "Gradient Ghost" estimator, which computes gradients
for chaotic systems using Least Squares Shadowing (LSS).

Chaotic systems (like Lorenz 63) have Lyapunov exponents > 0, causing infinitesimal
perturbations to grow exponentially. Standard Backpropagation Through Time (BPTT)
computes the gradient along the exact forward trajectory, leading to exploding
gradients for long time horizons (the "butterfly effect").

The Shadowing Lemma states that for a "pseudo-trajectory" (e.g. one with parameter
shift), there exists a true "shadow" trajectory nearby that stays close for a long time.
The LSS method computes the sensitivity of this *shadow* trajectory, which is
bounded and meaningful, unlike the sensitivity of the forward trajectory.

Reference:
    Wang, Q., Hu, R., & Blonigan, P. (2014). Least squares shadowing sensitivity analysis
    of chaotic limit cycle oscillations. Journal of Computational Physics, 267, 210-224.
"""

import torch
import torch.nn as nn
from torch.func import jacrev, functional_call, vmap
from typing import Callable, Tuple, Dict, List, Optional
import math


class GhostGradientEstimator(nn.Module):
    """Computes stable gradients for chaotic systems using Least Squares Shadowing.

    Solves the constrained optimization problem:
        min sum_t ||v_t||^2
        s.t. v_{t+1} = M_t v_t + P_t

    where:
        v_t = d(x_t)/d(theta)  (the "shadowing sensitivity")
        M_t = d(f)/d(x) evaluated at x_t
        P_t = d(f)/d(theta) evaluated at x_t

    This finds the sensitivity vector v_t that satisfies the linearized dynamics
    but has minimum norm (i.e., does not explode along the unstable manifold).

    The gradient of an objective J = <O> is then:
        dJ/dtheta = sum_t (dO/dx_t) * v_t

    Args:
        integrator: The integrator module (nn.Module).
        param_names: List of parameter names to differentiate w.r.t.
        time_window: Window size for LSS solver (to fit in memory).
                     If None, solves for the full trajectory.
    """

    def __init__(self, integrator: nn.Module, param_names: List[str], time_window: Optional[int] = None):
        super().__init__()
        self.integrator = integrator
        self.param_names = param_names
        self.time_window = time_window

    def _extract_linearized_dynamics(self, x_traj: torch.Tensor,
                                   dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Jacobians M_t and P_t for the trajectory.

        Args:
            x_traj: Trajectory (n_steps, dim)
            dt: Time step

        Returns:
            M: (n_steps-1, dim, dim)  Jacobian w.r.t. state (x_{t+1} / x_t)
            P: (n_steps-1, dim, n_params) Jacobian w.r.t. params
        """
        n_steps, dim = x_traj.shape
        params = dict(self.integrator.named_parameters())
        target_params = {k: params[k] for k in self.param_names}
        other_params = {k: v for k, v in params.items() if k not in target_params}

        # Flatten target params to a single vector for Jacobian computation
        flat_params_vec = torch.cat([p.view(-1) for p in target_params.values()])

        def step_wrapper(x_t, p_vec):
            # Reconstruct params dict
            curr = 0
            p_dict = {}
            for k, p in target_params.items():
                numel = p.numel()
                p_dict[k] = p_vec[curr:curr+numel].view(p.shape)
                curr += numel

            full_params = {**other_params, **p_dict}

            # Call integrator step
            # Note: Lorenz63.step takes (x, dt)
            # We assume integrator.step signature is compatible
            # For Lorenz63: step(x, dt)
            if hasattr(self.integrator, 'step'):
                # Try to bind the step function directly if possible to avoid dispatch overhead?
                # No, just call the bound method on the module instance (functional_call handles params)
                return self.integrator.step(x_t, dt)
            else:
                 raise NotImplementedError("Integrator must have a step method")

        # Compute Jacobians using vmap over time
        # M_t = d(x_{t+1})/d(x_t)
        # P_t = d(x_{t+1})/d(theta)

        # Inputs: x_traj[:-1], flat_params_vec (broadcasted)
        xs = x_traj[:-1]
        p_expanded = flat_params_vec.unsqueeze(0).expand(n_steps-1, -1)

        # Use jacrev to get both Jacobians
        # jacrev(func, argnums=(0, 1)) returns (jac_x, jac_p)

        def compute_jacs(x, p):
            return jacrev(step_wrapper, argnums=(0, 1))(x, p)

        M, P = vmap(compute_jacs)(xs, p_expanded)

        return M, P

    def _solve_lss_system(self, M: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Solve the Least Squares Shadowing linear system.

        Min sum ||v_t||^2 s.t. v_{t+1} - M_t v_t = P_t

        Using KKT system method (Lagrange multipliers).
        Constraints: C v = b
        Minimize: 1/2 v^T v
        Lagrangian: L = 1/2 v^T v + lambda^T (C v - b)

        Equations:
        1. v + C^T lambda = 0  => v = -C^T lambda
        2. C v = b             => -C C^T lambda = b => (C C^T) lambda = -b

        This requires solving for lambda, then recovering v.
        However, C C^T is (T-1)*dim x (T-1)*dim.

        Structure of constraints:
        v_1 - M_0 v_0 = P_0
        v_2 - M_1 v_1 = P_1
        ...

        Variables: v = [v_0, v_1, ..., v_T] (total (T+1)*dim)
        Constraints: T blocks of dim.

        Let's perform a direct dense solve for short trajectories (T < 2000).
        For T*dim = 3000, dense solve is fast.
        """
        T_steps = M.shape[0] # T-1 transitions
        dim = M.shape[1]
        n_params = P.shape[2]

        # Total variables v: (T_steps + 1) * dim
        # Total constraints: T_steps * dim

        n_vars = (T_steps + 1) * dim
        n_con = T_steps * dim

        # Build Constraint Matrix C (sparse structure, but we build dense for simplicity)
        # Constraint i (at time t=i): v_{i+1} - M_i v_i = P_i
        # Row block i corresponds to constraints at time i.
        # Col block i corresponds to variable v_i.

        # C is (n_con, n_vars)
        C = torch.zeros(n_con, n_vars, device=M.device, dtype=M.dtype)
        b = torch.zeros(n_con, n_params, device=M.device, dtype=M.dtype)

        for t in range(T_steps):
            row_start = t * dim
            row_end = (t + 1) * dim

            # Coeff for v_{t+1} is I
            col_v_next = (t + 1) * dim
            C[row_start:row_end, col_v_next:col_v_next+dim] = torch.eye(dim, device=M.device)

            # Coeff for v_t is -M_t
            col_v_curr = t * dim
            C[row_start:row_end, col_v_curr:col_v_curr+dim] = -M[t]

            # RHS is P_t (for each parameter column)
            b[row_start:row_end, :] = P[t]

        # Solve (C C^T) lambda = -b
        # C is usually full rank (linearized dynamics are surjective unless singular M)
        # Note: If M is invertible, dynamics are reversible.

        # A = C @ C.T
        # A is block tridiagonal.
        # But we use dense matmul for prototype.
        A = torch.matmul(C, C.T) # (n_con, n_con)

        # Cholesky solve or standard solve
        # A is symmetric positive definite?
        # C has full row rank?
        # v_{t+1} has coeff I, so rows are linearly independent. Yes.

        # Solve A * lambda = -b
        # We solve for each parameter column
        neg_b = -b

        # Regularize diagonal slightly for stability
        A.diagonal().add_(1e-6)

        lam = torch.linalg.solve(A, neg_b) # (n_con, n_params)

        # Recover v = -C^T lambda
        v_flat = -torch.matmul(C.T, lam) # (n_vars, n_params)

        # Reshape to (T+1, dim, n_params)
        v = v_flat.reshape(T_steps + 1, dim, n_params)

        return v

    def estimate_gradient(self, trajectory: torch.Tensor,
                        observable: Callable[[torch.Tensor], torch.Tensor],
                        dt: float) -> Dict[str, torch.Tensor]:
        """Compute stable gradient using Gradient Ghost (LSS).

        Args:
            trajectory: (n_steps, dim) or (batch, n_steps, dim)
            observable: Function O(x) -> scalar
            dt: Time step

        Returns:
            Dict of gradients per parameter
        """
        # Handle batching (average gradients over batch)
        if trajectory.ndim == 3:
            grads_list = [self.estimate_gradient(traj, observable, dt) for traj in trajectory]
            # Average
            avg_grads = {}
            for k in grads_list[0].keys():
                stacked = torch.stack([g[k] for g in grads_list])
                avg_grads[k] = stacked.mean(dim=0)
            return avg_grads

        # 1. Get Linearized Dynamics
        M, P = self._extract_linearized_dynamics(trajectory, dt)

        # 2. Solve LSS for sensitivities v_t
        # v: (n_steps, dim, n_params)
        v = self._solve_lss_system(M, P)

        # 3. Compute gradient of objective
        # J = 1/T * sum_t O(x_t)
        # dJ/dtheta = 1/T * sum_t (dO/dx_t * v_t)

        n_steps = trajectory.shape[0]

        # Compute dO/dx at each step
        # observable is usually O(x) -> scalar
        # But here we assume it's scalar output per timestep?
        # Or observable is function of whole trajectory?
        # Usually for chaotic systems, J is time average of O(x).

        # Let's assume observable computes a scalar from x
        # We need dO/dx_t

        # We'll use vmap to get dO/dx at each point
        def obs_wrapper(x):
            return observable(x.unsqueeze(0)).squeeze(0) # Handle batch dim in observable if needed

        # Gradient of O w.r.t. x (n_steps, dim)
        dO_dx = vmap(torch.func.grad(obs_wrapper))(trajectory)

        # Contract with v
        # dO_dx: (T, D), v: (T, D, P) -> (P,)
        grad_vec = torch.einsum('td,tdp->p', dO_dx, v) / n_steps

        # Map back to parameter dict
        grads = {}
        target_params = {k: p for k, p in self.integrator.named_parameters() if k in self.param_names}

        curr = 0
        for k, p in target_params.items():
            numel = p.numel()
            grads[k] = grad_vec[curr:curr+numel].view(p.shape)
            curr += numel

        return grads
