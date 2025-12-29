
# on unifying differentiable simulations

> “All chaos is order misunderstood.”    — Alexander Pope

This project explores a question I've wondered about for years:

> What does it mean to "differentiate through simulation"?


```bash
pip install -e ".[dev]"
```


## Benchmark implemented simulators

![Integrators](assets/integrators.png)


| Integrator | Type | Notes |
|------------|------|-------|
| `OverdampedLangevin` | Stochastic | High friction |
| `BAOAB` | Stochastic | NVT sampler |
| `VelocityVerlet` | Deterministic | NVE, symplectic |
| `NoseHooverChain` | Deterministic, no Ergodicity guarantee | NVT thermostat |
| `ESH` | Deterministic | no Ergodicity guarantee, very finicky ([arXiv:2111.02434](https://arxiv.org/abs/2111.02434)) |
| `GLE` | Stochastic | Colored noise |



## Potentials

![Potentials](assets/potentials.png)

| Potential | Dim | Use |
|-----------|-----|-----|
| `DoubleWell` | 1D | Barrier crossing |
| `AsymmetricDoubleWell` | 1D | Metastable populations |
| `MullerBrown` | 2D | Reaction paths |
| `LennardJones` | N×d | Clusters |
| `Harmonic` | d | Reference |


## Gradient Estimators

![Gradient Estimators](assets/gradient_estimators.png)

Two approaches for computing gradients of equilibrium observables with respect to potential parameters:

| Estimator | Method | Pros | Cons |
|-----------|--------|------|------|
| **BPTT** | Backprop through trajectory | Exact for finite horizon | Exploding gradients, memory-intensive |
| **REINFORCE** | Score function / TPT | Stable, O(1) memory | Higher variance, needs equilibrium samples |

The REINFORCE estimator uses the thermodynamic perturbation theory identity:
```
∇_θ ⟨O⟩ = -β Cov(O, ∇_θU) = -β [⟨O ∇_θU⟩ - ⟨O⟩⟨∇_θU⟩]
```

**Key findings on Asymmetric Double-Well** (see plot above):
1. **Well occupation gradient**: REINFORCE accurately estimates ∂P_right/∂b while BPTT underestimates by ~4x
2. **Optimization**: REINFORCE converges to optimal asymmetry (b→0) for equal well occupation; BPTT stalls
3. **Stability**: BPTT gradients become biased (~2x) at long trajectories; REINFORCE remains stable
4. **Harmonic validation**: Both methods agree with theory `d⟨x²⟩/dk = -kT/k²`

```bash
python -m uni_diffsim.gradient_estimators  # Generate the plot
```


## Usage

```python
import torch
from uni_diffsim import DoubleWell, BAOAB, ESH

# Setup
potential = DoubleWell()
x0 = torch.randn(100, 1)  # 100 particles

# Stochastic sampling
baoab = BAOAB(gamma=1.0, kT=0.5)
traj_x, traj_v = baoab.run(x0, None, potential.force, dt=0.01, n_steps=1000)

# Deterministic ergodic sampling (2D+)
x0_2d = torch.randn(10, 2)
esh = ESH(eps=0.1)
traj_x, traj_u, traj_r = esh.run(x0_2d, None, lambda x: -potential.force(x), n_steps=1000)

# Gradients work
k = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([[1.0]])
for _ in range(10):
    x, _, _ = esh.step(x, torch.randn(1, 1), torch.zeros(1), lambda x: k * x)
x.sum().backward()  # k.grad is defined
```

## Tests

```bash
pytest tests/ -v
```

