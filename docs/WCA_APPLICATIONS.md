# Cool Applications for WCA Dimer System

This document outlines several advanced applications of differentiable simulations using the `DimerWCA` system.

## 1. Solvation Design (Implemented in `scripts/demo_wca_solvation_design.py`)

**Goal:** Design the solvent properties to control the conformational state of a solute molecule.

**Description:**
The WCA dimer is a bistable molecule (Compact $\leftrightarrow$ Extended) immersed in a dense fluid. The relative population of the two states depends on the solvent pressure and interaction strength (`eps`).
Using differentiable simulation, we can automatically optimize the solvent's Lennard-Jones epsilon parameter to maximize the probability of the dimer being in the "Extended" state.

**Method:**
- **System**: `DimerWCA` with N=66 particles (2 dimer + 64 solvent).
- **Optimization Target**: Maximize $P(\text{extended})$.
- **Algorithm**: REINFORCE gradient estimator.
  - $\nabla_\theta \mathcal{L} \approx -\mathbb{E}[R \nabla_\theta \log p(\tau)]$
- **Results**: The optimizer adjusts `eps` to shift the equilibrium distribution.

**Run the demo:**
```bash
python -m scripts.demo_wca_solvation_design
```

## 2. Parameter Inference (Inverse Design)

**Goal:** Recover the true interaction parameters of a system given only macroscopic observables.

**Description:**
Suppose we observe an experimental radial distribution function (RDF) or average bond length, but we don't know the microscopic interaction strength (`eps` or `sigma`). We can start with a guess and use gradient descent to match the observation.

**Method:**
- **Loss Function**: $\mathcal{L} = (\langle r \rangle_{sim} - r_{target})^2$.
- **Algorithm**: `ReinforceEstimator` or `ImplicitDiffEstimator`.

## 3. Reaction Path Optimization (Steered MD)

**Goal:** Find an optimal time-dependent external force to drive the transition from Compact to Extended in minimal time or with minimal work.

**Description:**
Rare events like conformational changes can be accelerated by external forces. Differentiable simulation allows us to learn the optimal force profile $F(t)$ (parameterized by a neural network or spline) to facilitate this transition.

**Method:**
- **Control**: Add a time-dependent external force term to the potential.
- **Loss**: Penalty for failing to reach the target state + penalty for total work done.
- **Algorithm**: Backpropagation through time (BPTT) or Adjoint method (`CheckpointedNoseHoover`).

## 4. Temperature Scheduling (Simulated Annealing)

**Goal:** Optimize a temperature schedule $T(t)$ to efficiently sample both states and overcome the barrier.

**Description:**
Similar to reaction path optimization, but optimizing the thermostat temperature. This can be used to discover efficient annealing protocols automatically.
