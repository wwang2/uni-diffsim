
import torch
import time
from uni_diffsim.potentials import LennardJones, LennardJonesVerlet

def benchmark_verlet():
    print("Benchmarking Verlet Neighbor List vs Naive LJ (PBC)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Setup
    N_list = [500, 1000, 2000]
    box_L = 20.0
    cutoff = 2.5
    skin = 0.5
    
    for N in N_list:
        print(f"\n--- N = {N} ---")
        x = torch.rand(N, 3, device=device) * box_L
        x.requires_grad_(True)
        
        # 1. Naive (Standard vectorized O(N^2) energy calc)
        # Note: The 'LennardJones' class does O(N^2) at every energy call.
        lj_naive = LennardJones(box_size=box_L).to(device)
        
        # Warmup
        lj_naive.energy(x)
        
        t0 = time.time()
        n_steps = 20
        for _ in range(n_steps):
            e = lj_naive.energy(x)
            g = torch.autograd.grad(e.sum(), x, retain_graph=True)[0]
        t_naive = (time.time() - t0) / n_steps
        print(f"Naive (avg/step): {t_naive*1000:.2f} ms")
        
        # 2. Verlet List
        # Update neighbor list once, then run energy n_steps
        lj_verlet = LennardJonesVerlet(box_size=box_L, cutoff=cutoff, skin=skin).to(device)
        
        # Build list (overhead)
        t_build_start = time.time()
        n_pairs = lj_verlet.update_neighbor_list(x)
        t_build = time.time() - t_build_start
        print(f"Verlet Build List: {t_build*1000:.2f} ms (Pairs: {n_pairs})")
        
        # Run energy (using list)
        t0 = time.time()
        for _ in range(n_steps):
            e = lj_verlet.energy(x)
            g = torch.autograd.grad(e.sum(), x, retain_graph=True)[0]
        t_verlet = (time.time() - t0) / n_steps
        print(f"Verlet Energy (avg/step): {t_verlet*1000:.2f} ms")
        
        speedup = t_naive / t_verlet
        print(f"Speedup (Energy only): {speedup:.2f}x")
        
        # Total time (amortized if we update every 20 steps)
        t_total_avg = (t_build + t_verlet * n_steps) / n_steps
        print(f"Amortized Total (update every {n_steps}): {t_total_avg*1000:.2f} ms")
        print(f"Overall Speedup: {t_naive / t_total_avg:.2f}x")

if __name__ == "__main__":
    benchmark_verlet()

