
import os
import time
import torch
import matplotlib.pyplot as plt
from uni_diffsim.device import available_devices, get_device

assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(assets_dir, exist_ok=True)

print("Device Detection Report")
print("=" * 40)

devices = available_devices()
print(f"Available devices: {devices}")
print(f"Auto-selected: {get_device()}")

# Benchmark simple ops on each device
results = {}
n = 1000
for dev_name in devices:
    device = torch.device(dev_name)
    x = torch.randn(n, n, device=device)

    # Warmup
    _ = x @ x
    if dev_name == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        _ = x @ x
    if dev_name == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 10
    results[dev_name] = elapsed * 1000
    print(f"{dev_name}: {elapsed*1000:.2f} ms for {n}x{n} matmul")

fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#4a90d9", "#d94a4a", "#4ad94a"][:len(results)]
ax.bar(results.keys(), results.values(), color=colors)
ax.set_ylabel("Time (ms)")
ax.set_title(f"Matrix Multiplication ({n}x{n})")
ax.set_xlabel("Device")

plt.tight_layout()
plt.savefig(os.path.join(assets_dir, "device_benchmark.png"), dpi=150)
print(f"\nSaved benchmark plot to assets/device_benchmark.png")
