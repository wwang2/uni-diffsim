"""Device utilities for CPU/CUDA/MPS support."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device.
    
    Args:
        preference: "auto", "cpu", "cuda", or "mps"
    
    Returns:
        torch.device for computation
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS requested but not available")
    
    # Auto-detect
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(x, device: torch.device | str | None = None):
    """Move tensor or module to device."""
    if device is None:
        device = get_device()
    if isinstance(device, str):
        device = torch.device(device)
    return x.to(device)


def available_devices() -> list[str]:
    """Return list of available device names."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


if __name__ == "__main__":
    import os
    import time
    import matplotlib.pyplot as plt
    
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

