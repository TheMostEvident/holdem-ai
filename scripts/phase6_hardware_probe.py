"""Safe, tiny PyTorch CUDA compute probe for Pascal-class GPUs (optional)."""
from __future__ import annotations

import json
import os
import platform
import time


def main() -> None:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import torch
    except ImportError:
        report["torch"] = "not installed (Phase 6 engine benchmark still works)"
        print(json.dumps(report, indent=2))
        return
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        report["gpu"] = torch.cuda.get_device_name(device)
        report["compute_capability"] = torch.cuda.get_device_capability(device)
        report["vram_gb"] = round(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3), 2)
        try:
            torch.manual_seed(1)
            a = torch.randn((128, 128), device=device, requires_grad=True)
            start = time.perf_counter()
            loss = (a @ a).square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            report["forward_backward_pass"] = bool(torch.isfinite(loss).item() and torch.isfinite(a.grad).all().item())
            report["tiny_step_seconds"] = round(time.perf_counter() - start, 5)
        except Exception as exc:
            report["forward_backward_pass"] = False
            report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
