"""Unified device selection: XPU > CUDA > CPU.

No `.cuda()` anywhere in the codebase; everything uses `.to(device)`.
"""
from __future__ import annotations

import torch


def get_device(verbose: bool = True, override: str | None = None) -> torch.device:
    if override and override != "auto":
        device = torch.device(override)
        name = override
        if verbose:
            print(f"[device] forced to {device} by --device")
        return device
    xpu_ok = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda_ok = torch.cuda.is_available()
    if xpu_ok:
        device = torch.device("xpu")
        name = torch.xpu.get_device_name(0)
    elif cuda_ok:
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        name = "cpu"
    if verbose:
        print(f"[device] torch version : {torch.__version__}")
        print(f"[device] xpu available : {xpu_ok}")
        print(f"[device] cuda available: {cuda_ok}")
        print(f"[device] selected      : {device} ({name})")
        print(f"[device] dtype         : float32 (phase 1, no mixed precision)")
    return device
