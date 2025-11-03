import sys
import os

#!/usr/bin/env python3
"""Quick check if torch has CUDA available."""

def main():
    try:
        import torch
    except ImportError:
        print("torch: NOT INSTALLED")
        return 1
    print("python:", sys.version)
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print("current_device:", torch.cuda.current_device())
        print("device_name:", torch.cuda.get_device_name(0))

    if hasattr(torch, "cuda"):
        try:
            count = torch.cuda.device_count()
            current = torch.cuda.current_device()
            name = torch.cuda.get_device_name(current)
        except Exception:
            # safe fallback if some calls fail
            count = getattr(torch.cuda, "device_count", lambda: 0)()
            current = getattr(torch.cuda, "current_device", lambda: -1)()
            name = "<unknown>"
        print(f"device count: {count}")
        print(f"current device: {current} ({name})")

    return 0

if __name__ == "__main__":
    sys.exit(main())