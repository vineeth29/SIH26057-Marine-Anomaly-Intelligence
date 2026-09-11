"""
Post-install CUDA verification and training launch script.
Run this after torch cu121 is installed in train_env.

Usage:
    .\train_env\Scripts\python.exe scripts\verify_and_train.py

This script:
1. Verifies CUDA is available and prints hardware report
2. Runs a quick GPU sanity test (allocate, compute, free)
3. Prints recommended imgsz/batch for RTX 3050 4GB
4. Launches training

DO NOT run this script until train_env has cu121 torch installed.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def check_cuda():
    """Full CUDA verification — this is the gating check."""
    print("=" * 60)
    print("STEP 1: CUDA VERIFICATION")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("FATAL: torch not installed in this environment.")
        print("Run from train_env: .\\train_env\\Scripts\\python.exe scripts\\verify_and_train.py")
        sys.exit(1)

    print(f"  PyTorch version:       {torch.__version__}")
    print(f"  CUDA built version:    {torch.version.cuda}")
    print(f"  CUDA available:        {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("")
        print("FATAL: torch.cuda.is_available() == False")
        print("This environment does NOT have CUDA support.")
        print("")
        print("Possible causes:")
        print("  1. torch was installed without CUDA (cpu-only build)")
        print("  2. CUDA runtime not in PATH")
        print("  3. NVIDIA driver too old")
        print("")
        print("Fix:")
        print("  .\\train_env\\Scripts\\pip.exe install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \\")
        print("    --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
        sys.exit(1)

    # Print full hardware info
    torch.cuda.init()
    props = torch.cuda.get_device_properties(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    vram_total = round(total_bytes / 1024**3, 2)
    vram_free = round(free_bytes / 1024**3, 2)

    print("")
    print(f"  GPU Name:              {props.name}")
    print(f"  Compute Capability:    {props.major}.{props.minor}")
    print(f"  VRAM Total:            {vram_total} GB")
    print(f"  VRAM Free:             {vram_free} GB")
    print(f"  CUDA device count:     {torch.cuda.device_count()}")
    print("")

    # Sanity: allocate a 256MB tensor, run matmul, free
    print("  Running GPU sanity test (256MB allocation + matmul)...")
    try:
        import time
        n = 4096
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        t0 = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        del a, b, c
        torch.cuda.empty_cache()
        print(f"  GPU sanity test: PASS ({elapsed_ms:.1f}ms for {n}x{n} matmul)")
    except Exception as e:
        print(f"  GPU sanity test: FAIL — {e}")
        sys.exit(1)

    print("")
    print("CUDA VERIFICATION: PASS")
    print("=" * 60)

    return vram_total, vram_free


def main():
    vram_total, vram_free = check_cuda()

    # Determine config
    usable = vram_free - 0.5
    if usable >= 3.2:
        imgsz, batch = 1024, 2
        label = "imgsz=1024, batch=2"
    elif usable >= 2.8:
        imgsz, batch = 800, 8
        label = "imgsz=800, batch=8"
    elif usable >= 2.2:
        imgsz, batch = 640, 8
        label = "imgsz=640, batch=8"
    elif usable >= 1.6:
        imgsz, batch = 640, 4
        label = "imgsz=640, batch=4"
    else:
        imgsz, batch = 640, 2
        label = "imgsz=640, batch=2"

    print("")
    print("=" * 60)
    print("STEP 2: RECOMMENDED TRAINING CONFIG")
    print("=" * 60)
    print(f"  VRAM total:   {vram_total} GB")
    print(f"  VRAM free:    {vram_free} GB")
    print(f"  Usable:       {usable:.2f} GB (after 0.5 GB margin)")
    print(f"  Config:       {label}")
    print(f"  Model:        YOLOv8n")
    print(f"  AMP:          enabled")
    print(f"  Epochs:       100 (early stop patience=20)")
    print("=" * 60)

    # Launch training
    print("")
    print("=" * 60)
    print("STEP 3: LAUNCHING TRAINING")
    print("=" * 60)
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_multiclass.py"),
        f"--imgsz", str(imgsz),
        f"--batch", str(batch),
        "--epochs", "100",
        "--patience", "20",
    ]
    print(f"  Command: {' '.join(cmd)}")
    print("")

    ret = subprocess.run(cmd, cwd=str(ROOT))
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
