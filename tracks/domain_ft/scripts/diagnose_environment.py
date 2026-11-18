"""Inspect actual environment without assuming CUDA, torch, or large assets exist."""
import argparse
import ctypes
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def diagnose():
    root = Path(__file__).resolve().parents[1]
    usage = shutil.disk_usage(root)
    result = {
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(), "python": sys.version, "executable": sys.executable,
        "cpu": platform.processor(), "logical_cpus": __import__('os').cpu_count(),
        "disk": {"path": str(root), "total_bytes": usage.total, "free_bytes": usage.free},
        "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader"]),
        "git": command(["git", "rev-parse", "HEAD"]),
        "upstream": command(["git", "-C", str(root / "third_party/HiFT"), "rev-parse", "HEAD"]),
    }
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virt", "avail_virt", "avail_extended")]
        mem = MemoryStatus()
        mem.length = ctypes.sizeof(mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            result["memory"] = {"total_bytes": mem.total_phys, "available_bytes": mem.avail_phys}
    result["packages"] = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    try:
        import torch
        result["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            result["torch"].update({"gpu": prop.name, "vram_bytes": prop.total_memory, "capability": list(torch.cuda.get_device_capability(0)), "compiled_arches": torch.cuda.get_arch_list()})
            a = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
            result["torch"]["cuda_matmul_smoke_sum"] = (a @ a.T).sum().item()
    except Exception as exc:
        result["torch_error"] = repr(exc)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(diagnose(), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
