"""Fail closed rather than silently running multiple unsynchronized trainers."""
import os


def require_single_process(environ=None):
    """Validate launcher metadata for the pinned DataParallel-only trainer.

    This does not implement DDP, initialize NCCL, select CUDA devices, or certify
    multi-node correctness. A DDP migration requires a separately tested trainer.
    """
    environ = os.environ if environ is None else environ
    world_size = int(environ.get("WORLD_SIZE", "1"))
    rank = int(environ.get("RANK", "0"))
    local_rank = int(environ.get("LOCAL_RANK", "0"))
    if world_size != 1 or rank != 0 or local_rank != 0:
        raise RuntimeError(
            "Pinned HiFT supports one DataParallel process only; "
            "multi-process launch requires an explicit DDP implementation"
        )
    return 0, 1
