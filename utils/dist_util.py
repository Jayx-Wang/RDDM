import os

import torch as th
import torch.distributed as dist


def setup_dist():
    if dist.is_available() and dist.is_initialized():
        return

    backend = "nccl" if th.cuda.is_available() else "gloo"
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    if th.cuda.is_available():
        th.cuda.set_device(local_rank)


def dev():
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")
