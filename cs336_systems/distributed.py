import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEVICE = "cuda"
PARALLELISM_BACKEND = "nccl" if DEVICE == "cuda" else "gloo"
WORLD_SIZE = 2


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(PARALLELISM_BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,), device=DEVICE)
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(fn=distributed_demo, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
