import functools
import inspect
import os
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp
import typer
from cs336_basics.training_loop import (
    AdamW,
    cross_entropy,
    train_loop_wrapper,
)
from cs336_basics.transformer import ScaledDotProductAttentionLayer, TransformerLM
from tqdm import tqdm

from cs336_systems import ddp

REPORTS_DIR = Path(__file__).parent.parent / "reports"
BACKEND = "nccl"
MAX_WORLD_SIZE = 2


def with_model_and_optim_params(func):
    """Replace func's **kwargs with TransformerLM's hyperparameters (types and defaults taken from
    train_loop_wrapper), so typer exposes them as CLI options. func still receives them via **kwargs."""
    model_param_names = inspect.signature(TransformerLM).parameters
    optim_param_names = inspect.signature(AdamW).parameters
    model_params = [
        p
        for p in inspect.signature(train_loop_wrapper).parameters.values()
        if p.name in model_param_names or p.name in optim_param_names
    ]
    sig = inspect.signature(func)
    own_params = [p for p in sig.parameters.values() if p.kind != inspect.Parameter.VAR_KEYWORD]
    func.__signature__ = sig.replace(parameters=own_params + model_params)
    return func


def add_nvtx_label_to_attention():
    orig_attention_func = ScaledDotProductAttentionLayer.forward

    @functools.wraps(orig_attention_func)
    @nvtx.range("scaled dot product attention")
    def annotated_attention_func(*args, **kwargs):
        return orig_attention_func(*args, **kwargs)

    ScaledDotProductAttentionLayer.forward = annotated_attention_func


def get_random_batch(batch_size: int, context_length: int, vocab_size: int, device: str):
    return tuple(
        torch.randint(0, vocab_size, (batch_size, context_length), device=device) for _ in range(2)
    )


def _setup_process_group(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12390"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def parallelize_model(model: torch.nn.Module, parallel_mode):
    return ddp.DDP(model, parallel_mode)


def parallelization_wraper(
    rank,
    world_size,
    vocab_size,
    num_warmup_steps,
    num_iters,
    batch_size,
    do_backwards,
    do_optim_step,
    profile_memory,
    parallel_mode,
    output_queue,
    kwargs,
):
    device = _setup_process_group(rank, world_size, BACKEND)
    model = TransformerLM(vocab_size=vocab_size, **kwargs).to(device)
    model = parallelize_model(model, parallel_mode)
    optimizer = AdamW(model.parameters(), **kwargs)

    def do_step():
        optimizer.zero_grad()
        batch, batch_target = get_random_batch(
            batch_size, kwargs["context_length"], vocab_size, device
        )
        logits = model(batch)
        loss = cross_entropy(logits, batch_target)
        if do_backwards:
            with nvtx.range("backward"):
                loss.backward()
                model.finish_gradient_synchronization(optimizer)
                if do_optim_step:
                    with nvtx.range("step"):
                        optimizer.step()

    model.train()
    with nvtx.range("warmup"):
        for _ in tqdm(range(num_warmup_steps), desc="Warmup"):
            do_step()
    measurements = []
    if profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")
    with nvtx.range("measure"):
        for _ in tqdm(range(num_iters), desc="Measurment iters"):
            start_time = timeit.default_timer()
            dist.barrier()
            do_step()
            dist.barrier()
            end_time = timeit.default_timer()
            measurements.append(end_time - start_time)
    if profile_memory:
        torch.cuda.memory._dump_snapshot(str(REPORTS_DIR / "memory.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)
    if rank == 0:
        measurements = np.array(measurements)
        output_queue.put((measurements.mean(), measurements.std()))


@with_model_and_optim_params
def benchmark_transformer(
    num_warmup_steps: int = 0,
    num_iters: int = 10,
    batch_size: int = 16,
    profile_memory: bool = False,
    vocab_size: int = 10_000,
    do_backwards: bool = True,
    do_optim_step: bool = True,
    world_size: int = 2,
    parallel_mode: str = "naive_ddp",
    **kwargs,
):
    add_nvtx_label_to_attention()
    assert world_size <= MAX_WORLD_SIZE
    batch_size = batch_size // world_size
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    mp.spawn(
        fn=parallelization_wraper,
        args=(
            world_size,
            vocab_size,
            num_warmup_steps,
            num_iters,
            batch_size,
            do_backwards,
            do_optim_step,
            profile_memory,
            parallel_mode,
            output_queue,
            kwargs,
        ),
        nprocs=world_size,
        join=True,
    )
    mean, std = output_queue.get()
    stats = pd.DataFrame(
        {
            "World Size": [world_size],
            "Vocab Size": [vocab_size],
            "Parallel Mode": [parallel_mode],
            "Num Iters": [num_iters],
            "Mean Iter Time": [mean],
            "Iter Time STD": [std],
        }
    )
    print(stats)


if __name__ == "__main__":
    typer.run(benchmark_transformer)
