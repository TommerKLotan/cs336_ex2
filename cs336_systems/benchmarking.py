import functools
import inspect
import timeit
from pathlib import Path

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
import typer
from cs336_basics.training_loop import AdamW, cross_entropy, get_batch, get_vocab_size, load_datasets, train_loop_wrapper
from cs336_basics.transformer import ScaledDotProductAttentionLayer, TransformerLM

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def with_model_and_optim_params(func):
    """Replace func's **kwargs with TransformerLM's hyperparameters (types and defaults taken from
    train_loop_wrapper), so typer exposes them as CLI options. func still receives them via **kwargs."""
    model_param_names = inspect.signature(TransformerLM).parameters
    optim_param_names = inspect.signature(AdamW).parameters
    model_params = [p for p in inspect.signature(train_loop_wrapper).parameters.values() if p.name in model_param_names or p.name in optim_param_names]
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


@with_model_and_optim_params
def benchmark_transformer(
    num_warmup_steps: int = 5,
    num_iters: int = 10,
    batch_size: int = 16,
    profile_memory: bool = False,
    tokenization_model_name: str = "TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45",
    dataset_name: str = "TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45-TinyStoriesV2-GPT4",
    do_backwards: bool = True,
    do_optim_step: bool = True,
    **kwargs,
):
    add_nvtx_label_to_attention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, _ = load_datasets(dataset_name)
    model = TransformerLM(vocab_size=get_vocab_size(tokenization_model_name), **kwargs).to(device)
    optimizer = AdamW(model.parameters(), **kwargs)
    batch, batch_target = get_batch(train_dataset, batch_size, kwargs["context_length"], device)

    def do_computation():
        optimizer.zero_grad()
        model.train()
        logits = model(batch)
        loss = cross_entropy(logits, batch_target)
        if do_backwards:
            with nvtx.range("backward"):
                loss.backward()
                if do_optim_step:
                    with nvtx.range("step"):
                        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

    with nvtx.range("warmup"):
        for _ in range(num_warmup_steps):
            do_computation()

    measurements = []
    with nvtx.range("measure"):
        if profile_memory:
            torch.cuda.memory._record_memory_history(max_entries=1_000_000, stacks="python")
        for _ in range(num_iters):
            start_time = timeit.default_timer()
            do_computation()
            end_time = timeit.default_timer()
            measurements.append(end_time - start_time)
        if profile_memory:
            torch.cuda.memory._dump_snapshot(str(REPORTS_DIR / "memory.pickle"))
            torch.cuda.memory._record_memory_history(enabled=None)
    measurements = np.array(measurements)
    mean = np.mean(measurements)
    std = np.std(measurements)
    print(f"Ran {num_iters} measurements with mean: {mean}, std: {std}")


if __name__ == "__main__":
    typer.run(benchmark_transformer)
