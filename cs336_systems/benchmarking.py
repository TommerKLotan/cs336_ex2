import inspect
import timeit

import numpy as np
import torch
import typer
from cs336_basics.training_loop import AdamW, cross_entropy, get_batch, get_vocab_size, load_datasets, train_loop_wrapper
from cs336_basics.transformer import TransformerLM


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


@with_model_and_optim_params
def benchmark_transformer(
    num_warmup_steps: int = 5,
    num_iters: int = 10,
    batch_size: int = 16,
    tokenization_model_name: str = "TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45",
    dataset_name: str = "TinyStoriesV2-GPT4-train-bpe-2026-10-08_19-11-45-TinyStoriesV2-GPT4",
    do_backwards: bool = True,
    do_optim_step: bool = True,
    **kwargs,
):
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
            loss.backward()
            if do_optim_step:
                optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(num_warmup_steps):
        do_computation()

    measurements = []
    for _ in range(num_iters):
        start_time = timeit.default_timer()
        do_computation()
        end_time = timeit.default_timer()
        measurements.append(end_time - start_time)

    measurements = np.array(measurements)
    mean = np.mean(measurements)
    std = np.std(measurements)
    print(f"Ran {num_iters} measurements with mean: {mean}, std: {std}")


if __name__ == "__main__":
    typer.run(benchmark_transformer)
