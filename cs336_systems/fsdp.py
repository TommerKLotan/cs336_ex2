import os
from collections import OrderedDict
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.training import cross_entropy
from cs336_basics.transformer import EmbeddingLayer, LinearLayer, TransformerLM
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_systems.ddp import wait_on_handles

MASTER_RANK = 0


def master_print(*args, **kwargs):
    rank = kwargs.pop("rank", 0)
    if dist.get_rank() == rank:
        print(*args, **kwargs)


def should_shard_layer(m: torch.nn.Module) -> bool:
    return isinstance(m, LinearLayer) or isinstance(m, EmbeddingLayer)


class FSDP(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        compute_dtype: torch.dtype | None = None,
        anticipation_distance: int = 2,
    ):
        super().__init__()
        self.device = "cuda"
        self.anticipation_distance = anticipation_distance
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.handles = list()
        self.is_order_recorded = False
        self.order = []
        self.current_layer_ind: int = 0
        self.base_layers = [m for m in module.modules() if len(list(m.children())) == 0]

        # Set up order recording hooks for first forward pass
        self.order_recording_hooks = [
            m.register_forward_pre_hook(self._record_order_hook, with_kwargs=True)
            for m in self.base_layers
        ]

        # Initialize sharded layer states
        self.sharded_layer_states: OrderedDict[torch.nn.Module, list[Any]] = OrderedDict(
            (m, [0, "filled", "ready"]) for m in self.base_layers if should_shard_layer(m)
        )
        self.sharded_layer_params = OrderedDict(
            (m, torch.nn.Parameter(self._shard_module(m)))
            for m, _ in self.sharded_layer_states.items()
        )

        # Set up pre hooks that queue fillings. Only on sharded layers
        self.forward_pre_hooks = [
            m.register_forward_pre_hook(self._pre_forward_hook, with_kwargs=True)
            for m, _ in self.sharded_layer_states.items()
        ]
        # Set up post hooks
        self.forward_post_hooks = [
            m.register_forward_hook(self._post_forward_hook, with_kwargs=True)
            for m in self.base_layers
        ]
        # Set up backward pre hooks
        self.backward_pre_hooks = [
            m.register_full_backward_pre_hook(self._pre_backward_hook)
            for m, _ in self.sharded_layer_states.items()
        ]
        # Set up backward pass post hooks
        self.backward_post_hooks = [
            self._register_layer_grad_ready_hook(m, self._post_backward_hook)
            for m in self.base_layers
        ]

        self._register_non_sharded_params_grad_hooks()

    def _is_module_filled(self, m: torch.nn.Module):
        return all(p.untyped_storage().nbytes() != 0 for p in m.parameters())

    def _fill_layer(self, m: torch.nn.Module, with_grad: bool = False, do_async=False):
        master_print(f"filling layer: {m},{id(m)}")
        shard = self.sharded_layer_params[m].data
        gathered_param_buffer = torch.empty(self.world_size * shard.size(0), device=self.device)
        gathered_param_data: torch.Tensor = _unflatten_dense_tensors(
            gathered_param_buffer, tuple(m.parameters())
        )
        for param, param_data in zip(m.parameters(), gathered_param_data):
            param.untyped_storage().resize_(param_data.untyped_storage().nbytes())
            param.data.copy_(param_data)
            if with_grad:
                param.grad = None  # This is so autograd allocates a new tensor

        return dist.all_gather_into_tensor(gathered_param_buffer, shard, async_op=do_async)

    def _shard_module(self, m: torch.nn.Module):
        params = tuple(m.parameters())
        # TODO maybe make this more memory efficient
        shard = self._get_param_shard(_flatten_dense_tensors(params))
        self._empty_layer(m)
        return shard

    def _empty_layer(self, m: torch.nn.Module, with_grad: bool = False):
        master_print(f"emptying layer: {m}")
        for param in m.parameters():
            if param.untyped_storage().nbytes() == 0:
                break
            else:
                param.untyped_storage().resize_(0)
        if with_grad:
            layer_state = self.sharded_layer_states[m]
            sharded_layer_params = self.sharded_layer_params[m]
            if layer_state[2] != "ready":
                layer_state[2].wait()
            flat_grad = _flatten_dense_tensors(tuple(p.grad for p in m.parameters()))
            grad_shard = self._get_param_shard(flat_grad, clone=False)
            if sharded_layer_params.grad is not None:
                grad_shard.add_(sharded_layer_params.grad)
                sharded_layer_params.grad = grad_shard
            layer_state[2] = dist.reduce_scatter_tensor(
                grad_shard, flat_grad, dist.ReduceOp.SUM, async_op=True
            )
            for param in m.parameters():
                grad = param.grad
                if grad is not None:
                    if grad.untyped_storage().nbytes() == 0:
                        break
                    else:
                        grad.untyped_storage().resize_(0)
        self.sharded_layer_states[m][1] = "empty"

    def _get_param_shard(self, p, rank=None, clone: bool = True) -> torch.Tensor:
        if rank is None:
            rank = self.rank
        flat = p.view(-1)
        l = flat.size(0)
        shard_size = l // self.world_size
        shard = flat[rank * shard_size : (rank + 1) * shard_size]
        if clone:
            shard = shard.detach().clone()
        return shard

    def _record_order_hook(self, m, inputs, kwargs):
        self.order.append(m)
        return inputs, kwargs

    def _queue_layer_filling(self, m: torch.nn.Module, with_grad: bool = False):
        layer_state = self.sharded_layer_states.get(m, None)
        master_print(f"queing {m} , state: {layer_state}, {id(m)}")
        if layer_state is None:
            return
        layer_state[0] += 1
        if layer_state[1] == "empty":
            layer_state[1] = self._fill_layer(m, with_grad, do_async=True)

    def _register_layer_grad_ready_hook(self, layer: torch.nn.Module, callback):
        params = [p for p in layer.parameters() if p.requires_grad]
        state = {"params_left_to_wait_for": len(params)}

        def param_callback(p):
            state["params_left_to_wait_for"] -= 1
            if state["params_left_to_wait_for"] == 0:
                state["params_left_to_wait_for"] = len(params)
                callback(layer)

        for p in params:
            p.register_post_accumulate_grad_hook(param_callback)

    def _release_layer(self, m: torch.nn.Module, with_grad: bool = False):
        layer_state = self.sharded_layer_states.get(m, None)
        if layer_state is None:
            return
        layer_state[0] -= 1
        # master_print(f"releasing layer {m}, count: {layer_state[0]}")
        if layer_state[0] == 0:  # No one is queing the layer
            self._empty_layer(m, with_grad)

    def _wait_for_layer(self, m: torch.nn.Module, with_grad: bool = False):
        layer_state = self.sharded_layer_states[m]
        if layer_state[1] == "empty":
            self._queue_layer_filling(m, with_grad)
        if layer_state[1] != "filled":
            master_print(f"waiting for layer: {m},{id(m)}")
            handle = layer_state[1]
            handle.wait()  # type: ignore
            layer_state[1] = "filled"

    def _pre_forward_hook(self, m, inputs, kwargs):
        self._wait_for_layer(m)

    def _traverse_layer(self, m, inverse_order: bool = False, with_grad: bool = False):
        last_anticipated_layer = self.current_layer_ind + self.anticipation_distance
        first_anticipated_layer = 0 if self.current_layer_ind == -1 else last_anticipated_layer
        for layer in (self.order[::-1] if inverse_order else self.order)[
            first_anticipated_layer : last_anticipated_layer + 1
        ]:
            self._queue_layer_filling(layer, with_grad)
        self._release_layer(m, with_grad)
        self.current_layer_ind += 1
        if self.current_layer_ind == len(self.order):
            self.current_layer_ind = -1

    def _post_forward_hook(self, m, inputs, kwargs, output):
        self._traverse_layer(m)

    def _pre_backward_hook(self, m, output_grad):
        if self.current_layer_ind == -1:
            self._traverse_layer(None, inverse_order=True, with_grad=True)
        self._wait_for_layer(m)

    def _post_backward_hook(self, m):
        master_print(f"finished back on layer: {m}")
        self._traverse_layer(m, inverse_order=True, with_grad=True)

    def _register_non_sharded_params_grad_hooks(self):
        d = {v: k for k, v in self.module.named_parameters()}

        def grad_hook(param):
            master_print(f"param: {d[param]}")
            self.handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True))

        sharded_params = sum(
            ([d[p] for p in m.parameters()] for m in self.sharded_layer_params.keys()), []
        )

        master_print("bla")
        dist.barrier()
        for name, param in self.module.named_parameters():
            if name in sharded_params:
                continue
            param.register_post_accumulate_grad_hook(grad_hook)
        master_print("bad", sharded_params)

    def forward(self, *inputs, **kwargs):
        self.current_layer_ind = -1
        self._traverse_layer(None)  # Queue first layers

        output = self.module(*inputs, **kwargs)
        if not self.is_order_recorded:
            self.is_order_recorded = True
            for hook in self.order_recording_hooks:
                hook.remove()
            self.bck_current_layer_ind = len(self.order)

        return output

    def finish_gradient_synchronization(self):
        handles = [
            h for h in (state[2] for state in self.sharded_layer_states.values()) if h != "ready"
        ]
        wait_on_handles(handles)
        wait_on_handles(self.handles)


DEVICE = "cuda"
PARALLELISM_BACKEND = "nccl" if DEVICE == "cuda" else "gloo"
WORLD_SIZE = 2


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(PARALLELISM_BACKEND, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def distributed_main(rank, world_size):
    setup(rank, world_size)
    d_model = 32
    num_heads = 4
    context_length = 8
    vocab_size = 10
    num_layers = 1
    d_ff = 16
    batch_size = 2
    device = "cuda"
    transformer = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta=1000,
        device=device,
    )

    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    label = torch.randint(0, vocab_size, (batch_size, context_length))
    fsdp = FSDP(transformer, compute_dtype=torch.float16)
    y = fsdp(x)
    master_print("first output:", y)

    y = fsdp(x)
    master_print("second output:", y)
    loss = cross_entropy(y, label)

    def hook(*args):
        master_print("here!")

    loss.backward()

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(fn=distributed_main, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
