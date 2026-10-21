from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

MASTER_RANK = 0


def wait_on_handles(handles):
    for handle in handles:
        handle.wait()
    handles.clear()


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, mode="flatten_ddp"):
        super().__init__()
        self.module = module
        self.mode = mode
        self.handles = list()
        handles = [
            dist.broadcast(param.data, MASTER_RANK, async_op=True)
            for param in self._get_shared_params()
        ]
        wait_on_handles(handles)

        if self.mode == "grad_hook_ddp":
            self._register_grad_hooks()

    def _get_grad_params(self):
        return [param for param in self._get_shared_params() if param.requires_grad]

    def _get_shared_params(self):
        return [
            param
            for param_name, param in self.module.named_parameters()
            if "no_grad_fixed_param" not in param_name
        ]

    def _register_grad_hooks(self):
        def grad_hook(param):
            self.handles.append(dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True))

        for param in self._get_grad_params():
            param.register_post_accumulate_grad_hook(grad_hook)

    def forward(self, data):
        output = self.module(data)
        return output

    def finish_gradient_synchronization(self, optimizer: torch.optim.Optimizer):
        grad_tensors = [p.grad for p in self._get_grad_params()]
        if self.mode == "naive_ddp":
            handles = [
                dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True) for grad in grad_tensors
            ]
            wait_on_handles(handles)
        elif self.mode == "flatten_ddp":
            flattened = _flatten_dense_tensors(grad_tensors)
            dist.all_reduce(flattened, op=dist.ReduceOp.AVG, async_op=False)
            new_grad_tensors = _unflatten_dense_tensors(flattened, grad_tensors)
            for grad, new_grad in zip(grad_tensors, new_grad_tensors):
                grad.copy_(new_grad)
        elif self.mode == "grad_hook_ddp":
            wait_on_handles(self.handles)


@contextmanager
def have_params_changed(param_groups):
    orig = sum(([x.detach().clone() for x in group["params"]] for group in param_groups), [])
    yield
    new = sum((group["params"] for group in param_groups), [])
    print(f"have grad: {len([x for x in new if x.grad is not None])}")
    print(f"{len(new)} params, changed: {any(not torch.allclose(a, b) for a, b in zip(orig, new))}")


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: Any):
        params = list(params)
        if not isinstance(params[0], dict):
            param_groups = [{"params": params}]
        else:
            param_groups = params
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        shard_params = []
        for param_group in param_groups:
            shard_params.append(self._shard_group(param_group))
        self.optimizer = optimizer_cls(shard_params, **kwargs)
        super().__init__(params, dict())

    def _get_param_shard(self, p, rank=None):
        with torch.no_grad():
            if rank is None:
                rank = self.rank
            flat = p.view(-1)
            l = flat.size(0)
            shard_size = l // self.world_size
            return flat[rank * shard_size : (rank + 1) * shard_size]

    def step(self, closure=None, **kwargs):
        for param_group, sharded_param_group in zip(self.param_groups, self.optimizer.param_groups):
            for p, sharded_p in zip(param_group["params"], sharded_param_group["params"]):
                with torch.no_grad():
                    if p.grad is not None:
                        sharded_p.grad = self._get_param_shard(p.grad)

        self.optimizer.step(closure, **kwargs)
        handles = []
        with torch.no_grad():
            for param_group in self.param_groups:
                for p in param_group["params"]:
                    shard = self._get_param_shard(p)
                    handles.append(dist.all_gather_into_tensor(p.view(-1), shard, async_op=True))
        wait_on_handles(handles)

    def _shard_group(self, param_group):
        sharded_group = param_group.copy()
        sharded_group["params"] = [self._get_param_shard(p) for p in sharded_group["params"]]
        return sharded_group

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        self.optimizer.add_param_group(self._shard_group(param_group))
