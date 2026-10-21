from collections import OrderedDict

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from cs336_basics.transformer import LinearLayer, EmbeddingLayer
MASTER_RANK = 0


def should_shard_layer(m: torch.nn.Module)->bool:
    return isinstance(m, LinearLayer) or isinstance(m, EmbeddingLayer)


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.module = module
        self.handles = list()
        self.is_order_recorded = False
        self.order = []
        self.base_layers = [m for m in module.modules() if len(list(m.children())) == 0]
        self.order_recording_hooks = [m.register_forward_pre_hook(self._record_order_hook, with_kwargs=True) for m in self.base_layers]
        self.sharded_layer_params =OrderedDict([m,  for m in self.base_layers if should_shard_layer(m)])

        for m in self.sharded_layers

    

    def _get_param_shard(self, p, rank=None):
        if rank is None:
            rank = self.rank
        flat = p.view(-1)
        l = flat.size(0)
        shard_size = l // self.world_size
        return flat[rank * shard_size : (rank + 1) * shard_size].detach.clone()


    def get_parameter_shard(param_tensor):



    def _record_order_hook(self, m, inputs, kwargs):
        self.order.append(m)
        return inputs, kwargs

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

    def pre_forward_hook(self, m, input):
        if not self.is_order_recorded:
            


    def post_forward_hook(self, m, input, output):
        pass



    def forward(self, *inputs, **kwargs):
        output = self.module(*inputs, **kwargs)
        if not self.is_order_recorded:
            self.is_order_recorded = True
            for hook in self.order_recording_hooks:
                hook.remove()

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
