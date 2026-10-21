import torch
import torch.distributed as dist

MASTER_RANK = 0


def wait_on_handles(handles):
    for handle in handles:
        handle.wait()


class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        handles = [
            dist.broadcast(param.data, MASTER_RANK, async_op=True)
            for param_name, param in module.named_parameters()
            if "no_grad_fixed_param" not in param_name
        ]
        wait_on_handles(handles)

    def forward(self, data):
        output = self.module(data)
        return output


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    handles = [
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, async_op=True)
        for param_name, param in ddp_model.module.named_parameters()
        if "no_grad_fixed_param" not in param_name and param.grad is not None
    ]
    wait_on_handles(handles)
