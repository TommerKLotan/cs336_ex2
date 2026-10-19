import einx
import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print(f"first FF layer output type: {x.dtype}")
        x = self.ln(x)
        print(f"first layer norm output type: {x.dtype}")
        x = self.fc2(x)
        return x


def do_operation():
    dim = 100
    device = "cuda"
    model = ToyModel(dim, dim).to(device)
    print(f"model weight dtype: {model.fc1.weight.data.dtype}")
    input = torch.randn(dim, device=device)
    target = torch.randint(0, 2, (1,), device=device)
    logits = model(input)
    print(f"logits type: {logits.dtype}")
    logits = einx.id("d -> 1 d", logits)
    loss = nn.CrossEntropyLoss()(logits, target)
    print(f"loss type: {loss.dtype}")
    loss.backward()
    print(f"model grad dtype: {model.fc1.weight.grad.dtype}")


def main():
    do_operation()
    print(10 * "-")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        do_operation()


if __name__ == "__main__":
    main()
