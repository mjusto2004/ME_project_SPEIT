
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 560)
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    model = MLP()

    # Test with 10 simulations
    x = torch.randn(10, 3)
    y = model(x)

    print(model)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
