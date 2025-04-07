"""LeNet-5 in PyTorch.

See the paper "Gradient-Based Learning Applied to Document Recognition" for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5_class(nn.Module):
    def __init__(self, device="cpu", args: dict = None) -> None:
        super(LeNet5_class, self).__init__()
        self.NUM_CLASSES = args["num_classes"]
        self.DEVICE = device
        print("LeNet5 constructor called.")

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(input=self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(input=self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(input=self.fc1(x))
        x = F.relu(input=self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    net = LeNet5_class(args={"num_classes": 10})
    print(net(torch.randn(1, 1, 32, 32)))
