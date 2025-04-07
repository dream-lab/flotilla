import torch
import torch.nn as nn
import torch.nn.functional as F


class FedAT_CNN(nn.Module):
    def __init__(self, device="cpu", args: dict = None):
        super(FedAT_CNN, self).__init__()
        self.NUM_CLASSES = args["num_classes"]
        self.DEVICE = device

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(
            in_features=64 * 4 * 4, out_features=64
        )  # input image size has to be 32x32
        self.fc2 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # print("shape of input: ", x.shape)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print("shape after 1 conv: ", x.shape)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # print("shape after 2 conv: ", x.shape)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        # print("shape after 3 conv: ", x.shape)
        # x = x.view(-1, 64 * 8 * 8)  # input image size has to be 32x32
        x = torch.flatten(x, 1)
        # print("shape after flatten: ", x.shape)
        x = F.relu(self.fc1(x))
        # print("shape after 1 linear: ", x.shape)
        x = self.fc2(x)
        # print("shape after 2 linear: ", x.shape)

        return x
