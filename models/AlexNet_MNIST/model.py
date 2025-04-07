import torch
import torch.nn as nn


class AlexNet_class(nn.Module):
    def __init__(self, device="cpu", args: dict = None) -> None:
        print("AlexNet contructor called.")
        super(AlexNet_class, self).__init__()
        self.NUM_CLASSES = args["num_classes"]
        self.DEVICE = device

        # model implemented from https://github.com/l5shi/Image-Recognition-on-MNIST-dataset/blob/master/AlexNet.ipynb
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 12 * 12, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = x.view(-1, 32 * 12 * 12)
        x = self.classifier(x)
        return x
