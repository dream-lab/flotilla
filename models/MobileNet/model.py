"""MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
"""
import torch
from torchvision.models import MobileNetV2


class MobileNet(MobileNetV2):
    def __init__(self, device="cpu", args: dict = None) -> None:
        super().__init__(num_classes=args["num_classes"])


if __name__ == "__main__":
    net = MobileNet(num_classes=1000)
    print(net(torch.randn(3, 3, 224, 224)))
