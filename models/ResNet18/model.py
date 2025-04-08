import torch
from torchvision.models.resnet import ResNet, BasicBlock

## model with pretraining
# class ResNet18_Model(ResNet):
#     def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), args: dict = None) -> None:
#         super().__init__(BasicBlock, [2,2,2,2], num_classes=args["num_classes"])
#         # self.load_state_dict(torch.load("resnet18-f37072fd.pth"))
#         self.load_state_dict(torch.load('resnet18-f37072fd.pth'))

class ResNet18_Model(ResNet):
    @staticmethod
    def append_dropout(model, rate=0.1):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                ResNet18_Model.append_dropout(model=module, rate=rate)
            if isinstance(module, torch.nn.ReLU):
                new = torch.nn.Sequential(module, torch.nn.Dropout2d(p=rate))
                setattr(model, name, new)

    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), args: dict = None) -> None:
        super().__init__(BasicBlock, [2,2,2,2], num_classes=args["num_classes"])
        ResNet18_Model.append_dropout(model=self, rate=args['dropout_rate'])
        # self.load_state_dict(torch.load("/home/fedml/fedml-ng/models/ResNet18/resnet18-f37072fd.pth"))


#model = ResNet18_Model(args={'num_classes': 200, 'dropout_rate': 0.1})
#print(len(list(model.children())))
