import torch.nn.modules.loss


def loss_function_selection():
    # Returns K L Divergance loss
    return torch.nn.KLDivLoss
