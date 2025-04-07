import torch.nn.modules.loss


def loss_function_selection():
    # Return Mean absolute error loss = | actual - pred |
    return torch.nn.L1Loss
