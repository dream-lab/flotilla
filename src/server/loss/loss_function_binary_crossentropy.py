import torch.nn.modules.loss


def loss_function_selection():
    # Returns binary cross entropy loss
    return torch.nn.BCELoss
