import torch.nn.modules.loss


def loss_function_selection():
    # Returns cross entropy loss
    return torch.nn.CrossEntropyLoss
