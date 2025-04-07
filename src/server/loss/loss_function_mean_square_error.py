import torch.nn.modules.loss


def loss_function_selection():
    # Returns mean square error loss
    return torch.nn.MSELoss
