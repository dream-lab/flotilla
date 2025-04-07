import torch.optim.adadelta


def optimizer_selection(params, lr):
    return torch.optim.Adadelta(params=params, lr=lr)
