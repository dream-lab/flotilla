import torch.optim.sgd


def optimizer_selection(params, lr):
    return torch.optim.SGD(params=params, lr=lr)
