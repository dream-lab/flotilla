import torch.optim.adagrad


def optimizer_selection(params, lr):
    return torch.optim.Adagrad(params=params, lr=lr)
