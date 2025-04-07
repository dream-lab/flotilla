import torch.optim.adam


def optimizer_selection(params, lr):
    return torch.optim.Adam(params=params, lr=lr)
