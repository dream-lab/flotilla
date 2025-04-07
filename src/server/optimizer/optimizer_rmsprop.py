import torch.optim.rmsprop


def optimizer_selection(params, lr):
    return torch.optim.RMSprop(params=params, lr=lr)
