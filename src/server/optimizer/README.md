All the optimizers are consolidated here
added few optimizers from https://pytorch.org/docs/stable/optim.html#algorithms

To add a new or custom optimizers, please follow the below naming convention of the file

    optimizer_<optimizer_name>.py
Kindly ensure the new optimizer has a method optimizer_selection(params,lr) that takes model_parameters, learning rate as input and returns the optimizer

To select an optimizer, please navigate to please navigate to config/[training_config.yaml](..%2F..%2Fconfig%2Ftraining_config.yaml)
and update 

    train_config:
        optimizer: optimizer_name