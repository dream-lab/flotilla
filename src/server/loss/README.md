All the loss functions are consolidated here
added few loss functions from https://pytorch.org/docs/stable/nn.html#loss-functions

To add a new or custom loss function, please follow the below naming convention of the file

    loss_function_<loss_function>.py
Kindly ensure the new loss function has a method loss_function_selection() that returns the loss function

To select a loss function, please navigate to please navigate to config/[training_config.yaml](..%2F..%2Fconfig%2Ftraining_config.yaml)
and update 

    train_config:
        loss_function: loss_function