import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


class CustomModelTrainer:
    def __init__(self) -> None:
        pass

    def fedprox_regularizer(self, current_model, global_model, mu=0):
        res = 0.0
        for layer in global_model.keys():
            diff = torch.sub(current_model[layer], global_model[layer])
            res += torch.linalg.norm(diff)

        return torch.mul(mu / 2, res).item()

    # (model, train_loader, test_loader: None, epochs, criterion, optimizer, args:dict)
    def train_model(
        self,
        model,
        results,
        train_loader,
        epochs,
        timeout_s: float,
        loss_func=None,
        optimizer=None,
        device=torch.device("cpu"),
        test_loader=None,
        args: dict = None,
        start_time: float = None,
    ):
        pass

        mu = args["mu"]
        # Setting the loss function
        if loss_func is None:
            print("Loss was none, using default Loss Function")
            loss_func = torch.nn.CrossEntropyLoss
        cost = loss_func()

        # Setting the optimizer with the model parameters and learning rate
        if optimizer is None:
            print("Optimizer was none, using default Optimizer")
            optimizer = torch.optim.Adam(params=model.parameters(), lr=args["lr"])

        for param_group in optimizer.param_groups:
            print("Optimizer learning rate = ", param_group["lr"])

        # update optimizer with current model parameters.
        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({"params": [p for p in model.parameters()]})

        # setting model to train mode
        model.train()
        original_global_model_wts = model.state_dict()

        total_num_mini_batches = 0

        start_time = time.time()

        exit_flag = False
        total_loss = 0
        total = 0
        avg_loss = 0
        correct = 0
        total_accuracy = 0
        batch_size = 0
        float_epochs = 0.0

        try:
            for epoch in range(epochs):
                num_mini_batches = 0
                # for idx, (train_x, train_label) in tqdm(
                #     enumerate(train_loader),
                #     total=len(train_loader),
                #     desc="Mini Batches",
                # ):

                for idx, (train_x, train_label) in enumerate(train_loader):
                    # print(f"Epoch {epoch}/ minibatch {idx}")
                    batch_size = len(train_x)
                    data_entries = len(train_loader)

                    train_x = train_x.to(device)
                    train_label = train_label.to(device)
                    optimizer.zero_grad()
                    predict_y = model(train_x)

                    loss = cost(predict_y, train_label) + self.fedprox_regularizer(
                        model.state_dict(), original_global_model_wts, mu
                    )

                    loss.backward()

                    optimizer.step()

                    total += len(train_x)
                    total_loss += loss.item()

                    current_correct = (
                        (torch.argmax(predict_y, 1) == train_label).float().sum()
                    ).item()

                    current_loss = round(loss.item(), 3)
                    correct += current_correct
                    avg_loss = round(total_loss / (total_num_mini_batches + 1), 3)
                    total_accuracy = round((correct / total) * 100, 3)
                    current_accuracy = round((current_correct / len(train_x)) * 100, 3)
                    epochs = epoch

                    num_mini_batches += 1
                    total_num_mini_batches += 1
                    float_epochs = epoch + (num_mini_batches / data_entries)

                    if time.time() - start_time > timeout_s:
                        print("TRAINER TIMEOUT!!")
                        exit_flag = True
                        break

                print(
                    f"epochs={float_epochs}, avg_loss={avg_loss}, total_accuracy={total_accuracy}"
                )
                if exit_flag:
                    break
        except Exception as e:
            print(
                "FedAT_CNN.CustomTrainer.train_model exception in training loop = ", e
            )

        print(
            f"epochs, {float_epochs}, avg_loss ,{avg_loss}, total_accuracy, {total_accuracy}"
        )
        print(f"Training Round Finished {time.time() - start_time}sec")
        results = {
            "time_taken_s": (time.time() - start_time),
            "num_epochs": float_epochs,
            "total_mini_batches": total_num_mini_batches,
            "loss": avg_loss,
            "accuracy": total_accuracy,
        }
        return results

    def validate_model(
        self,
        model,
        dataloader,
        device: str = "cpu",
        loss_func=None,
        optimizer=None,
        round_no=None,
        args: dict = None,
    ):
        model.eval()
        model.to(device)
        acc = 0
        count = 0
        total_loss = 0
        batches = 0

        if loss_func is None:
            print("Loss was none, using default Loss Function")
            loss_func = torch.nn.CrossEntropyLoss

        with torch.no_grad():
            cost = loss_func()
            for i, (x_batch, y_batch) in enumerate(dataloader):
                # if i >= 1:
                #     break
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                y_pred = self.model(x_batch)
                loss = cost(y_pred, y_batch)
                total_loss += loss.item()
                acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()
                count += len(y_batch)
                batches = i + 1

        acc = (acc / count) * 100
        loss = total_loss / batches

        model.train()
        res = {"accuracy": acc, "loss": loss}
        print("Result of validation : res = ", res)
        return res
