"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import time

import torch
from tqdm import tqdm

from client.client_file_manager import get_model_class


class ClientTrainer:
    def __init__(
        self,
        temp_dir_path: str,
        model_id: str,
        model_class: str,
        loss_fn=None,
        optimizer=None,
        device: str = "cpu",
        use_custom_trainer=False,
        model_args: dict = None,
        custom_trainer_args: dict = None,
        use_custom_validator=False,
        custom_validator_args: dict = None,
    ) -> None:
        self.device = torch.device(device)

        self.stop_training_flag = False
        self.loss_func = loss_fn
        self.optimizer = optimizer

        self.temp_dir_path = temp_dir_path
        self.model_id = model_id

        self.model = get_model_class(self.temp_dir_path, self.model_id, model_class)(
            self.device, args=model_args
        )

        self.custom_trainer_args = None
        self.use_custom_trainer = use_custom_trainer
        if use_custom_trainer:
            self.custom_trainer_args = custom_trainer_args

        self.use_custom_validator = use_custom_validator
        if use_custom_validator:
            self.custom_validator_args = custom_validator_args

    def set_loss_function(self, loss_func) -> None:
        self.loss_func = loss_func

    def set_device(self, device: str) -> None:
        self.device = device

    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer

    def load_model_from_checkpoint(self, checkpoint) -> None:
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

    def get_model_wts(self):
        return self.model.to("cpu").state_dict()

    def stop_training(self):
        self.stop_training_flag = True

    def exit_check(
        self,
        epochs,
        max_epochs,
        max_mini_batches,
        num_mini_batches,
        start_time,
        timeout_duration_s,
    ):
        exit_flag = False
        if max_mini_batches and (num_mini_batches >= max_mini_batches):
            exit_flag = True
            return exit_flag
        elif max_epochs and (epochs >= max_epochs):
            exit_flag = True
            return exit_flag
        elif timeout_duration_s and (time.time() - start_time > timeout_duration_s):
            exit_flag = True
            return exit_flag

        return exit_flag

    def default_train_model_classifier(
        self,
        lr: float,
        train_loader,
        test_loader=None,
        num_epochs=None,
        timeout_duration_s=None,
        max_mini_batches=None,
        max_epochs=None,
    ):
        # Setting the loss function
        if not self.loss_func:
            print("Loss was none, using default Loss Function")
            self.set_loss_function(torch.nn.CrossEntropyLoss)
        cost = self.loss_func()

        # Setting the optimizer with the model parameters and learning rate
        if self.optimizer is None:
            print("Optimizer was none, using default Optimizer")
            self.set_optimizer(torch.optim.Adam(params=self.model.parameters(), lr=lr))

        for param_group in self.optimizer.param_groups:
            print("Optimizer learning rate = ", param_group["lr"])

        # update optimizer with current model parameters.
        self.optimizer.param_groups.clear()
        self.optimizer.state.clear()
        self.optimizer.add_param_group({"params": [p for p in self.model.parameters()]})

        # setting model to train mode
        self.model.train()

        total_num_mini_batches = 0

        start_time = time.time()

        exit_flag = False
        total_loss = 0
        total = 0
        avg_loss = 0
        correct = 0
        epochs = 0
        total_accuracy = 0
        batch_size = 0
        float_epochs = 0.0
        for epoch in range(num_epochs):
            num_mini_batches = 0
            for idx, (train_x, train_label) in tqdm(
                enumerate(train_loader), total=len(train_loader), desc="Mini Batches"
            ):
                batch_size = len(train_x)
                data_entries = len(train_loader)
                exit_flag = self.exit_check(
                    epochs,
                    max_epochs,
                    max_mini_batches,
                    num_mini_batches,
                    start_time,
                    timeout_duration_s,
                )
                if exit_flag:
                    break

                train_x = train_x.to(self.device)
                train_label = train_label.to(self.device)
                self.optimizer.zero_grad()
                predict_y = self.model(train_x)

                loss = cost(predict_y, train_label)
                loss.backward()

                self.optimizer.step()

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

                exit_flag = self.exit_check(
                    epochs,
                    max_epochs,
                    max_mini_batches,
                    num_mini_batches,
                    start_time,
                    timeout_duration_s,
                )
                if exit_flag:
                    break

                if self.stop_training_flag:
                    return {
                        "run_time": (time.time() - start_time),
                        "num_epochs": float_epochs,
                        "total_num_minibatches": total_num_mini_batches,
                        "loss": avg_loss,
                        "accuracy": total_accuracy,
                    }
            if exit_flag:
                break
        print(
            f"epochs, {float_epochs}, avg_loss ,{avg_loss}, total_accuracy, {total_accuracy}"
        )
        print(f"Training Round Finished {time.time() - start_time}sec")
        return {
            "time_taken_s": (time.time() - start_time),
            "num_epochs": float_epochs,
            "total_mini_batches": total_num_mini_batches,
            "loss": avg_loss,
            "accuracy": total_accuracy,
        }

    def train_model(
        self,
        lr: float,
        train_loader,
        test_loader=None,
        num_epochs=None,
        timeout_duration_s=None,
        max_mini_batches=None,
        max_epochs=None,
        model_checkpoint=None,
    ):
        # loading model from checkpoint
        if model_checkpoint:
            self.load_model_from_checkpoint(checkpoint=model_checkpoint)

        # Setting the loss function
        # if self.loss_func is None:
        #     print("client_trainer.ClientTrainer.train_model :: WARNING - Loss was none")
        #     self.set_loss_function(torch.nn.CrossEntropyLoss)

        # cost = self.loss_func()

        # # Setting the optimizer with the model parameters and learning rate
        # if self.optimizer is None:
        #     print(
        #         "client_trainer.ClientTrainer.train_model :: WARNING - Optimizer was none"
        #     )
        #     self.set_optimizer(torch.optim.Adam(params=self.model.parameters(), lr=lr))

        # # update optimizer with current model parameters.
        # self.optimizer.param_groups.clear()
        # self.optimizer.state.clear()
        # self.optimizer.add_param_group({"params": [p for p in self.model.parameters()]})

        # setting model to train mode
        # self.model.train()

        results = dict()

        if self.use_custom_trainer:
            print("CLIIENT_TRAINER.train_model:: Using custom trainer")
            trainer = get_model_class(
                path=self.temp_dir_path,
                model_id=self.model_id,
                class_name="CustomModelTrainer",
            )()

            results = trainer.train_model(
                model=self.model,
                results=results,
                train_loader=train_loader,
                epochs=num_epochs,
                test_loader=test_loader,
                args=self.custom_trainer_args,
                timeout_s=timeout_duration_s,
            )
            print(f"CLIENT_TRAINER.train_model:: Results - {results}")
        else:
            print("CLIIENT_TRAINER.train_model:: Using default trainer")
            results = self.default_train_model_classifier(
                lr=lr,
                train_loader=train_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                timeout_duration_s=timeout_duration_s,
                max_mini_batches=max_mini_batches,
                max_epochs=max_epochs,
            )
            print(f"CLIENT_TRAINER.train_model:: Results - {results}")
        return results

    def validate_model(
        self,
        test_loader,
        model_checkpoint=None,
    ):
        print("CLIENT_TRAINER.validate_model() called!")
        # loading model from checkpoint
        if model_checkpoint:
            self.load_model_from_checkpoint(checkpoint=model_checkpoint)

            # Setting the loss function
        if self.loss_func is None:
            print("client_trainer.ClientTrainer.train_model :: WARNING - Loss was none")
            self.set_loss_function(torch.nn.CrossEntropyLoss)

        cost = self.loss_func()

        # Setting the optimizer with the model parameters and learning rate
        if self.optimizer is None:
            print(
                "client_trainer.ClientTrainer.train_model :: WARNING - Optimizer was none"
            )
            self.set_optimizer(torch.optim.Adam(params=self.model.parameters(), lr=lr))

        if self.use_custom_validator:
            print("CLIENT_TRAINER.validate_model:: Custom validator being used.")
            validator = get_model_class(
                path=self.temp_dir_path,
                model_id=self.model_id,
                class_name="CustomModelTrainer",
            )

            res = validator.validate_model(
                self,
                model=self.model,
                dataloader=test_loader,
                device=self.device,
                loss_func=self.loss_func,
                optimizer=self.optimizer,
                args=self.custom_validator_args,
            )

        else:
            print("CLIENT_TRAINER.validate_model:: Default validator being used.")
            self.model = self.model.to(self.device)

            self.model.eval()

            acc = 0
            count = 0
            total_loss = 0
            batches = 0
            with torch.no_grad():
                cost = self.loss_func()
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    # if i >= 1:
                    #     break
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    y_pred = self.model(x_batch)
                    loss = cost(y_pred, y_batch)
                    total_loss += loss.item()
                    acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()
                    count += len(y_batch)
                    batches = i + 1

            acc = (acc / count) * 100
            loss = total_loss / batches

            self.model.train()
            res = {"accuracy": acc, "loss": loss}
        print("Result of validation : res = ", res)
        return res
