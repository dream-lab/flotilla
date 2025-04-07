"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import torch
from tqdm import tqdm

from server.load_loss import load_loss
from server.load_optimizer import load_optimizer
from server.server_file_manager import get_model_class
from utils.logger import FedLogger


class ServerModelManager:
    def __init__(
        self,
        id,
        model_dir,
        model_class,
        batch_size,
        val_data_path,
        torch_device=torch.device("cpu"),
        model_args: dict = None,
        use_custom_dataloader=False,
        custom_dataloader_args: dict = None,
        use_custom_validator=False,
        custom_validator_args=None,
    ) -> None:
        self.id = id
        self.torch_device = torch_device
        self.model_dir = model_dir

        torch.manual_seed(1122001)
        self.model = get_model_class(path=model_dir, class_name=model_class)(
            device=torch_device, args=model_args
        )

        self.logger = FedLogger(id=self.id, loggername="SERVER_MODEL_MANAGER")

        if use_custom_dataloader:
            DataLoader = get_model_class(
                path=model_dir, class_name="CustomDataLoader"
            )()
            _, self.data = DataLoader.get_train_test_dataset_loaders(
                batch_size=batch_size,
                dataset_path=val_data_path,
                args=custom_dataloader_args,
            )
        else:
            self.data = self.test_dataset_loader(
                path=val_data_path, batch_size=batch_size
            )

        self.use_custom_validator = use_custom_validator
        self.custom_validator_args = custom_validator_args

    def get_model_weights(self):
        self.model.to("cpu")
        return self.model.state_dict()

    def set_model_weights(self, model_weights):
        self.model.load_state_dict(model_weights)

    def get_model_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_optimizer(self):
        return self.optimizer

    # TODO add check if None
    def set_optimizer(self, lr, optimizer, custom):
        opt = load_optimizer(self.id, optimizer, custom)
        self.optimizer = opt.optimizer_selection(self.model.parameters(), lr=lr)

    # TODO add check if None
    def set_loss_fun(self, loss_fun, custom):
        loss = load_loss(self.id, loss_fun, custom)
        self.loss_fun = loss.loss_function_selection()

    def get_loss_fun(self):
        return self.loss_fun

    def test_dataset_loader(self, path: str, batch_size=50):
        test_dataset = torch.load(path).dataset
        print("Length of test dataset", len(test_dataset))

        data = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=True
        )
        return data

    def validate_model(
        self,
        device: str = "cpu",
        loss_func=None,
        optimizer=None,
        round_no=None,
    ):
        if self.use_custom_validator:
            validator = get_model_class(
                path=self.model_dir, class_name="CustomModelTrainer"
            )
            res = validator.validate_model(
                self,
                model=self.model,
                dataloader=self.data,
                device=device,
                loss_func=loss_func,
                optimizer=optimizer,
                round_no=round_no,
                args=self.custom_validator_args,
            )

        else:
            self.model = self.model.to(self.torch_device)

            self.model.eval()

            acc = 0
            count = 0
            total_loss = 0
            batches = 0
            with torch.no_grad():
                cost = self.loss_fun()
                for i, (x_batch, y_batch) in tqdm(
                    enumerate(self.data), total=len(self.data), desc="Validation Round"
                ):
                    x_batch = x_batch.to(self.torch_device)
                    y_batch = y_batch.to(self.torch_device)
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
            print(res)

        return res
