import time
import torch
from tqdm import tqdm
from  torch.cuda.amp import autocast

class CustomModelTrainer:
    def __init__(self) -> None:
        print("ResNet18 trainer intialized!")
        pass

    def train_model(
        self,
        model,
        results,
        train_loader,
        epochs,
        round_no,
        timeout_s: float,
        loss_func=None,
        optimizer=None,
        device=torch.device("cuda"),
        test_loader=None,
        args: dict = None,
        start_time: float = None,
    ):
        print("ResNet18 trainer called!")
        try:
            # Setting the loss function
            cost = torch.nn.CrossEntropyLoss()

            # Setting the optimizer with the model parameters and learning rate
            # optimizer = torch.optim.Adam(params=model.parameters(), lr=args["lr"], weight_decay=args['weight_decay'])
            # optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])
            optimizer = torch.optim.SGD(model.parameters(), lr=args["lr"], momentum=args['momentum'])

            for param_group in optimizer.param_groups:
                print("OPTIMIZER LEARNING RATE = ", param_group["lr"])

            # update optimizer with current model parameters.
            optimizer.param_groups.clear()
            optimizer.state.clear()
            optimizer.add_param_group({"params": [p for p in model.parameters()]})

            # setting model to train mode
            model.train()
            model.to(device)
            print("Model is training on device = ", device)

            total_num_mini_batches = 0

            start_time = time.time()

            exit_flag = False
            total_loss = 0
            total = 0
            avg_loss = 0
            correct = 0
            total_accuracy = 0
            float_epochs = 0.0

            for epoch in range(epochs):
                num_mini_batches = 0
                for (train_x, train_label) in tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc="Mini Batches",
                ):
                    data_entries = len(train_loader)

                    train_x = train_x.to(device).squeeze()
                    train_label = train_label.to(device).squeeze()
                    optimizer.zero_grad()
                    
                    with autocast():
                        predict_y = model(train_x)
                        loss = cost(predict_y, train_label)

                    loss.backward()

                    optimizer.step()

                    total += len(train_x)
                    total_loss += loss.item()

                    current_correct = (
                        (torch.argmax(predict_y, 1) == train_label).float().sum()
                    ).item()

                    correct += current_correct
                    avg_loss = round(total_loss / (total_num_mini_batches + 1), 3)
                    total_accuracy = round((correct / total) * 100, 3)
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
                "ResNet18.CustomTrainer.train_model exception in training loop = ", e
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
        res = self.validate_model(model,test_loader,device,cost)
        results.update(res)
        return results

    def validate_model(
        self,
        model,
        dataloader,
        device = torch.device("cuda"),
        loss_func=None,
        optimizer=None,
        round_no=None,
        args: dict = None,
    ):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.eval()
        model.to(device)
        print("Model is validating on device = ", device)
        acc = 0
        count = 0
        total_loss = 0
        batches = 0

        try:
            loss_func = torch.nn.CrossEntropyLoss

            with torch.no_grad():
                cost = loss_func()
                for i, (x_batch, y_batch) in enumerate(dataloader):
                    # if i >= 1:
                    #     break
                    x_batch = x_batch.to(device).squeeze()
                    y_batch = y_batch.to(device).squeeze()

                    with autocast():
                        y_pred = model(x_batch)
                        loss = cost(y_pred, y_batch)
                    
                    total_loss += loss.item()
                    acc += (torch.argmax(y_pred, 1) == y_batch).float().sum().item()
                    count += len(y_batch)
                    batches = i + 1

            acc = (acc / count) * 100
            loss = total_loss / batches
        
        except Exception as e:
            print(
                "ResNet18.CustomTrainer.validate_model exception = ", e
            )

        res = {"val_accuracy": acc, "val_loss": loss}
        print("Result of validation : res = ", res)
        return res
