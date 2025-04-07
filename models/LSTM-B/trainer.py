import torch
import time
import torch.optim as optim
from torch import nn
from tqdm import tqdm


class CustomModelTrainer:
    def __init__(self):
        print("LSTM-B trainer intiliazied")
        pass

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
        def get_batch(data, seq_len, idx):
            src = data[:, idx : idx + seq_len]
            target = data[:, idx + 1 : idx + seq_len + 1]
            return src, target

        def train_one_epoch(
            model, data, optimizer, criterion, batch_size, seq_len, clip, device
        ):
            epoch_loss = 0
            model.train()
            # drop all batches that are not a multiple of seq_len
            num_batches = data.shape[-1]
            data = data[:, : num_batches - (num_batches - 1) % seq_len]
            num_batches = data.shape[-1]

            hidden = model.init_hidden(batch_size, device)
            correct_predictions = 0
            total_num_samples = 0

            for idx in range(
                0, num_batches - 1, seq_len
            ):  # The last batch can't be a src
                optimizer.zero_grad()
                hidden = model.detach_hidden(hidden)

                src, target = get_batch(data, seq_len, idx)
                src, target = src.to(device), target.to(device)
                batch_size = src.shape[0]
                prediction, hidden = model(src, hidden)

                prediction = prediction.reshape(batch_size * seq_len, -1)
                target = target.reshape(-1)
                loss = criterion(prediction, target)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                epoch_loss += loss.item() * seq_len

                correct_predictions += sum(
                    torch.argmax(prediction, dim=1) == target
                ).item()
                total_num_samples += len(target)

            return {
                "loss": epoch_loss / num_batches,
                "accuracy": (correct_predictions / total_num_samples) * 100,
            }

        print("TRAINING MODEL")

        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device used to train = ", device)
        seq_len = args["seq_len"]
        clip = args["clip"]
        lr = args["lr"]
        batch_size = args["batch_size"]

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        loss = list()
        accuracy = list()

        batches_comepleted = 0
        for e in range(epochs):
            train_results = train_one_epoch(
                model=model,
                data=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                batch_size=batch_size,
                seq_len=seq_len,
                clip=clip,
                device=device,
            )
            loss.append(train_results["loss"])
            accuracy.append(train_results["accuracy"])
            print(f"Training round {e}", train_results)
            batches_comepleted += batch_size
            if time.time() - start_time > timeout_s:
                break

        results = {
            "total_mini_batches": batches_comepleted,
            "loss": sum(loss) / len(loss),
            "accuracy": (sum(accuracy) / len(accuracy)),
            "time_taken_s": time.time() - start_time,
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
        def get_batch(data, seq_len, num_batches, idx):
            src = data[:, idx : idx + seq_len]
            target = data[:, idx + 1 : idx + seq_len + 1]
            return src, target

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            seq_len = args["seq_len"]
            batch_size = args["batch_size"]
        except Exception:
            print("EXCEPTION IN ARGS")
        epoch_loss = 0
        model.eval()
        num_batches = dataloader.shape[-1]
        data = dataloader[:, : num_batches - (num_batches - 1) % seq_len]
        num_batches = data.shape[-1]
        criterion = nn.CrossEntropyLoss()

        hidden = model.init_hidden(batch_size, device)
        correct_predictions = 0
        total_num_samples = 0

        with torch.no_grad():
            print("VALIDATING")
            for idx in range(0, num_batches - 1, seq_len):
                hidden = model.detach_hidden(hidden)
                src, target = get_batch(dataloader, seq_len, num_batches, idx)
                src, target = src.to(device), target.to(device)
                batch_size = src.shape[0]

                prediction, hidden = model(src, hidden)
                prediction = prediction.reshape(batch_size * seq_len, -1)
                target = target.reshape(-1)

                loss = criterion(prediction, target)
                epoch_loss += loss.item() * seq_len
                correct_predictions += sum(
                    torch.argmax(prediction, dim=1) == target
                ).item()
                total_num_samples += len(target)

        results = {
            "loss": epoch_loss / num_batches,
            "accuracy": (correct_predictions / total_num_samples) * 100,
        }
        print("VALIDATION RESULTS:", results)
        return results
