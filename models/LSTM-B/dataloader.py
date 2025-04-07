import torch


class CustomDataLoader:
    def __init__(self) -> None:
        pass

    def get_train_test_dataset_loaders(
        self, batch_size=1, dataset_path=None, args: dict = None
    ):
        test_data_path = dataset_path.replace("train", "validation")

        print("Test dataset path:", test_data_path)

        train_data = torch.load(dataset_path)
        test_data = torch.load(test_data_path)

        return train_data, test_data
