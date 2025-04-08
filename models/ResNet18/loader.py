import torch
import math
import pickle

class CustomDataLoader:
    def __init__(self):
        print("Custom dataloader initialized.")

    def get_train_test_dataset_loaders(
        self, batch_size=50, dataset_path=None, args: dict = None
    ):  
        if "val_data" in dataset_path:
            print("LOADING GLOBAL DATASET")
            try:
                dataset = torch.load(dataset_path).dataset
            except Exception as e:
                print("ResNet18.CustomDataLoader :: Loading via torch failed; ", e)
        
            try:
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
            except Exception as e:
                print("ResNet18.CustomDataLoader :: Loading via pickle failed; ", e)
                
            train_loader = None
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last = True)
            
            print(
                "ResNet18:CustomDataloader.get_train_test_loader:: dataloader size - ",
                len(test_loader),
            )
        
        else:
            
            
            try:
                dataset = torch.load(dataset_path).dataset
            except Exception as e:
                print("ResNet18.CustomDataLoader :: Loading via torch failed; ", e)
            
            try:
                with open(dataset_path, 'rb') as f:
                    dataset = pickle.load(f)
            except Exception as e:
                print("ResNet18.CustomDataLoader :: Loading via pickle failed; ", e)
            
            dataset_len = len(dataset)

            split_idx = math.floor(0.95 * dataset_len)

            train_dataset = torch.utils.data.Subset(dataset, list(range(0, split_idx)))
            test_dataset = torch.utils.data.Subset(dataset, list(range(split_idx, dataset_len)))

            train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, batch_size=args['batch_size'], drop_last = True
            )
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, drop_last = True)

            print(
                "ResNet18:CustomDataloader.get_train_test_loader:: dataloader size - ",
                len(train_loader),
                len(test_loader),
            )

        return train_loader, test_loader
