"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import math

import torch


class DataLoader:
    def __init__(self):
        pass

    def get_train_loader(self, batch_size=16, dataset_path=None):
        train_dataset = torch.load(dataset_path).dataset
        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )
        return train_loader

    def get_test_loader(self, batch_size=16, dataset_path=None):
        test_dataset = torch.load(dataset_path).dataset
        test_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, batch_size=batch_size
        )
        return test_loader

    def get_train_test_dataset_loaders(self, batch_size=16, dataset_path=None):
        dataset = torch.load(dataset_path).dataset

        dataset_len = len(dataset)

        split_idx = math.floor(0.95 * dataset_len)

        train_dataset = torch.utils.data.Subset(dataset, list(range(0, split_idx)))
        test_dataset = torch.utils.data.Subset(
            dataset, list(range(split_idx, dataset_len))
        )

        print(
            "client_dataset_loader.get_train_test_loader:: dataset size - ",
            len(train_dataset),
            len(test_dataset),
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True, batch_size=batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, shuffle=True, batch_size=batch_size
        )

        print(
            "client_dataset_loader.get_train_test_loader:: dataloader size - ",
            len(train_loader),
            len(test_loader),
        )

        return train_loader, test_loader
