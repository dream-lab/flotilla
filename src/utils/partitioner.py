import argparse
import math
import os.path
import yaml
import torch
import numpy as np
import scipy
from collections import Counter
import torchvision.transforms as transforms
import torchvision
from sklearn.model_selection import train_test_split


def random_seed(seed=100):
    np.random.seed(seed)


def get_dataset_summary(dataloader):
    try:
        dataset = dataloader.dataset
        num_items = len(dataset)
        class_distrb = dict(Counter(y for _, (x, y) in enumerate(dataset)))
        for i, v in class_distrb.items():
            class_distrb[i] = v / num_items
        data_summary = {"label_distribution": class_distrb, "num_items": num_items}

        return data_summary

    except Exception as e:
        print("get_data_summary:: Exception - ", e)


def dirchlet(
    dataset,
    num_clients: int,
    alpha: float = 0.1,
    dataset_name: str = "MNIST",
    path: str = "./data/",
    min_samples: int = 100,
    task: str = "train",
):
    # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    data = dataset.dataset
    dataset_content, dataset_label = (data.data, data.targets)
    label_distribution = dict(Counter([y for _, y in data]))
    print(label_distribution)
    num_classes = len(label_distribution)
    dataidx_map = {}
    min_size = 0
    K = num_classes
    N = len(dataset_label)

    while min_size < min_samples:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < N / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "dirichlet/")
    os.makedirs(path, exist_ok=True)
    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]

        temp = os.path.join(path, f"part_{client}/")
        os.makedirs(temp, exist_ok=True)
        filename = f"a_{alpha}_part_{client}.pth"

        save_partition(client, data, dataset_name, filename, idxs, temp, task)

        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

    print("Created dirichlet partitions!!")

    del data

def limit_label(dataset,
                max_classes,
                num_client,
                f: float = 1.0,
                path="./data/",
                dataset_name: str = "MNIST",
                task: str = "train"):
    client_idx = []

    data = dataset.dataset
    label_distribution = dict(Counter([y for _, y in data]))
    print(label_distribution)
    num_classes = len(label_distribution)

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "limit_label/")
    os.makedirs(path, exist_ok=True)

    classes = dict()
    for i, (x, y) in enumerate(data):
        if y in classes.keys():
            classes[y].append(i)
        else:
            classes[y] = [i]
    print(classes.keys())
    for client in range(num_client):
        selected_labels = np.random.choice(num_classes,
                                           size=min(max_classes, num_classes),
                                           replace=False)
        print("selected labels: ", selected_labels)

        idx = []
        for x in selected_labels:
            fraction = np.random.choice(classes[x],
                                        size=min(math.ceil(f * len(classes[x])), len(classes[x])),
                                        replace=False)
            idx.extend(fraction)

        client_idx.append(idx)
        temp = os.path.join(path, f"part_{client}/")
        os.makedirs(temp, exist_ok=True)
        filename = f"ll_part_{client}.pth"

        # save the partitions
        save_partition(client, data, dataset_name, filename, idx, temp, task)

    print("Created limit_label partitions!!")

def limit_label_unique(dataset,
                max_classes,
                num_client,
                samples,
                path="./data/",
                dataset_name: str = "MNIST",
                task: str = "train"):
    client_idx = []

    data = dataset.dataset
    label_distribution = dict(Counter([y for _, y in data]))
    print(label_distribution)
    num_classes = len(label_distribution)

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "limit_label_samples/")
    os.makedirs(path, exist_ok=True)

    classes = dict()
    for i, (x, y) in enumerate(data):
        if y in classes.keys():
            classes[y].append(i)
        else:
            classes[y] = [i]
    for client in range(num_client):
        selected_labels = np.random.choice(num_classes,
                                           size=min(max_classes, num_classes),
                                           replace=False)
        print("selected labels: ", selected_labels)
        # need to handle if the samples in class is empty
        idx = []
        for x in selected_labels:
            fraction = np.random.choice(classes[x],
                                        size=min(samples, len(classes[x])),
                                        replace=False)
            idx.extend(fraction)
            for j in fraction:
                classes[x].remove(j)

        client_idx.append(idx)
        temp = os.path.join(path, f"part_{client}/")
        os.makedirs(temp, exist_ok=True)
        filename = f"ll_part_{client}.pth"

        # save the partitions
        save_partition(client, data, dataset_name, filename, idx, temp, task)

    print("Created limit_label partitions!!")


def equal_label_distribution(dataset,
                             path="./data/",
                             dataset_name: str = "MNIST",
                             task: str = "train"):
    data = dataset.dataset
    label_distribution = dict(Counter([y for _, y in data]))
    print(label_distribution)
    num_classes = len(label_distribution)

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "eq_part_complete/")
    os.makedirs(path, exist_ok=True)

    classes = dict()

    for i, (x, y) in enumerate(data):
        if y in classes.keys():
            classes[y].append(i)
        else:
            classes[y] = [i]

    least = min(label_distribution.values())
    print(f"least is {least}")
    idx = []

    for key in classes.keys():
        for i in classes[key][:least]:
            idx.append(i)

    # temp = os.path.join(path, f'part_{part}/')
    # os.makedirs(temp, exist_ok=True)

    filename = f"equal_labels.pth"
    save_partition(1, data, dataset_name, filename, idx, path, task)


def equal_partition(dataset,
                    num_client,
                    path="./data/",
                    dataset_name: str = "MNIST",
                    task: str = "train"):
    data = dataset.dataset
    label_distribution = dict(Counter([y for _, y in data]))

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "iid/")
    os.makedirs(path, exist_ok=True)

    classes = dict()

    for i, (x, y) in enumerate(data):
        if y in classes.keys():
            classes[y].append(i)
        else:
            classes[y] = [i]

    least = min(label_distribution.values())
    min_samples = least // num_client
    print(f"least label wise samples: {least}")
    for i in range(num_client):
        part_idx = list()
        print(f"partition - {i}")
        for k in classes.keys():
            for j in range(i * min_samples, (i + 1) * min_samples):
                part_idx.append(classes[k][j])

        temp = os.path.join(path, f"part_{i}/")
        os.makedirs(temp, exist_ok=True)
        filename = f"iid_part_{i}.pth"
        save_partition(i, data, dataset_name, filename, part_idx, temp, task)

    print("Generated partitions!!!")

def equal_samples(dataset,
                  num_client,
                  num_samples,
                  path="./data/",
                  dataset_name: str = "MNIST",
                  task: str = "train"):
    data = dataset.dataset
    label_distribution = dict(Counter([y for _, y in data]))

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "iid_eq_samples/")
    os.makedirs(path, exist_ok=True)

    classes = dict()

    for i, (x, y) in enumerate(data):
        if y in classes.keys():
            classes[y].append(i)
        else:
            classes[y] = [i]

    samples_per_class = num_samples // len(classes.keys())


    for part in range(num_client):
        idx = []
        for k in classes.keys():
            temp = np.random.choice(classes[k],
                                    size=min(samples_per_class, len(classes[k])),
                                    replace=False)
            # or
            # end = len(classes[k]) if ((part+1)*samples_per_class > len(classes[k])) else (part+1)*samples_per_class
            # temp = classes[k][part*samples_per_class:end]
            idx.extend(temp)
        temp = os.path.join(path, f"part_{part}/")
        os.makedirs(temp, exist_ok=True)
        filename = f"iid_part_{part}.pth"
        save_partition(part, data, dataset_name, filename, idx, temp, task)

    print("Generated partitions!!!")

def probability_distribution(dataset,
                             distribution: list,
                             path: str = "./data/",
                             dataset_name: str = "MNIST",
                             task: str = "train"):
    data = dataset.dataset
    label_distribution = dict(Counter([y for _, y in data]))
    print(label_distribution)

    path = os.path.join(path, dataset_name,task)
    path = os.path.join(path, "probability/")
    os.makedirs(path, exist_ok=True)

    classes = dict()

    for i, (x, y) in enumerate(data):
        if y in classes.keys():
            classes[y].append(i)
        else:
            classes[y] = [i]

    slices_indicies = dict()
    for key in classes.keys():
        slices_indicies[key] = 0
    client_idxs = []
    part = 0
    for probability in distribution:
        idx = []
        for key in classes.keys():
            temp_num = math.ceil(probability * len(classes[key]))
            start = slices_indicies[key]
            end = slices_indicies[key] + temp_num
            if end > len(classes[key]):
                end = len(classes[key])
            for i in range(start, end):
                idx.append(classes[key][i])
                pass
            slices_indicies[key] = end

        client_idxs.append(idx)

        temp = os.path.join(path, f"part_{part}/")
        os.makedirs(temp, exist_ok=True)

        filename = f"probability_{dataset_name}_part_{part}.pth"

        save_partition(part, data, dataset_name, filename, idx, temp, task)
        part += 1


def save_partition(client, data, dataset_name, filename, idxs, temp, task):
    partition = torch.utils.data.Subset(data, idxs)
    partition_loader = torch.utils.data.DataLoader(
        partition, batch_size=1, shuffle=False
    )
    torch.save(partition_loader, os.path.join(temp, filename))
    summary = get_dataset_summary(partition_loader)
    config = dict()
    config["dataset_details"] = {
        "data_filename": filename,
        "dataset_id": dataset_name,
        "dataset_tags": ["IMAGE"],
        "suitable_models": ["LeNet5", "AlexNet"],
    }
    config["metadata"] = summary
    yaml_file = f"{task}_dataset_config.yaml"
    with open(os.path.join(temp, yaml_file), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

def save_test_partition(data, dataset_name, filename="test.pth", task="test"):
    partition = data.dataset
    partition_loader = torch.utils.data.DataLoader(
        partition, batch_size=1, shuffle=False
    )
    path = os.path.join('./data',dataset_name, task)
    os.makedirs(path, exist_ok=True)
    torch.save(partition_loader, os.path.join(path, filename))
    summary = get_dataset_summary(partition_loader)
    config = dict()
    config["dataset_details"] = {
        "data_filename": filename,
        "dataset_id": dataset_name,
        "dataset_tags": ["IMAGE"],
        "suitable_models": ["LeNet5", "AlexNet"],
    }
    config["metadata"] = summary
    yaml_file = f"{task}_dataset_config.yaml"
    with open(os.path.join(path, yaml_file), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_mnist():
    # Get MNIST data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )

    trainset = torchvision.datasets.MNIST(
        root="./data/", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root="./data/", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )

    return [trainloader, testloader]


def get_emnist(split: str):
    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    trainset = torchvision.datasets.EMNIST(
        root="./data/", train=True, download=True, transform=transform, split=split
    )
    testset = torchvision.datasets.EMNIST(
        root="./data/", train=False, download=True, transform=transform, split=split
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False
    )

    return [trainloader, testloader]


def get_cifar10():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([224, 224], interpolation=0),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    return [trainloader, testloader]


def get_cifar100():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize([224, 224], interpolation=0),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)

    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    return [trainloader, testloader]


def get_imagenet():
    train_path = "C:\\fed\V2\\fedml-ng\data\data\Imagenet\\train"
    test_path = "C:\\fed\V2\\fedml-ng\data\data\Imagenet\\val"

    transform = transforms.Compose(
        transforms=[
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize([224, 224], interpolation=0),
        ]
    )
    # download the tar files mentioned in https://pytorch.org/vision/main/generated/torchvision.datasets.ImageNet.html before calling this function
    # selected 100 class from 1000

    train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_path, transform=transform)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

    return [trainloader, testloader]


def get_data(dataset_name):
    if dataset_name == "MNIST":
        data = get_mnist()
    elif dataset_name == "EMNIST":
        data = get_emnist("digits")
    elif dataset_name == "CIFAR10":
        data = get_cifar10()
    elif dataset_name == "CIFAR100":
        data = get_cifar100()
    elif dataset_name == "IMAGENET":
        data = get_imagenet()
    else:
        data = None
    return data


if __name__ == "__main__":
    random_seed(100)

    datasets = ["mnist", "emnist", "cifar10", "cifar100", "imagenet"]
    technique = ["dirichlet", "limit_label", "limit_label_unique", "probability", "equal_label", "iid", "fixed_samples"]

    # parser
    parser = argparse.ArgumentParser(description="Script to generate data partitions.")

    parser.add_argument(
        "-partition", choices=technique, help="select partition technique"
    )

    args, remaining_args = parser.parse_known_args()

    if args.partition == "dirichlet":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument("-a", type=float, required=True, help="alpha")
        subparser.add_argument(
            "-n_clients", type=int, required=True, help="no_of partitions"
        )
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-min_samples", type=int, required=False, default=100, help="min_samples"
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        if data == None:
            subparser.print_help()
        else:
            dirchlet(
                dataset=data[0],
                num_clients=subargs.n_clients,
                alpha=subargs.a,
                dataset_name=dataset_name,
                path=subargs.path,
                min_samples=subargs.min_samples,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    elif args.partition == "limit_label":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument("-f", type=float, required=False, help="fraction")
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-n_clients", type=int, required=True, help="no_of partitions"
        )
        subparser.add_argument(
            "-labels_per_client", type=int, required=True, help="labels per partition"
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        if data == None:
            subparser.print_help()
        else:
            limit_label(
                dataset=data[0],
                dataset_name=dataset_name,
                f=subargs.f,
                max_classes=subargs.labels_per_client,
                num_client=subargs.n_clients,
                path=subargs.path,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    elif args.partition == "limit_label_unique":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument("-samples", type=int, required=False, help="samples per class")
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-n_clients", type=int, required=True, help="no_of partitions"
        )
        subparser.add_argument(
            "-labels_per_client", type=int, required=True, help="labels per partition"
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        if data == None:
            subparser.print_help()
        else:
            limit_label_unique(
                dataset=data[0],
                dataset_name=dataset_name,
                samples=subargs.samples,
                max_classes=subargs.labels_per_client,
                num_client=subargs.n_clients,
                path=subargs.path,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    elif args.partition == "probability":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )
        subparser.add_argument(
            "-probability",
            nargs="+",
            required=True,
            help="list of probabilities ex: 0.2 0.3 0.5",
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        prob = [float(x) for x in subargs.probability]
        if data == None:
            subparser.print_help()
        elif sum(prob) > 1.0:
            print("sum probabilities sum is more than 1 !")
            subparser.print_help()
        else:
            probability_distribution(
                dataset=data[0],
                distribution=prob,
                dataset_name=dataset_name,
                path=subargs.path,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    elif args.partition == "iid":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )
        subparser.add_argument(
            "-n_clients", type=int, required=True, help="no_of partitions"
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        if data == None:
            subparser.print_help()
        else:
            equal_partition(
                dataset=data[0],
                num_client=subargs.n_clients,
                dataset_name=dataset_name,
                path=subargs.path,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    elif args.partition == "equal_label":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        if data == None:
            subparser.print_help()
        else:
            equal_label_distribution(
                dataset=data[0],
                dataset_name=dataset_name,
                path=subargs.path,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    elif args.partition == "fixed_samples":
        subparser = argparse.ArgumentParser(add_help=True)
        subparser.add_argument(
            "-data", type=str, required=True, help="dataset", choices=datasets
        )
        subparser.add_argument(
            "-path",
            type=str,
            required=False,
            nargs="?",
            default="./data/",
            help="path for saving partitions (optional)",
        )
        subparser.add_argument(
            "-n_clients", type=int, required=True, help="no_of partitions"
        )
        subparser.add_argument(
            "-samples", type=int, required= True, help="samples per partition"
        )

        subargs = subparser.parse_args(remaining_args)
        dataset_name = subargs.data.upper()
        data = get_data(dataset_name)
        if data == None:
            subparser.print_help()
        else:
            equal_samples(
                dataset=data[0],
                num_client=subargs.n_clients,
                num_samples=subargs.samples,
                dataset_name=dataset_name,
                path=subargs.path,
                task="train"
            )
            save_test_partition(data=data[1], dataset_name=dataset_name)

    else:
        parser.print_help()
