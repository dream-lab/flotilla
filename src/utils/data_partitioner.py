import torch
import os
from torchvision import datasets, transforms

# Step 1: Load MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
mnist_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
print(mnist_dataset.data[0])
# Step 2: Calculate partition sizes
total_size = len(mnist_dataset)
partition_size = total_size // 8
partitions = []
save_dir = "./data"

# Step 3: Create DataLoader for each partition
for i in range(8):
    start_idx = i * partition_size
    end_idx = (i + 1) * partition_size if i < 7 else total_size
    partition = torch.utils.data.Subset(mnist_dataset, list(range(start_idx, end_idx)))
    print("Partition data entries", len(partition))
    data_loader = torch.utils.data.DataLoader(partition, batch_size=1, shuffle=True)
    partitions.append(data_loader)

# Step 4: Save the partitions
for idx, data_loader in enumerate(partitions):
    torch.save(data_loader, os.path.join(save_dir, f"partition-{idx}.pth"))

# Step 5: Load the partitions
train_loader = torch.utils.data.DataLoader(
    torch.load(os.path.join(save_dir, "partition-0.pth")), shuffle=True, batch_size=1
)
train_dataset = torch.load(os.path.join(save_dir, "partition-0.pth")).dataset
print(train_dataset[0], len(train_loader))
