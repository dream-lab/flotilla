import pickle
import sys
from collections import Counter

import torch

dataset_path = sys.argv[1]

try:
    dataset = torch.load(dataset_path).dataset
    num_items = len(dataset)
    class_distrb = dict(Counter(dataset[i][1] for i in range(len(dataset))))
    for i, v in class_distrb.items():
        class_distrb[i] = v / num_items
    data_summary = {
        "label_distribution": class_distrb,
        "num_items": num_items,
        "data_filename": dataset_path,
    }

    print(data_summary)
    summary_filename = dataset_path.split(".")[0] + "_summary.data"

    print(summary_filename)
    with open(summary_filename, "wb") as f:
        pickle.dump(obj=data_summary, file=f)
except Exception as e:
    print("get_data_summary:: Exception - ", e)
