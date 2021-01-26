import pandas as pd
import csv
from tqdm import tqdm

# Splitting data
# Training set : first 74 ids
# Test set     : rest 50 ids

skeletons = pd.read_csv("../../data/casia-b_pose_coco.csv")

header = list(skeletons)

ids_train = list(range(1, 60))
ids_valid = list(range(60, 75))
ids_test = list(range(75, 125))

balancing = {
    "nm": 1,
    "cl": 3,
    "bg": 3
}

# Store the different sets in lists according to the indexes assigned above
data = {"train": [], "valid": [], "train_valid": [], "test": [],
        "train_balanced": [], "valid_balanced": [],  "train_valid_balanced": []}

for skeleton in tqdm(skeletons.values.tolist()):
    label = skeleton[0].split('/')[1].split('-')
    p_id = int(label[0])
    p_ws = label[1]

    if p_id in ids_train:
        data["train"].append(skeleton)
        for _ in range(balancing[p_ws]):
            data["train_balanced"].append(skeleton)

    if p_id in ids_valid:
        data["valid"].append(skeleton)
        for _ in range(balancing[p_ws]):
            data["valid_balanced"].append(skeleton)

    if p_id in ids_valid or p_id in ids_train:
        data["train_valid"].append(skeleton)
        for _ in range(balancing[p_ws]):
            data["train_valid_balanced"].append(skeleton)

    if p_id in ids_test:
        data["test"].append(skeleton)

for split, lines in data.items():
    print(f"Saving {split}...")
    with open(f"../../data/casia-b_pose_{split}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for line in lines:
            writer.writerow(line)
