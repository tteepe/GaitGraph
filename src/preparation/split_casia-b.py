import os
import csv
import pandas as pd
from tqdm import tqdm


# Splitting data
# Training set : first 74 ids
# Test set     : rest 50 ids
def main(input_file, output_dir):
    skeletons = pd.read_csv(input_file)

    header = list(skeletons)

    ids_train = list(range(1, 60))
    ids_valid = list(range(60, 75))
    ids_test = list(range(75, 125))

    # Store the different sets in lists according to the indexes assigned above
    data = {"train": [], "valid": [], "train_valid": [], "test": []}

    for skeleton in tqdm(skeletons.values.tolist()):
        label = skeleton[0].split('/')[1].split('-')
        p_id = int(label[0])
        p_ws = label[1]

        if p_id in ids_train:
            data["train"].append(skeleton)

        if p_id in ids_valid:
            data["valid"].append(skeleton)

        if p_id in ids_valid or p_id in ids_train:
            data["train_valid"].append(skeleton)

        if p_id in ids_test:
            data["test"].append(skeleton)

    for split, lines in data.items():
        print(f"Saving {split}...")
        with open(os.path.join(output_dir, f"casia-b_pose_{split}.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for line in lines:
                writer.writerow(line)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split CASIA-B")
    parser.add_argument("input_file")
    parser.add_argument("--output_dir", default='../../data')

    args = parser.parse_args()
    main(**vars(args))
