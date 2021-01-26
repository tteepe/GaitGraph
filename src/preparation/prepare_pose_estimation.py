import csv
import itertools

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from tqdm import tqdm

from datasets import DatasetDetections, CropToBox
from pose_estimator.pose_estimator_hrnet import PoseEstimatorHRNet
from pose_estimator.utils import *
from visualization.utils import keypoints

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def pose_estimation(dataset_base_path, detection_list, output_file):

    pose_estimator = PoseEstimatorHRNet(
        config_path="../pose_estimator/inference-config.yaml",
        weights_path="../../weights/pose_hrnet_w32_384x288.pth",
    )
    transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = DatasetDetections(
        dataset_base_path,
        detection_list,
        sample_transform=CropToBox(pose_estimator.config),
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transform_normalize,
            ]
        ),
    )
    data_loader = DataLoader(dataset, batch_size=200, shuffle=False, num_workers=8)
    print(f"Data loaded: {len(data_loader)} batches")

    file = open(output_file, "w")
    writer = csv.writer(file)
    header = [[f"{k}_x", f"{k}_y", f"{k}_conf"] for k in keypoints.values()]
    writer.writerow(["image_name"] + list(itertools.chain.from_iterable(header)))

    poses = dict()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        imgs = data[0].squeeze()
        names = data[1]
        centers, scales = data[2]

        with torch.no_grad():
            # compute output heatmap
            output = pose_estimator.model(imgs)
            preds, maxvals = get_final_preds(
                pose_estimator.config,
                output.clone().cpu().numpy(),
                np.asarray(centers),
                np.asarray(scales),
            )

            result = np.append(preds, maxvals, axis=2)

            for j in range(imgs.shape[0]):
                poses[names[j]] = result[j]
                writer.writerow([names[j]] + list(result[j].reshape(-1)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect poses in dataset")
    parser.add_argument("dataset_base_path")
    parser.add_argument("detection_list", default='../../data/casia-b_detections.csv')
    parser.add_argument("output_file", default='../../data/casia-b_pose_coco.csv')

    args = parser.parse_args()
    pose_estimation(**vars(args))
