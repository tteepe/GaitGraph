import csv

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets import DatasetSimple
from detector.detector_yolov3 import DetectorYOLOv3
from detector.detector_utils import preprocess_image
from detector.utils import non_max_suppression, rescale_boxes

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def detection(dataset_base_path, image_list, output_file):
    dataset = DatasetSimple(
        dataset_base_path, image_list, transform=preprocess_image
    )
    data_loader = DataLoader(dataset, batch_size=30, shuffle=False, num_workers=8)
    print(f"Data loaded: {len(data_loader)} batches")

    file = open(output_file, "w")
    writer = csv.writer(file)
    writer.writerow(["image_name", "x", "y", "w", "h"])

    detector = DetectorYOLOv3(
        model_def="../detector/config/yolov3.cfg",
        weights_path="../../models/yolov3.weights",
    )

    human_candidates = dict()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        imgs = data[0].squeeze()
        names = data[1]

        # Configure input
        input_imgs = torch.autograd.Variable(imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = detector.model(input_imgs)
            detections = non_max_suppression(
                detections, detector.conf_thres, detector.nms_thres
            )

        for j in range(imgs.shape[0]):
            human_candidates[names[j]] = list()
            if detections[j] is None:
                continue

            detection = detections[j].data.cpu().numpy()
            detection = rescale_boxes(detection, detector.img_size, (240, 320))

            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                box_w = x2 - x1
                box_h = y2 - y1

                if int(cls_pred) == 0:
                    human_candidates[names[j]].append([x1, y1, box_w, box_h])

            if len(human_candidates[names[j]]) < 1:
                print(
                    f"{names[j]}: Invalid detections ({len(human_candidates[names[j]])}), skipping"
                )
                continue

            writer.writerow([names[j]] + human_candidates[names[j]][0])

    file.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect people in dataset")
    parser.add_argument("dataset_base_path")
    parser.add_argument("image_list")
    parser.add_argument("output_file")

    args = parser.parse_args()
    detection(**vars(args))
