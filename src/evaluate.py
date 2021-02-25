import sys
import time

import numpy as np
import pandas
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from common import get_model_resgcn
from utils import AverageMeter
from datasets import dataset_factory
from datasets.augmentation import ShuffleSequence, SelectSequenceCenter, ToTensor, MultiInput
from datasets.graph import Graph


def _evaluate_casia_b(embeddings):
    """
    Test dataset consists of sequences of last 50 ids from CASIA B Dataset.
    Data is divided in the following way:
    Gallery Set:
        NM 1, NM 2, NM 3, NM 4
    Probe Set:
        Subset 1:
            NM 5, NM 6
         Subset 2:
            BG 1, BG 2
         Subset 3:
            CL 1, CL 2
    """

    gallery = {k: v for (k, v) in embeddings.items() if k[1] == 0 and k[2] <= 4}
    gallery_per_angle = {}
    for angle in range(0, 181, 18):
        gallery_per_angle[angle] = {k: v for (k, v) in gallery.items() if k[3] == angle}

    probe_nm = {k: v for (k, v) in embeddings.items() if k[1] == 0 and k[2] >= 5}
    probe_bg = {k: v for (k, v) in embeddings.items() if k[1] == 1}
    probe_cl = {k: v for (k, v) in embeddings.items() if k[1] == 2}

    correct = np.zeros((3, 11, 11))
    total = np.zeros((3, 11, 11))
    for gallery_angle in range(0, 181, 18):
        gallery_embeddings = np.array(list(gallery_per_angle[gallery_angle].values()))
        gallery_targets = list(gallery_per_angle[gallery_angle].keys())
        gallery_pos = int(gallery_angle / 18)

        probe_num = 0
        for probe in [probe_nm, probe_bg, probe_cl]:
            for (target, embedding) in probe.items():
                subject_id, _, _, probe_angle = target
                probe_pos = int(probe_angle / 18)

                distance = np.linalg.norm(gallery_embeddings - embedding, ord=2, axis=1)
                min_pos = np.argmin(distance)
                min_target = gallery_targets[int(min_pos)]

                if min_target[0] == subject_id:
                    correct[probe_num, gallery_pos, probe_pos] += 1
                total[probe_num, gallery_pos, probe_pos] += 1

            probe_num += 1

    accuracy = correct / total

    # Exclude same view
    for i in range(3):
        accuracy[i] -= np.diag(np.diag(accuracy[i]))

    accuracy_flat = np.sum(accuracy, 1) / 10

    header = ["NM#5-6", "BG#1-2", "CL#1-2"]

    accuracy_avg = np.mean(accuracy)
    sub_accuracies_avg = np.mean(accuracy_flat, 1)
    sub_accuracies = dict(zip(header, list(sub_accuracies_avg)))

    dataframe = pandas.DataFrame(
        np.concatenate((accuracy_flat, sub_accuracies_avg[..., np.newaxis]), 1),
        header,
        list(range(0, 181, 18)) + ["mean"],
    )

    return correct, accuracy_avg, sub_accuracies, dataframe


def evaluate(data_loader, model, evaluation_fn, log_interval=10, use_flip=False):
    model.eval()
    batch_time = AverageMeter()

    # Calculate embeddings
    with torch.no_grad():
        end = time.time()
        embeddings = dict()
        for idx, (points, target) in enumerate(data_loader):
            if use_flip:
                bsz = points.shape[0]
                data_flipped = torch.flip(points, dims=[1])
                points = torch.cat([points, data_flipped], dim=0)

            if torch.cuda.is_available():
                points = points.cuda(non_blocking=True)

            output = model(points)

            if use_flip:
                f1, f2 = torch.split(output, [bsz, bsz], dim=0)
                output = torch.mean(torch.stack([f1, f2]), dim=0)

            for i in range(output.shape[0]):
                sequence = tuple(
                    int(t[i]) if type(t[i]) is torch.Tensor else t[i] for t in target
                )
                embeddings[sequence] = output[i].cpu().numpy()

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % log_interval == 0:
                print(
                    f"Test: [{idx}/{len(data_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                )
                sys.stdout.flush()

    return evaluation_fn(embeddings)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on dataset")
    parser.add_argument("dataset", choices=["casia-b"])
    parser.add_argument("weights_path")
    parser.add_argument("data_path")
    parser.add_argument("--network_name", default="resgcn-n39-r8")
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_layer_size", type=int, default=128)
    parser.add_argument("--use_multi_branch", action="store_true")
    parser.add_argument("--shuffle", action="store_true")

    opt = parser.parse_args()

    # Config for dataset
    graph = Graph("coco")
    dataset_class = dataset_factory(opt.dataset)
    evaluation_fn = None
    if opt.dataset == "casia-b":
        evaluation_fn = _evaluate_casia_b

    # Load data
    dataset = dataset_class(
        opt.data_path,
        train=False,
        sequence_length=opt.sequence_length,
        transform=transforms.Compose(
            [
                SelectSequenceCenter(opt.sequence_length),
                ShuffleSequence(opt.shuffle),
                MultiInput(graph.connect_joint, opt.use_multi_branch),
                ToTensor()
            ]
        ),
    )
    data_loader = DataLoader(dataset, batch_size=opt.batch_size)
    print(f"Data loaded: {len(data_loader)} batches")

    # Init model
    model, model_args = get_model_resgcn(graph, opt)

    if torch.cuda.is_available():
        model.cuda()

    # Load weights
    checkpoint = torch.load(opt.weights_path)
    model.load_state_dict(checkpoint["model"])

    result, accuracy_avg, sub_accuracies, dataframe = evaluate(
        data_loader, model, evaluation_fn, use_flip=True
    )

    print("\n")
    print((dataframe * 100).round(2))
    print(f"AVG: {accuracy_avg*100} %")
    print("=================================")
    print((dataframe * 100).round(1).to_latex())
    print((dataframe * 100).round(1).to_markdown())


if __name__ == "__main__":
    main()
