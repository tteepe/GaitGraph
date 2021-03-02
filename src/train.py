import sys
import time

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

from datasets import dataset_factory
from datasets.augmentation import *
from datasets.graph import Graph
from evaluate import evaluate, _evaluate_casia_b
from losses import SupConLoss

from common import *
from utils import AverageMeter


def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (points, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        points = torch.cat([points[0], points[1]], dim=0)
        labels = target[0]

        if torch.cuda.is_available():
            points = points.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        with torch.cuda.amp.autocast(enabled=opt.use_amp):
            # compute loss
            features = model(points)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = criterion(features, labels)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.log_interval == 0:
            print(
                f"Train: [{epoch}][{idx + 1}/{len(train_loader)}]\t"
                f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"loss {losses.val:.3f} ({losses.avg:.3f})"
            )
            sys.stdout.flush()

    return losses.avg


def main(opt):
    opt = setup_environment(opt)
    graph = Graph("coco")

    # Dataset
    transform = transforms.Compose(
        [
            MirrorPoses(opt.mirror_probability),
            FlipSequence(opt.flip_probability),
            RandomSelectSequence(opt.sequence_length),
            ShuffleSequence(opt.shuffle),
            PointNoise(std=opt.point_noise_std),
            JointNoise(std=opt.joint_noise_std),
            MultiInput(graph.connect_joint, opt.use_multi_branch),
            ToTensor()
        ],
    )

    dataset_class = dataset_factory(opt.dataset)
    dataset = dataset_class(
        opt.train_data_path,
        train=True,
        sequence_length=opt.sequence_length,
        transform=TwoNoiseTransform(transform),
    )

    dataset_valid = dataset_class(
        opt.valid_data_path,
        sequence_length=opt.sequence_length,
        transform=transforms.Compose(
            [
                SelectSequenceCenter(opt.sequence_length),
                MultiInput(graph.connect_joint, opt.use_multi_branch),
                ToTensor()
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=opt.batch_size_validation,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    # Model & criterion
    model, model_args = get_model_resgcn(graph, opt)
    criterion = SupConLoss(temperature=opt.temp)

    print("# parameters: ", count_parameters(model))

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, opt.gpus)

    if opt.cuda:
        model.cuda()
        criterion.cuda()

    # Trainer
    optimizer, scheduler, scaler = get_trainer(model, opt, len(train_loader))

    # Load checkpoint or weights
    load_checkpoint(model, optimizer, scheduler, scaler, opt)

    # Tensorboard
    writer = SummaryWriter(log_dir=opt.tb_path)

    sample_input = torch.zeros(opt.batch_size, model_args["num_input"], model_args["num_channel"],
                               opt.sequence_length, graph.num_node).cuda()
    writer.add_graph(model, input_to_model=sample_input)

    best_acc = 0
    loss = 0
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(
            train_loader, model, criterion, optimizer, scheduler, scaler, epoch, opt
        )

        time2 = time.time()
        print(f"epoch {epoch}, total time {time2 - time1:.2f}")

        # tensorboard logger
        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

        # evaluation
        result, accuracy_avg, sub_accuracies, dataframe = evaluate(
            val_loader, model, opt.evaluation_fn, use_flip=True
        )
        writer.add_text("accuracy/validation", dataframe.to_markdown(), epoch)
        writer.add_scalar("accuracy/validation", accuracy_avg, epoch)
        for key, sub_accuracy in sub_accuracies.items():
            writer.add_scalar(f"accuracy/validation/{key}", sub_accuracy, epoch)

        print(f"epoch {epoch}, avg accuracy {accuracy_avg:.4f}")
        is_best = accuracy_avg > best_acc
        if is_best:
            best_acc = accuracy_avg

        if opt.tune:
            tune.report(accuracy=accuracy_avg)

        if epoch % opt.save_interval == 0 or (is_best and epoch > opt.save_best_start * opt.epochs):
            save_file = os.path.join(opt.save_folder, f"ckpt_epoch_{'best' if is_best else epoch}.pth")
            save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, scheduler, scaler, opt, opt.epochs, save_file)

    log_hyperparameter(writer, opt, best_acc, loss)

    print(f"best accuracy: {best_acc*100:.2f}")


def _inject_config(config):
    opt_new = {k: config[k] if k in config.keys() else v for k, v in vars(opt).items()}
    main(argparse.Namespace(**opt_new))


def tune_():
    hyperband = HyperBandScheduler(metric="accuracy", mode="max")

    analysis = tune.run(
        _inject_config,
        config={},
        stop={"accuracy": 0.90, "training_iteration": 100},
        resources_per_trial={"gpu": 1},
        num_samples=10,
        scheduler=hyperband
    )

    print("Best config: ", analysis.get_best_config(metric="accuracy", mode="max"))

    df = analysis.results_df
    print(df)


if __name__ == "__main__":
    import datetime

    opt = parse_option()

    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    opt.model_name = f"{date}_{opt.dataset}_{opt.network_name}" \
                     f"_lr_{opt.learning_rate}_decay_{opt.weight_decay}_bsz_{opt.batch_size}"

    if opt.exp_name:
        opt.model_name += "_" + opt.exp_name

    opt.model_path = f"../save/{opt.dataset}_models"
    opt.tb_path = f"../save/{opt.dataset}_tensorboard/{opt.model_name}"

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.evaluation_fn = None
    if opt.dataset == "casia-b":
        opt.evaluation_fn = _evaluate_casia_b

    if opt.tune:
        tune_()
    else:
        main(opt)
