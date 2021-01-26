import os
import argparse
import torch
from models.st_gcn.st_gcn import STGCNEmbedding
import models.ResGCNv1


def parse_option():
    parser = argparse.ArgumentParser(description="Training model on gait sequence")
    parser.add_argument("dataset", choices=["casia-b", "outdoor-gait", "tum-gaid"])
    parser.add_argument("train_data_path", help="Path to train data CSV")
    parser.add_argument("--valid_data_path", help="Path to validation data CSV")
    parser.add_argument("--valid_split", type=float, default=0.2)

    parser.add_argument("--checkpoint_path", help="Path to checkpoint to resume")
    parser.add_argument("--weight_path", help="Path to weights for model")

    # Optionals
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--gpus", default="0", help="-1 for CPU, use comma for multiple gpus"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batch_size_validation", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50, help="save frequency")
    parser.add_argument(
        "--save_best_start", type=float, default=0.3, help="save frequency"
    )
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--exp_name", help="Name of the experiment")

    parser.add_argument("--network_name", default="resgcn-n39-r4")
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--embedding_layer_size", type=int, default=256)
    parser.add_argument("--temporal_kernel_size", type=int, default=9)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr_decay_rate", type=float, default=0.1, help="decay rate for learning rate"
    )
    parser.add_argument("--point_noise_std", type=float, default=0.05)
    parser.add_argument("--joint_noise_std", type=float, default=0.1)
    parser.add_argument("--flip_probability", type=float, default=0.5)
    parser.add_argument("--mirror_probability", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--use_multi_branch", action="store_true")
    parser.add_argument(
        "--temp", type=float, default=0.07, help="temperature for loss function"
    )
    opt = parser.parse_args()

    # Sanitize opts
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(",")]

    return opt


def log_hyperparameter(writer, opt, accuracy, loss):
    writer.add_hparams(
        {
            "batch_size": opt.batch_size,
            "sequence_length": opt.sequence_length,
            "embedding_layer_size": opt.embedding_layer_size,
            "dropout": opt.dropout,
            "learning_rate": opt.learning_rate,
            "lr_decay_rate": opt.lr_decay_rate,
            "point_noise_std": opt.point_noise_std,
            "weight_decay": opt.weight_decay,
            "temp": opt.temp,
        },
        {
            "hparam/accuracy": accuracy,
            "hparam/loss": loss,
        },
    )


def setup_environment(opt):
    # HACK: Fix tensorboard
    import tensorflow as tf
    import tensorboard as tb

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus_str
    opt.cuda = opt.gpus[0] >= 0
    torch.device("cuda" if opt.cuda else "cpu")

    return opt


def get_model_stgcn(opt):
    # Model
    input_channels = 3
    edge_importance_weighting = True
    graph_args = {"strategy": "spatial"}

    embedding_net = STGCNEmbedding(
        input_channels,
        graph_args,
        edge_importance_weighting=edge_importance_weighting,
        embedding_layer_size=opt.embedding_layer_size,
        temporal_kernel_size=opt.temporal_kernel_size,
        dropout=opt.dropout,
    )

    return embedding_net


def get_model_resgcn(graph, opt):
    model_args = {
        "A": torch.tensor(graph.A, dtype=torch.float32, requires_grad=False),
        "num_class": opt.embedding_layer_size,
        "num_input": 1 if not opt.use_multi_branch else 3,
        "num_channel": 3 if not opt.use_multi_branch else 6,
        "parts": graph.parts,
    }
    return models.ResGCNv1.create(opt.network_name, **model_args)


def get_trainer(model, opt, steps_per_epoch):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, opt.learning_rate, epochs=opt.epochs, steps_per_epoch=steps_per_epoch
    )
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    return optimizer, scheduler, scaler


def load_checkpoint(model, optimizer, scheduler, scaler, opt):
    if opt.checkpoint_path is not None:
        checkpoint = torch.load(opt.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        opt.start_epoch = checkpoint["epoch"]

    if opt.weight_path is not None:
        checkpoint = torch.load(opt.weight_path)
        model.load_state_dict(checkpoint["model"], strict=False)


def save_model(model, optimizer, scheduler, scaler, opt, epoch, save_file):
    print("==> Saving...")
    state = {
        "opt": opt,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
    }
    torch.save(state, save_file)
    del state


def count_parameters(model):
    """
    Useful function to compute number of parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
