"""Benchmark script for Graph Neural Networks

Author : Emmanouil Karystinaios

Reference repo : https://github.com/melkisedeath/musym-GDL
"""
# ---------------------------- Standarize Testing Condinditions --------------------------------
import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)


import os, math
import argparse
# import torch
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.wandb import WandbLoggerCallback

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from musym.benchmark.utils import DataModule




def select_lighning_model(model):
    if model == "GraphSMOTE":
        from musym.models.rgcn_homo.GraphSMOTE.GraphSMOTE_lighting import GraphSMOTELightning, DataModule
        model = GraphSMOTELightning
        return model
    elif model == "SAGE":
        from musym.benchmark.model_acc.bench_sage_lightning import SAGELightning, DataModule
        model = SAGELightning
        return model
    elif model == "SMOTE":
        from musym.benchmark.model_acc.bench_smote_lightning import SmoteLightning, DataModule
        return model
    elif model == "SMOTEmbed":
        from musym.benchmark.model_acc.bench_smotembed_lightning import SmoteEmbedLightning, DataModule
        model = SmoteEmbedLightning
        return model
    else:
        raise ValueError("model name {} is not recognized or not implement for Pytorch Lightning.".format(model))


def train_lightning_tune(config, num_gpus=0):
    # check cuda
    # use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fanouts = [int(_) for _ in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)

    model = select_lighning_model(config["model"])
    datamodule = DataModule(
        dataset_name=config["dataset"], data_cpu=config["data_cpu"], fan_out=fanouts,
        batch_size=config["batch_size"], num_workers=config["num_workers"], device=device, init_weights=config["init_weights"])
    model = model(
        datamodule.in_feats, config["num_hidden"], datamodule.n_classes, config["num_layers"],
        F.relu, config["dropout"], config["lr"])

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=3)
    trainer = Trainer(
        gpus=math.ceil(num_gpus),
        max_epochs=config["num_epochs"],
        # logger=WandbLogger(project="SMOTE", group="{}-Lightning.format(config["model"])", job_type="Cadence-Detection"),
        callbacks=[
          checkpoint_callback,
          TuneReportCallback(
              {
                  "loss": "val_loss_epoch",
                  "mean_accuracy": "val_acc_epoch",
                  "val_fscore" : "val_fscore_epoch",
              },
              on="validation_end")
        ])
    trainer.fit(model, datamodule=datamodule)



def bench_tune_lighting():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='cad')
    argparser.add_argument('--model', type=str, default='GraphSMOTE')
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-samples', type=int, default=1000)
    argparser.add_argument('--gpus-per-trial', type=float, default=0.5)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the graph, node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()
    args.data_dir = os.path.join(os.path.dirname(__file__), "data")

    num_samples = args.num_samples
    # --------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)
    gpus_per_trial = args.gpus_per_trial
    config["num_hidden"] = tune.choice([32, 64])
    config["fan_out"] = tune.choice(["5,10", "5,10,15"])
    config["lr"] = tune.loguniform(1e-4, 1e-1)
    config["batch_size"] = tune.choice([512, 1024, 2048])

    scheduler = ASHAScheduler(
        max_t=config["num_epochs"],
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["num_hidden", "num_layers", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_lightning_tune,
            num_gpus=gpus_per_trial,
            ),
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        callbacks= [
            WandbLoggerCallback(project="SMOTE", group="{}-Lightning".format(config["model"]), job_type=config["dataset"])
            ],
        progress_reporter=reporter,
        name="tune_{}_{}".format(config["dataset"], config["model"]) )

    print("Best hyperparameters found were: ", analysis.best_config)



def bench_lightning():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0)
    argparser.add_argument('--dataset', type=str, default='cad')
    argparser.add_argument('--model', type=str, default='GraphSMOTE')
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument("--fan-out", type=str, default="5, 10")
    argparser.add_argument("--lr", type=float, default=1e-4)
    argparser.add_argument("--batch-size", type=int, default=1024)
    argparser.add_argument("--num-hidden", type=int, default=64)
    argparser.add_argument('--gpus-per-trial', type=float, default=0.5)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the graph, node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--init-weights", action='store_true', help="Initialize the graph weights")
    args = argparser.parse_args()
    args.data_dir = os.path.join(os.path.dirname(__file__), "data")

    # --------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    # check cuda
    use_cuda = torch.cuda.is_available() and config["gpu"] >= 0
    device = torch.device('cuda:%d' % config["gpu"] if use_cuda else "cpu")

    fanouts = [int(_) for _ in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)

    model = select_lighning_model(config["model"])
    datamodule = DataModule(
        dataset_name=config["dataset"], data_cpu=config["data_cpu"], fan_out=fanouts,
        batch_size=config["batch_size"], num_workers=config["num_workers"], device=device, init_weights=config["init_weights"])
    model = model(
        datamodule.in_feats, config["num_hidden"], datamodule.n_classes, config["num_layers"],
        F.relu, config["dropout"], config["lr"])

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=3)
    trainer = Trainer(
        gpus=[config["gpu"]],
        max_epochs=config["num_epochs"],
        logger=WandbLogger(project="Bench-SMOTE", group=config["dataset"], job_type="{}-Lightning".format(config["model"])),
        callbacks=[
            checkpoint_callback,
        ])
    trainer.fit(model, datamodule=datamodule)



if __name__ == "__main__":
    bench_lightning()