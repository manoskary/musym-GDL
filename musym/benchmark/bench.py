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


import os
import argparse
# import torch
import torch.nn.functional as F
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger


def select_lighning_model(model):
    if model == "GraphSMOTE":
        from musym.models.rgcn_homo.GraphSMOTE.GraphSMOTE_lighting import GraphSMOTELightning, DataModule
        model = GraphSMOTELightning
        datamodule = DataModule
        return model, datamodule
    else:
        raise ValueError("model name {} is not recognized or not implement for Pytorch Lightning.".format(model))


def train_lightning_tune(config):
    # check cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % torch.cuda.current_device() if use_cuda else 'cpu')

    model, datamodule = select_lighning_model(config["model"])
    fanouts = [int(_) for _ in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)
    datamodule = datamodule(
        config["dataset"], config["data_cpu"], fanouts,
        device, config["batch_size"], config["num_workers"])
    model = model(
        datamodule.in_feats, ["num_hidden"], datamodule.n_classes, config["num_layers"],
        F.relu, config["dropout"], config["lr"])

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    trainer = Trainer(gpus=[config["gpu"]] if config["gpu"] != -1 else None,
                      accelerator="auto",
                      strategy="ddp",
                      auto_scale_batch_size="binsearch",
                      max_epochs=config["num_epochs"],
                      logger=WandbLogger(project="SMOTE", group="GraphSMOTE-Lightning", job_type="Cadence-Detection"),
                      callbacks=[
                          checkpoint_callback,
                          TuneReportCallback(
                              {
                                  "loss": "ptl/val_loss",
                                  "mean_accuracy": "ptl/val_acc"
                              },
                              on="validation_end")
                          ])
    trainer.fit(model, datamodule=datamodule)



def bench_tune_lighting():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='cad')
    argparser.add_argument('--model', type=str, default='GraphSMOTE')
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-samples', type=int, default=10)
    argparser.add_argument('--gpus-per-trial', type=float, default=0.5)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
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
    config["num_hidden"] = tune.choice([16, 32, 64])
    config["num_layers"] = tune.choice(["5,10", "10,15", "5", "5,10,15"])
    config["lr"] = tune.loguniform(1e-4, 1e-1)
    config["batch_size"] = tune.choice([256, 512, 1024, 2048])

    scheduler = ASHAScheduler(
        max_t=config["num_epochs"],
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["num_hidden", "num_layers", "lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        train_lightning_tune,
        resources_per_trial={
            "cpu": 1,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_{}_{}".format(config["dataset"], config["model"]) )

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    bench_tune_lighting()