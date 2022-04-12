from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import torch.nn.functional as F
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from musym.utils import load_and_save, min_max_scaler
import torch
from musym.models.cad.models import CadModelLightning, CadDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
import dgl
import math
import wandb
from datetime import datetime
from train_lightning import prepare_and_postprocess


def train_cad_tune(config, data):
    """
        Main Call for Node Classification with Node2Vec + GraphSMOTE.
    """
    g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
    piece_idx, onsets, score_duration, num_gpus = data

    fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)
    config["shuffle"] = bool(config["shuffle"])

    # create model
    datamodule = CadDataModule(
        g=g, n_classes=n_classes, in_feats=node_features.shape[1],
        train_nid=train_nids, val_nid=val_nids, test_nid=test_nids,
        data_cpu=config["data_cpu"], fan_out=fanouts,
        batch_size=config["batch_size"], num_workers=config["num_workers"], use_ddp=False)
    model = CadModelLightning(
        node_features=node_features, labels=labels,
        in_feats=datamodule.in_feats, n_hidden=config["num_hidden"],
        n_classes=datamodule.n_classes, n_layers=config["num_layers"],
        activation=F.relu, dropout=config["dropout"], lr=config["lr"],
        loss_weight=config["gamma"], ext_mode=config["ext_mode"], weight_decay=config["weight_decay"], adj_thresh=config["adjacency_threshold"],)
    model_name = "{}-({})x{}_lr={:.04f}_bs={}_lw={:.04f}".format(
        "Net", config["fan_out"], config["num_hidden"],
        config["lr"], config["batch_size"], config["gamma"])
    wandb.init(
        project="Cad Learning",
        group=config["dataset"],
        job_type="TUNE_{}x{}".format(config["num_layers"], config["num_hidden"]),
        reinit=True,
        name=model_name
    )

    # Train
    dt = datetime.today()
    dt_str = "{}.{}.{}.{}.{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    checkpoint_callback = ModelCheckpoint(
        dirpath="./cad_checkpoints/{}-{}".format(model_name, dt_str),
        monitor='val_fscore_epoch',
        mode="max",
        save_top_k=5,
        save_last=True,
        filename='{epoch}-{val_fscore_epoch:.2f}-{train_loss:.2f}'
    )
    # early_stopping = EarlyStopping('val_fscore', mode="max", patience=10)
    wandb_logger = WandbLogger(
        project="Cad Learning",
        group=config["dataset"],
        job_type="TUNE_C_{}x{}".format(config["num_layers"], config["num_hidden"]),
        name=model_name,
        reinit=True
    )

    wandb_logger.log_hyperparams(config)
    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        progress_bar_refresh_rate=0,
        logger=wandb_logger,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss_epoch",
                    "mean_accuracy": "val_fscore_epoch",
                    "val_acc": "val_acc_epoch",
                    "val_fscore": "val_fscore_epoch",
                    "val_auroc": "val_auroc_epoch",
                    "train_loss": "train_loss_epoch",
                    "train_acc": "train_acc_epoch",
                    "train_auroc": "train_auroc_epoch",
                    "train_fscore": "train_fscore_epoch"
                },
                on="validation_end"),
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "val_loss_epoch",
                    "mean_accuracy": "val_fscore_epoch"
                },
                filename="tune_checkpoint",
                on="validation_end"
            )
        ])
    trainer.fit(model, datamodule=datamodule)
    model.freeze()
    postprocess_val_acc, postprocess_val_f1, binary_val_fscore = prepare_and_postprocess(g, model, config["batch_size"],
                                                                                         train_nids, test_nids,
                                                                                         labels, node_features,
                                                                                         score_duration, piece_idx,
                                                                                         onsets, 0)

    wandb.log({"positive_class_val_fscore": binary_val_fscore,
               "Postprocess Onset-wise Val Accuracy": postprocess_val_acc,
               "Postprocess Onset-wise Val F1": postprocess_val_f1,
               })


argparser = argparse.ArgumentParser(description='Cadence Learning GraphSMOTE')
argparser.add_argument('--dataset', type=str, default="cad_pac_wtc")
argparser.add_argument('--gpus-per-trial', type=float, default=0.5)
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument('--num-epochs', type=int, default=100)
argparser.add_argument('--num-hidden', type=int, default=128)
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--lr', type=float, default=0.001123)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument("--weight-decay", type=float, default=5e-4,
                       help="Weight for L2 loss")
argparser.add_argument("--gamma", type=float, default=0.001248,
                       help="weight of decoder regularization loss.")
argparser.add_argument("--ext-mode", type=str, default=None, choices=["lstm", "attention"])
argparser.add_argument("--fan-out", default=[5, 10])
argparser.add_argument('--shuffle', type=int, default=True)
argparser.add_argument('--adjacency_threshold', type=float, default=0.01)
argparser.add_argument("--batch-size", type=int, default=1024)
argparser.add_argument("--num-workers", type=int, default=0)
argparser.add_argument("--num-samples", type=int, default=150)
argparser.add_argument('--data-cpu', action='store_true',
                       help="By default the script puts all node features and labels "
                            "on GPU when using it to save time for data copy. This may "
                            "be undesired if they cannot fit in GPU memory at once. "
                            "This flag disables that.")
argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
args = argparser.parse_args()

config = args if isinstance(args, dict) else vars(args)
gpus_per_trial = args.gpus_per_trial
config["fan_out"] = tune.choice(["5,10", "5,10,15"])
config["lr"] = tune.uniform(0.0001, 0.01)
config["weight_decay"] = tune.uniform(1e-5, 1e-2)
config["gamma"] = tune.uniform(0.0, 1.0)
config["batch_size"] = 2048
config["num_hidden"] = tune.choice([64, 128, 256])


# --------------- Dataset Loading -------------------------
g, n_classes = load_and_save(config["dataset"], config["data_dir"])
g = dgl.add_self_loop(dgl.add_reverse_edges(g))
# training defs
labels = g.ndata.pop('label')
node_features = g.ndata.pop('feat')
piece_idx = g.ndata.pop("score_name")
onsets = node_features[:, 0]
score_duration = node_features[:, 3]

# Validation and Testing
train_nids = torch.nonzero(g.ndata.pop('train_mask'), as_tuple=True)[0]
val_nids = torch.nonzero(g.ndata.pop('val_mask'), as_tuple=True)[0]
train_nids = torch.cat((train_nids, val_nids))
test_nids = val_nids = torch.nonzero(g.ndata.pop('test_mask'), as_tuple=True)[0]


node_features = min_max_scaler(node_features)


reporter = CLIReporter(
        parameter_columns=["num_hidden", "fan_out", "lr", "gamma", "weight_decay"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

scheduler = ASHAScheduler(
        max_t=config["num_epochs"],
        grace_period=20,
        reduction_factor=2)

data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
           piece_idx, onsets, score_duration, gpus_per_trial

analysis = tune.run(
        tune.with_parameters(
            train_cad_tune,
            data=data
            ),
        resources_per_trial={
            "cpu": 2,
            "gpu": gpus_per_trial
        },
        metric="mean_accuracy",
        mode="max",
        config=config,
        num_samples=config["num_samples"],
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_{}_preproc+PE+sp+t".format(config["dataset"]))

print("Best hyperparameters found were: ", analysis.best_config)
