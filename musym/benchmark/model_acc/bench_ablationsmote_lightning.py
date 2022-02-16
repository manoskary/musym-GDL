import torch as torch
import torch.nn.functional as F
import argparse
import os
from musym.models.rgcn_homo.GraphSMOTE.models import AblationSMOTE
from torchmetrics import Accuracy, F1, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from musym.benchmark.utils import DataModule

from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCallback


class AlbationSMOTE(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr,
                 rem_smote=False,
                 rem_gnn_enc=False,
                 rem_gnn_clf=False,
                 rem_adjmix=False,
                 loss_weight=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.module = AblationSMOTE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, rem_smote, rem_gnn_clf, rem_adjmix, rem_gnn_enc)
        self.lr = lr
        self.loss_weight = loss_weight
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_fscore = F1(n_classes, average="macro")
        self.val_auroc = AUROC(num_classes=n_classes, average="macro")

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        adj = mfgs[-1].adj().to_dense()[:len(batch_labels), :len(batch_labels)].to(self.device)
        batch_pred, upsampl_lab, embed_loss = self.module(mfgs, batch_inputs, adj, batch_labels)
        loss = F.cross_entropy(batch_pred, upsampl_lab) + embed_loss * self.loss_weight
        self.train_acc(torch.softmax(batch_pred, 1), upsampl_lab)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        batch_pred, batch_labels = self.module.val_forward(mfgs, batch_inputs, batch_labels)
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_acc(torch.softmax(batch_pred, 1), batch_labels)
        self.val_fscore(torch.softmax(batch_pred, 1), batch_labels)
        self.val_auroc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_fscore", self.val_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer




def evaluate(model, g, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    nfeat = g.ndata['feat']
    labels = g.ndata['label']
    with torch.no_grad():
        pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    test_acc = Accuracy()
    return test_acc(torch.softmax(pred[val_nid], -1), labels[val_nid].to(pred.device))


def ablation_study(config):
    rem_smote = False
    rem_gnn_enc = False
    rem_gnn_clf = False
    rem_adjmix = False
    if config["mode"] == "no-smote":
        rem_smote = True
    elif config["mode"] == "no-enc":
        rem_gnn_enc = True
    elif config["mode"] == "no-gnn-clf":
        rem_gnn_clf = True
    elif config["mode"] == "no-adj-mix":
        rem_adjmix = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlbationSMOTE
    fanouts = [int(_) for _ in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)
    datamodule = DataModule(
        dataset_name=config["dataset"], data_cpu=config["data_cpu"], fan_out=fanouts,
        batch_size=config["batch_size"], num_workers=config["num_workers"], device=device, init_weights=config["init_weights"], load_dir=config["load_dir"])

    model = model(
        datamodule.in_feats, config["num_hidden"], datamodule.n_classes, config["num_layers"],
        F.relu, config["dropout"], config["lr"], loss_weight=config["gamma"],
        rem_smote=rem_smote, rem_adjmix=rem_adjmix, rem_gnn_enc=rem_gnn_enc, rem_gnn_clf=rem_gnn_clf
    )

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=3)
    trainer = Trainer(gpus=1,
                      max_epochs=config["num_epochs"],
                      callbacks=[
                          checkpoint_callback,
                          TuneReportCallback(
                              {
                                  "loss": "val_loss_epoch",
                                  "mean_accuracy": "val_acc_epoch",
                                  "val_fscore": "val_fscore_epoch",
                                  "val_auroc": "val_auroc_epoch",
                                  "train_acc": "train_acc_epoch",
                              },
                              on="validation_end")
                      ])
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='cora')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--gamma', type=float, default=0.001)
    argparser.add_argument('--batch-size', type=int, default=64)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=2,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the graph, node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--init-weights', action='store_true',
                           help="Initializes random weights for the edges of the training graph.")
    # argparser.add_argument("--mode", type=str, default="full",
    #                        help="Mode of AblationSMOTE ",
    #                        choices = ["full", "no-smote", "no-enc", "no-gnn-clf", "no-adj-mix"]
    #                        )
    argparser.add_argument("--load-dir", type=str, default=".",
                           )
    args = argparser.parse_args()

    config = args if isinstance(args, dict) else vars(args)
    config["mode"] = tune.grid_search(["full", "no-smote", "no-enc", "no-gnn-clf", "no-adj-mix"])

    reporter = CLIReporter(
        parameter_columns=["mode"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            ablation_study
        ),
        resources_per_trial={
            "cpu": config["num_workers"] if config["num_workers"] > 0 else 1,
            "gpu": 1
        },
        metric="loss",
        mode="min",
        config=config,
        callbacks=[
            WandbLoggerCallback(project="AblationSMOTE", group=config["dataset"])
        ],
        progress_reporter=reporter,
        name="tune_{}_{}_{}".format(config["dataset"], AblationSMOTE, "mode"))




