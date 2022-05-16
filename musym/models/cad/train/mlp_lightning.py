import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from pytorch_lightning import Trainer
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
import dgl
import torch
import os
from musym.utils import load_and_save, min_max_scaler
from sklearn.metrics import f1_score, precision_recall_fscore_support
import wandb
import numpy as np
from musym.models.cad.models import MLPSMOTE
from torchmetrics import Accuracy, AUROC, F1
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset


class CustomCadDataset(Dataset):
    def __init__(self, idx, features, labels):
        super(CustomCadDataset, self).__init__()
        self.cad_features = features[idx]
        self.cad_labels = labels[idx]

    def __len__(self):
        return len(self.cad_labels)

    def __getitem__(self, idx):
        feat = self.cad_features[idx]
        label = self.cad_labels[idx]
        return feat, label

class CadModelLightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 activation,
                 dropout,
                 lr,
                 loss_weight = 0.007,
                 weight_decay=5e-4,
        ):
        super(CadModelLightning, self).__init__()
        self.save_hyperparameters()
        self.loss_weight = loss_weight
        self.module = MLPSMOTE(in_feats, n_hidden, n_classes, activation, dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_fscore = F1(num_classes=n_classes, average="macro")
        self.train_auroc = AUROC(num_classes=n_classes, average="macro")
        self.val_fscore = F1(num_classes=n_classes, average="macro")
        self.val_auroc = AUROC(num_classes=n_classes, average="macro")
        self.test_fscore = F1(n_classes, average="macro")
        self.test_auroc = AUROC(num_classes=n_classes, average="macro")
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def training_step(self, batch, batch_idx):
        cad_features, cad_labels = batch
        batch_inputs = cad_features.to(self.device)
        batch_labels = cad_labels.to(self.device)
        batch_pred, upsampl_lab = self.module(batch_inputs, batch_labels)
        loss = self.train_loss(batch_pred, upsampl_lab)
        batch_pred = F.softmax(batch_pred, dim=1)
        self.train_acc(batch_pred, upsampl_lab)
        self.train_fscore(batch_pred[:len(batch_labels)], batch_labels)
        self.train_auroc(batch_pred[:len(batch_labels)], batch_labels)
        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_fscore", self.train_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        cad_features, cad_labels = batch
        batch_inputs = cad_features.to(self.device)
        batch_labels = cad_labels.to(self.device)
        batch_pred = self.module.pred_forward(batch_inputs)
        self.val_acc(batch_pred, batch_labels)
        loss = F.cross_entropy(batch_pred, batch_labels)
        batch_pred = F.softmax(batch_pred, dim=1)
        self.val_fscore(batch_pred, batch_labels)
        self.val_auroc(batch_pred, batch_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_fscore", self.val_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_fscore"
            }
        }


class CadDataModule(LightningDataModule):
    def __init__(self, features, labels, train_nid, val_nid=[], test_nid=[], batch_size=1000, num_workers=4):
        super().__init__()
        self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.train_dataset = CustomCadDataset(train_nid, features, labels)
        self.val_dataset = CustomCadDataset(val_nid, features, labels)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)






def save_predictions(labels, preds, idx, score_duration, piece_idx):
    import pandas as pd
    piece_idx = piece_idx[idx]
    labels = labels[idx]
    score_duration = score_duration[idx]
    for name in torch.unique(piece_idx):
        durs = score_duration[piece_idx == name]
        X = preds[piece_idx == name]
        y = labels[piece_idx == name]
        sorted_durs, resorted_idx = torch.sort(durs)
        X = X[resorted_idx]
        y = y[resorted_idx]
        z1, z2, z3 = list(
            zip(*[(udur.item(), X[sorted_durs == udur].argmax(1).tolist(), y[sorted_durs == udur].tolist()) for udur in
                  torch.unique(sorted_durs)]))
        pd.DataFrame(data={"score_duration": z1, "predictions": z2, "target": z3}).to_csv(
            "../samples/predictions_score_{}.csv".format(name))


def to_sequences(labels, preds, idx, score_duration, piece_idx, onsets, n_classes):
    seqs = list()
    trues = list()
    avr_f1 = "binary" if n_classes == 2 else "macro"
    # Make args same dimensions as preds
    piece_idx = piece_idx[idx]
    labels = labels[idx]
    score_duration = score_duration[idx]
    onsets = onsets[idx]
    # Select Downbeats
    o = torch.zeros(len(onsets))
    o[torch.nonzero(onsets == 0, as_tuple=True)[0]] = 1.00
    # Filter out up-beat instances
    mod_onsets = torch.remainder(onsets, 1)
    filter_beats_idx = torch.nonzero(mod_onsets == 0, as_tuple=True)[0]
    labels = labels[filter_beats_idx]
    preds = preds[filter_beats_idx]
    piece_idx = piece_idx[filter_beats_idx]
    score_duration = score_duration[filter_beats_idx]
    onsets = onsets[filter_beats_idx]
    f1_score_binary = f1_score(labels.numpy(), preds.argmax(dim=1).numpy(), average=avr_f1)
    # Start Building Sequence per piece name.
    for name in torch.unique(piece_idx):
        # Gather on non-augmented Pieces
        if name != 0:
            durs = score_duration[piece_idx == name]
            X = preds[piece_idx == name]
            y = labels[piece_idx == name]
            sorted_durs, resorted_idx = torch.sort(durs)
            X = X[resorted_idx]
            y = y[resorted_idx]
            new_X = []
            new_y = []
            # group by onset
            for udur in torch.unique(sorted_durs):
                x = X[sorted_durs == udur]
                z = y[sorted_durs == udur]
                if len(x.shape) > 1:
                    # Can be Max or Mean aggregation of likelihoods.
                    new_X.append(x.mean(dim=0))
                    # new_X.append(x.max(0)[0])
                    new_y.append(z.max().unsqueeze(0))
                else:
                    new_X.append(x)
                    new_y.append(z)
            s = torch.vstack(new_X)
            seqs.append(s)
            trues.append(torch.cat(new_y))
    return seqs, trues, f1_score_binary


def prepare_and_postprocess(g, model, batch_size, train_nids, val_nids, labels, node_features, score_duration, piece_idx, onsets, device, n_classes):
    avr_f1 = "binary" if n_classes == 2 else "macro"
    eval_model = model.module.to(device)
    val_loader = DataLoader(CustomCadDataset(val_nids, node_features, labels), batch_size=batch_size, shuffle=False, drop_last=False)
    val_prediction = eval_model.inference(val_loader, device)
    X_val, y_val, val_fscore = to_sequences(labels, val_prediction.detach().cpu(), val_nids, score_duration, piece_idx, onsets, n_classes)
    X_val = torch.vstack(X_val).numpy()
    y_val = torch.cat(y_val).numpy()
    y_pred = np.argmax(X_val, axis=1)
    thresh_val_acc = np.equal(y_pred, y_val).astype(float).sum() / len(y_val)
    thresh_pres, thresh_rec, thresh_f1, val_sup = precision_recall_fscore_support(y_val, y_pred, average=avr_f1)
    metrics = {
        "Onset_wise Fscore": val_fscore,
        "Threshold Beat-wise Val Accuracy": thresh_val_acc,
        "Threshold Beat-wise Val F1": thresh_f1,
        "Threshold Beat-wise Val Precision": thresh_pres,
        "Threshold Beat-wise Val Recall": thresh_rec,
        "Threshold Beat-wise Val Support": val_sup,
        }
    return metrics


def train(scidx, data, args, type=""):
    g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
    piece_idx, onsets, score_duration, device, dataloader_device, config = data

    group = config["dataset"]
    job_type = "MLP-{}-{}".format(type, config["num_hidden"])
    mn = "Pretrained-Net" if config["load_from_checkpoints"] else "Net"
    model_name = "{}_{}-({})x{}_lr={:.04f}_bs={}_lw={:.01f}".format(
        mn, scidx,
        config["fan_out"], config["num_hidden"],
        config["lr"], config["batch_size"], config["gamma"])

    run = wandb.init(
        project="Cadence Detection",
        group=group,
        job_type=job_type,
        reinit=True,
        name=model_name,
        config=config
    )

    datamodule = CadDataModule(
        features=node_features, labels=labels,
        train_nid=train_nids, val_nid=val_nids, test_nid=test_nids,
        batch_size=config["batch_size"], num_workers=config["num_workers"])
    if config["load_from_checkpoints"]:
        artifact = run.use_artifact('melkisedeath/Cadence Detection/model-{}:latest'.format(config["wandb_id"]), type='model')
        artifact_dir = artifact.download()
        model = CadModelLightning.load_from_checkpoint(
            checkpoint_path=os.path.join(os.path.normpath(artifact_dir), "model.ckpt"),
            in_feats=node_features.shape[1], n_hidden=config["num_hidden"],
            n_classes=n_classes, activation=F.relu, dropout=config["dropout"],
            lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        model = CadModelLightning(
            in_feats=node_features.shape[1], n_hidden=config["num_hidden"],
            n_classes=n_classes, activation=F.relu, dropout=config["dropout"],
            lr=config["lr"], weight_decay=config["weight_decay"]
            )

    wandb_logger = WandbLogger(
        project="Cadence Detection",
        group=group,
        job_type=job_type,
        name=model_name,
        log_model=True,
        reinit=True
    )
    trainer = Trainer(gpus=args.num_gpus,
                      auto_select_gpus=True if args.num_gpus else False,
                      max_epochs=config["num_epochs"],
                      logger=wandb_logger
                      )

    if not args.skip_training:
        trainer.fit(model, datamodule=datamodule)

    model.freeze()
    metrics = prepare_and_postprocess(g, model, config["batch_size"],
                                                            train_nids, test_nids,
                                                            labels, node_features,
                                                            score_duration, piece_idx,
                                                            onsets, device, n_classes)

    wandb.log(metrics)





def main(args):
    """
    Main Call for Node Classification with Node2Vec + GraphSMOTE + DBNN.
    """
    # --------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    if args.config_path != "" and os.path.exists(args.config_path) and args.config_path.endswith(".json"):
        import json
        with open(args.config_path, "r") as f:
            config_update = json.load(f)
        config = {k: (config_update[k]["value"] if k in config_update.keys() else v) for k,v in config.items()}

    config["shuffle"] = bool(config["shuffle"])

    # --------------- Dataset Loading -------------------------
    if args.features == "all":
        g, n_classes = load_and_save("cad_feature_" + args.dataset, args.data_dir)
    elif args.features == "local":
        g, n_classes = load_and_save("cad_local_" + args.dataset, args.data_dir)
    elif args.features == "cad":
        g, n_classes = load_and_save("cad_algo_" + args.dataset, args.data_dir)
    g = dgl.add_self_loop(dgl.add_reverse_edges(g))
    # training defs
    labels = g.ndata.pop('label')
    if args.cad_type == "pac":
        labels = (labels == 1).long()
        n_classes = 2
    elif args.cad_type == "riac":
        labels = (labels != 0).long()
        n_classes = 2
    elif args.cad_type == "hc":
        labels = (labels == 2).long()
        n_classes = 2
    elif args.cad_type in ["multiclass", "multi"]:
        pass
    else:
        raise AttributeError("Cadence Type {} not recognized".format(args.cad_type))

    # training defs
    train_nids = torch.nonzero(g.ndata.pop('train_mask'), as_tuple=True)[0]
    node_features = g.ndata.pop('feat')
    piece_idx = g.ndata.pop("score_name")
    onsets = node_features[:, 0]
    score_duration = node_features[:, 3]

    # Validation and Testing
    val_nids = torch.nonzero(g.ndata.pop('val_mask'), as_tuple=True)[0]
    train_nids = torch.cat((train_nids, val_nids))
    test_nids = val_nids = torch.nonzero(g.ndata.pop('test_mask'), as_tuple=True)[0]
    # check cuda
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % torch.cuda.current_device() if use_cuda else 'cpu')
    dataloader_device = "cpu"


    node_features = min_max_scaler(node_features)
    # create model
    # ========= kfold TRAINING ============
    if args.loocv:
        unique_scores = torch.unique(piece_idx)
        for scidx in unique_scores:
            val_nids = test_nids = torch.nonzero(piece_idx == scidx, as_tuple=True)[0]
            train_nids = torch.nonzero(piece_idx != scidx, as_tuple=True)[0]
            data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
                   piece_idx, onsets, score_duration, device, dataloader_device, config
            train(scidx, data, args, type="LOOCV")
    elif args.kfold:
        unique_scores = torch.unique(piece_idx)
        num_folds = args.kfold
        for fold_num in range(num_folds):
            pick = torch.randperm(len(unique_scores)) + 1
            train_fold = pick[:int(len(unique_scores)*0.7)]
            val_fold = pick[int(len(unique_scores)*0.7): int(len(unique_scores)*0.7) + int(len(unique_scores)*0.1)]
            test_fold = pick[int(len(unique_scores)*0.7) + int(len(unique_scores)*0.1):]
            val_nids = torch.cat([torch.nonzero(piece_idx==scidx, as_tuple=True)[0] for scidx in val_fold])
            test_nids = torch.cat([torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in test_fold])
            train_nids = torch.cat([torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in train_fold])
            data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
                   piece_idx, onsets, score_duration, device, dataloader_device, config
            train(fold_num, data, args, type="kFold")
    else:
        if "wtc" in args.dataset:
            train_fold = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            val_fold = [13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        elif "hsq" in args.dataset:
            train_fold = list(range(1, 25, 1))
            val_fold = list(range(25, 46, 1))
        else:
            num_scores = len(torch.unique(piece_idx))
            train_fold = list(range(int(num_scores*0.7)))
            val_fold = list(range(int(num_scores*0.7), num_scores, 1))
        val_nids = test_nids = torch.cat([torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in val_fold])
        train_nids = torch.cat([torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in train_fold])

        data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
               piece_idx, onsets, score_duration, device, dataloader_device, config
        train("", data, args, type="S")


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='Cadence Learning GraphSMOTE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--dataset", type=str, default="wtc", choices=["wtc", "hsq", "msq", "quartets"])
    argparser.add_argument("--cad-type", type=str, default="pac", choices=["pac", "riac", "hc", "multiclass", "multi"])
    argparser.add_argument("--features", type=str, default="all", choices=["local", "all", "cad"])
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=0)
    argparser.add_argument('--lr', type=float, default=0.0007)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=0.005,
                           help="Weight for L2 loss")
    argparser.add_argument("--gamma", type=float, default=0.5,
                           help="weight of decoder regularization loss.")
    argparser.add_argument("--ext-mode", type=str, default=None, choices=["None", "lstm", "attention"])
    argparser.add_argument("--fan-out", type=str, default='')
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=1024)
    argparser.add_argument("--adjacency_threshold", type=float, default=0.5)
    argparser.add_argument("--num-workers", type=int, default=4)
    argparser.add_argument("--wandb_id", type=str, default="")
    argparser.add_argument("--config-path", type=str, default="")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
    argparser.add_argument("--load_from_checkpoints", action="store_true")
    argparser.add_argument("--kfold", type=int, default=0)
    argparser.add_argument("--skip_training", action="store_true")
    argparser.add_argument("--loocv", action="store_true")
    args = argparser.parse_args()
    pred = main(args)



