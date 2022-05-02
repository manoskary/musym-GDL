import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from musym.models.cad.models.cad_model import FullGraphCadLightning, FullGraphDataModule, CadGraphDataset
from pytorch_lightning import Trainer
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import dgl
import torch
from datetime import datetime
import os
from torch.utils.data import DataLoader
from musym.utils import load_and_save, min_max_scaler
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import wandb
import numpy as np


def to_sequences(labels, preds, idx, score_duration, piece_idx, onsets, n_classes):
    seqs = list()
    trues = list()
    avr_f1 = "binary" if n_classes == 2 else "macro"
    # Make args same dimensions as preds
    idx = [torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in torch.unique(piece_idx[idx])]
    piece_idx = torch.cat([piece_idx[scidx] for scidx in idx], dim=0)
    score_duration = torch.cat([score_duration[scidx] for scidx in idx])
    onsets = torch.cat([onsets[scidx] for scidx in idx])

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
            is_onset = onsets[piece_idx == name]
            X = preds[piece_idx == name]
            y = labels[piece_idx == name]
            sorted_durs, resorted_idx = torch.sort(durs)
            X = X[resorted_idx]
            y = y[resorted_idx]
            is_onset = is_onset[resorted_idx]
            new_X = []
            new_y = []
            # group by onset
            for udur in torch.unique(sorted_durs):
                # x = torch.cat((X[sorted_durs == udur], is_onset[sorted_durs == udur].unsqueeze(1)), dim=1)
                x = X[sorted_durs == udur]
                z = y[sorted_durs == udur]
                if len(x.shape) > 1:
                    # Can be Max or Mean aggregation of likelihoods.
                    # new_X.append(x.mean(dim=0))
                    new_X.append(x.max(0)[0])
                    new_y.append(z.max().unsqueeze(0))
                else:
                    new_X.append(x)
                    new_y.append(z)
            seqs.append(torch.vstack(new_X))
            trues.append(torch.cat(new_y))
    return seqs, trues, f1_score_binary


def prepare_and_postprocess(g, model, val_nids, labels, node_features, score_duration, piece_idx, onsets, device, n_classes):
    avr_f1 = "binary" if n_classes == 2 else "macro"
    eval_model = model.module.to(device)
    dataset = CadGraphDataset(g, val_nids, piece_idx, node_features, labels)
    val_loader = DataLoader(dataset)
    val_prediction, val_labels = eval_model.inference(val_loader, device)
    X_val, y_val, val_fscore = to_sequences(val_labels, val_prediction.detach().cpu(), val_nids, score_duration, piece_idx, onsets, n_classes)
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


def post_metrics(preds, target):
    tn, fp, fn, tp = confusion_matrix(target, preds).ravel()
    f1 = f1_score(target, preds, average = "binary")
    metrics = {
        "true_positives": tp,
        "true_negatives": tn,
        "false_negatives": fn,
        "num_beats": len(preds),
        "f1_score": f1
    }
    return metrics


def train(scidx, data, args, type=""):
    g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
    piece_idx, onsets, score_duration, device, dataloader_device, config = data

    group = config["dataset"]
    job_type = "FullNet-{}_{}x{}".format(type, config["num_layers"], config["num_hidden"])

    datamodule = FullGraphDataModule(
        g=g, node_features=node_features, labels=labels, piece_idx=piece_idx,
        in_feats=node_features.shape[1], train_nid=train_nids, val_nid=val_nids, num_workers=config["num_workers"]
    )
    model = FullGraphCadLightning(
        in_feats=node_features.shape[1], n_hidden=config["num_hidden"],
        n_classes=n_classes, n_layers=config["num_layers"],
        activation=F.relu, dropout=config["dropout"], lr=config["lr"],
        loss_weight=config["gamma"], weight_decay=config["weight_decay"],
        adj_thresh=config["adjacency_threshold"])
    model_name = "{}_{}-{}x{}_lr={:.04f}_lw={:.01f}".format(
        "FullNet", scidx,
        config["num_layers"], config["num_hidden"],
        config["lr"], config["gamma"])

    wandb.init(
        project="Cad Learning",
        group=group,
        job_type=job_type,
        reinit=True,
        name=model_name
    )
    # Train
    dt = datetime.today()
    dt_str = "{}.{}.{}.{}.{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    checkpoint_callback = ModelCheckpoint(
        dirpath="./cad_checkpoints/{}/{}/{}-{}".format(group, job_type, model_name, dt_str),
        monitor='val_fscore_epoch',
        mode="max",
        save_top_k=5,
        save_last=True,
        filename='{epoch}-{val_fscore_epoch:.2f}-{train_loss:.2f}'
    )
    # early_stopping = EarlyStopping('val_fscore', mode="max", patience=10)
    wandb_logger = WandbLogger(
        project="Cad Learning",
        group=group,
        job_type=job_type,
        name=model_name,
        reinit=True
    )
    trainer = Trainer(gpus=args.num_gpus if args.num_gpus > 0 else None,
                      auto_select_gpus=True if args.num_gpus > 0 else False,
                      max_epochs=config["num_epochs"],
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=0)

    if not args.skip_training:
        trainer.fit(model, datamodule=datamodule)

    if args.postprocess:
        model.freeze()
        metrics = prepare_and_postprocess(
            g, model,
            test_nids, labels, node_features,
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

    g, n_classes = load_and_save(config["dataset"], args.data_dir)
    g = dgl.add_self_loop(dgl.add_reverse_edges(g))
    # training defs
    labels = g.ndata.pop('label')
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
        for i in range(3):
            if "wtc" in args.dataset:
                train_fold = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                val_fold = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
            elif "hsq" in args.dataset:
                train_fold = list(range(1, 25, 1))
                val_fold = list(range(25, 46, 1))
            else:
                raise ValueError("Invalid training configuration. A split has not been defined for the {} dataset.".format(args.dataset))
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
    argparser.add_argument("--dataset", type=str, default="cad_pac_wtc")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=0.0007)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=0.005,
                           help="Weight for L2 loss")
    argparser.add_argument("--gamma", type=float, default=0.5,
                           help="weight of decoder regularization loss.")
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--adjacency_threshold", type=float, default=0.5)
    argparser.add_argument("--num-workers", type=int, default=1)
    argparser.add_argument("--chk_path", type=str, default="")
    argparser.add_argument("--config-path", type=str, default="")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
    argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
    argparser.add_argument("--load_from_checkpoints", action="store_true")
    argparser.add_argument("--kfold", type=int, default=0)
    argparser.add_argument("--skip_training", action="store_true")
    argparser.add_argument("--loocv", action="store_true")
    args = argparser.parse_args()
    pred = main(args)