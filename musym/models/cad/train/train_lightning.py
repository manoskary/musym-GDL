import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from musym.models.cad.models import CadModelLightning, CadDataModule
from pytorch_lightning import Trainer
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import dgl
import torch
from datetime import datetime
import os
from musym.utils import load_and_save, min_max_scaler
from musym.models.cad.train.cad_learning import postprocess
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support
import wandb
import numpy as np


def to_sequences(labels, preds, idx, score_duration, piece_idx, onsets):
    seqs = list()
    trues = list()

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
    f1_score_binary = f1_score(labels.numpy(), preds.argmax(dim=1).numpy(), average="binary")
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
                    new_X.append(x.mean(dim=0))
                    # new_X.append(x.max(0)[0])
                    new_y.append(z.max().unsqueeze(0))
                else:
                    new_X.append(x)
                    new_y.append(z)
            seqs.append(torch.vstack(new_X))
            trues.append(torch.cat(new_y))
    return seqs, trues, f1_score_binary


def prepare_and_postprocess(g, model, batch_size, train_nids, val_nids, labels, node_features, score_duration, piece_idx, onsets, device):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.module.n_layers)
    eval_model = model.module.to(device)
    train_loader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler, batch_size=batch_size)
    val_loader = dgl.dataloading.NodeDataLoader(g, val_nids, sampler, batch_size=batch_size)
    train_prediction = eval_model.inference(train_loader, node_features, labels, device)[train_nids]
    val_prediction = eval_model.inference(val_loader, node_features, labels, device)[val_nids]
    X_train, y_train, _ = to_sequences(labels, train_prediction.detach().cpu(), train_nids, score_duration, piece_idx, onsets)
    X_val, y_val, val_fscore = to_sequences(labels, val_prediction.detach().cpu(), val_nids, score_duration, piece_idx, onsets)
    post_val_acc, post_val_f1 = postprocess(X_train, y_train, X_val, y_val)
    X_val = torch.vstack(X_val).numpy()
    y_val = torch.cat(y_val).numpy()
    y_pred = np.argmax(X_val, axis=1)
    thresh_val_acc = np.equal(y_pred, y_val).astype(float).sum() / len(y_val)
    thresh_pres, thresh_rec, thresh_f1, val_sup = precision_recall_fscore_support(y_val, y_pred, average="binary")
    metrics = {
        "Onset_wise Fscore": val_fscore,
        "Postprocess Beat-wise Val Accuracy": post_val_acc,
        "Postprocess Beat-wise Val F1": post_val_f1,
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


def train(scidx, data, args):
    g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
    piece_idx, onsets, score_duration, device, dataloader_device, fanouts, config = data


    datamodule = CadDataModule(
        g=g, n_classes=n_classes, in_feats=node_features.shape[1],
        train_nid=train_nids, val_nid=val_nids, test_nid=test_nids,
        data_cpu=args.data_cpu, fan_out=fanouts, batch_size=config["batch_size"],
        num_workers=config["num_workers"], use_ddp=args.num_gpus > 1)
    if config["load_from_checkpoints"]:
        model = CadModelLightning.load_from_checkpoint(
            checkpoint_path=config["chk_path"],
            node_features=node_features, labels=labels,
            in_feats=datamodule.in_feats, n_hidden=config["num_hidden"],
            n_classes=datamodule.n_classes, n_layers=config["num_layers"],
            activation=F.relu, dropout=config["dropout"], lr=config["lr"],
            loss_weight=config["gamma"], ext_mode=config["ext_mode"], weight_decay=config["weight_decay"],
            adj_thresh=config["adjacency_threshold"])
        model_name = "Pretrained-Net"
    else:
        model = CadModelLightning(
            node_features=node_features, labels=labels,
            in_feats=datamodule.in_feats, n_hidden=config["num_hidden"],
            n_classes=datamodule.n_classes, n_layers=config["num_layers"],
            activation=F.relu, dropout=config["dropout"], lr=config["lr"],
            loss_weight=config["gamma"], ext_mode=config["ext_mode"], weight_decay=config["weight_decay"],
            adj_thresh=config["adjacency_threshold"])
        model_name = "Net"
    model_name = "{}_{}-({})x{}_lr={:.04f}_bs={}_lw={:.04f}".format(
        model_name, scidx,
        config["fan_out"], config["num_hidden"],
        config["lr"], config["batch_size"], config["gamma"])

    wandb.init(
        project="Cad Learning",
        group=config["dataset"],
        job_type="GraphSMOTE_{}x{}".format(config["num_layers"], config["num_hidden"]),
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
        job_type="GraphSMOTE_{}x{}".format(config["num_layers"], config["num_hidden"]),
        name=model_name,
        reinit=True
    )
    trainer = Trainer(gpus=args.num_gpus,
                      auto_select_gpus=True,
                      max_epochs=config["num_epochs"],
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback])

    if not args.skip_training:
        trainer.fit(model, datamodule=datamodule)

    if args.postprocess:
        model.freeze()
        metrics = prepare_and_postprocess(g, model, config["batch_size"],
                                                                train_nids, test_nids,
                                                                labels, node_features,
                                                                score_duration, piece_idx,
                                                                onsets, device)

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

    fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)
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
    # ========= LOOCV TRAINING ============
    if args.kfold:
        unique_scores = torch.unique(piece_idx)
        num_folds = args.kfold
        for fold_num in range(num_folds):
            pick = torch.randperm(len(unique_scores))
            train_fold = pick[:int(len(unique_scores)/2)]
            val_fold = pick[int(len(unique_scores)/2):]
            val_nids = test_nids = torch.cat([torch.nonzero(piece_idx==scidx, as_tuple=True)[0] for scidx in val_fold])
            train_nids = torch.cat([torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in train_fold])
            data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
                   piece_idx, onsets, score_duration, device, dataloader_device, fanouts, config
            train(fold_num, data, args)
    else:
        data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
               piece_idx, onsets, score_duration, device, dataloader_device, fanouts, config
        train("", data, args)


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
    argparser.add_argument("--ext-mode", type=str, default=None, choices=["None", "lstm", "attention"])
    argparser.add_argument("--fan-out", type=str, default='10,25')
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=2048)
    argparser.add_argument("--adjacency_threshold", type=float, default=0.5)
    argparser.add_argument("--num-workers", type=int, default=10)
    argparser.add_argument("--chk_path", type=str, default="")
    argparser.add_argument("--config-path", type=str, default="")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
    argparser.add_argument("--preprocess", action="store_true", help="Train and store graph embedding")
    argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
    argparser.add_argument("--eval", action="store_true", help="Preview Results on Validation set.")
    argparser.add_argument("--load_from_checkpoints", action="store_true")
    argparser.add_argument("--kfold", type=int, default=0)
    argparser.add_argument("--skip_training", action="store_true")
    args = argparser.parse_args()
    pred = main(args)