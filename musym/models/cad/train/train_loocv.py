import json

from musym.models.cad.models import LOOLoop, CadLoocvDataModule, CadModelLightning
from pytorch_lightning import Trainer
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import dgl
import torch
from datetime import datetime
import os, json
from musym.utils import load_and_save, min_max_scaler
from pytorch_lightning.callbacks import EarlyStopping
from musym.models.cad.train.cad_learning import postprocess, to_sequences
from sklearn.metrics import f1_score, confusion_matrix
import wandb


def prepare_and_postprocess(g, model, batch_size, train_nids, val_nids, labels, node_features, score_duration, piece_idx, onsets, device):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(model.module.n_layers)
    eval_model = model.module.to(device)
    train_loader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler, batch_size=batch_size)
    val_loader = dgl.dataloading.NodeDataLoader(g, val_nids, sampler, batch_size=batch_size)
    train_prediction = eval_model.inference(train_loader, node_features, labels, device)[train_nids]
    val_prediction = eval_model.inference(val_loader, node_features, labels, device)[val_nids]
    X_train, y_train, train_fscore, _ , _ = to_sequences(labels, train_prediction.detach().cpu(), train_nids, score_duration, piece_idx, onsets)
    X_val, y_val, val_fscore, test_predictions, test_labels = to_sequences(labels, val_prediction.detach().cpu(), val_nids, score_duration, piece_idx, onsets)
    postprocess_val_acc, postprocess_val_f1 = postprocess(X_train, y_train, X_val, y_val)
    return postprocess_val_acc, postprocess_val_f1, val_fscore, test_predictions, test_labels

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


def train(data, args):
    g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
    piece_idx, onsets, score_duration, device, dataloader_device, fanouts, config = data


    datamodule = CadLoocvDataModule(
        dataset=args.dataset, data_dir=args.data_dir,
        data_cpu=args.data_cpu, fan_out=fanouts, batch_size=config["batch_size"],
        num_workers=config["num_workers"], use_ddp=args.num_gpus > 1)

    model = CadModelLightning(
        node_features=node_features, labels=labels,
        in_feats=80, n_hidden=config["num_hidden"],
        n_classes=2, n_layers=config["num_layers"],
        activation=F.relu, dropout=config["dropout"], lr=config["lr"],
        loss_weight=config["gamma"], ext_mode=config["ext_mode"], weight_decay=config["weight_decay"],
        adj_thresh=config["adjacency_threshold"])
    model_name = "Net"
    model_name = "{}-({})x{}_lr={:.04f}_bs={}_lw={:.04f}".format(
        model_name,
        config["fan_out"], config["num_hidden"],
        config["lr"], config["batch_size"], config["gamma"])

    wandb.init(
        project="Cad Learning",
        group=config["dataset"],
        job_type="GraphSMOTE_LOOCV_{}x{}".format(config["num_layers"], config["num_hidden"]),
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
    wandb_logger = WandbLogger(
        project="Cad Learning",
        group=config["dataset"],
        job_type="GraphSMOTE_LOOCV_{}x{}".format(config["num_layers"], config["num_hidden"]),
        name=model_name,
        reinit=True
    )
    trainer = Trainer(gpus=args.num_gpus,
                      auto_select_gpus=True,
                      max_epochs=config["num_epochs"],
                      logger=wandb_logger,
                      callbacks=[checkpoint_callback],
                      num_sanity_val_steps=0
                      )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = LOOLoop(11, export_path="./cad_checkpoints/{}-LOOCV".format(model_name))
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=datamodule)

    if args.postprocess:
        model.freeze()
        postprocess_val_acc, postprocess_val_f1, binary_val_fscore, test_predictions, test_labels = prepare_and_postprocess(g, model, config["batch_size"],
                                                                train_nids, test_nids,
                                                                labels, node_features,
                                                                score_duration, piece_idx,
                                                                onsets, device)

        print("Positive Class Val Fscore : ", binary_val_fscore)
        wandb.log({"positive_class_val_fscore": binary_val_fscore,
                   "Postprocess Onset-wise Val Accuracy" : postprocess_val_acc,
                   "Postprocess Onset-wise Val F1": postprocess_val_f1,
                   })
        return test_predictions, test_labels


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
    test_nids = torch.nonzero(g.ndata.pop('test_mask'), as_tuple=True)[0]
    # check cuda
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % torch.cuda.current_device() if use_cuda else 'cpu')
    dataloader_device = "cpu"


    node_features = min_max_scaler(node_features)
    # create model
    data = g, n_classes, labels, train_nids, val_nids, test_nids, node_features, \
           piece_idx, onsets, score_duration, device, dataloader_device, fanouts, config
    train(data, args)


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='Cadence Learning GraphSMOTE LOOCV')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("--num-gpus", type=int, default=1)
    argparser.add_argument("--dataset", type=str, default='cad_pac_wtc')
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=5e-3)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--gamma", type=float, default=0.1,
                           help="weight of decoder regularization loss.")
    argparser.add_argument("--ext-mode", type=str, default=None, choices=["None", "lstm", "attention"])
    argparser.add_argument("--fan-out", type=str, default='5,10')
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=4096)
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
    argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
    args = argparser.parse_args()
    pred = main(args)