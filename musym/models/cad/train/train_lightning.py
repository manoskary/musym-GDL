from musym.models.cad.models import Node2vecModel, CadModelLightning, CadDataModule, positional_encoding
from pytorch_lightning import Trainer
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import dgl
import torch
from datetime import datetime
import os
from musym.utils import load_and_save, min_max_scaler
from pytorch_lightning.callbacks import EarlyStopping


def main(args):
    """
    Main Call for Node Classification with Node2Vec + GraphSMOTE + DBNN.
    """
    # --------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)
    # if args.config_path != "" and os.path.exists(args.config_path) and args.config_path.endswith(".json"):
    #     import json
    #     with open(args.config_path, "r") as f:
    #         config_update = json.load(f)
    #     for key in config.keys():
    #         config = {k: (config_update[k]["value"] if k in config_update.keys() else v) for k,v in config.items()}

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

    # ------------ Pre-Processing Node2Vec ----------------------
    emb_path = os.path.join(config["data_dir"], config["dataset"], "node_emb.pt")
    nodes = g.nodes()
    if config["preprocess"]:
        nodes_train, y_train = nodes[train_nids], labels[train_nids]
        nodes_val, y_val = nodes[val_nids], labels[val_nids]
        eval_set = [(nodes_train, y_train), (nodes_val, y_val)]
        pp_model = Node2vecModel(g=g, embedding_dim=64, walk_length=20, p=0.25, q=4.0, num_walks=10, device=device, eval_set=eval_set, eval_steps=1)
        pp_model.train(epochs=30, batch_size=512)
        node_emb = pp_model.embedding().detach().cpu()
        node_features = torch.cat((node_features, node_emb), dim=1)
        torch.save(node_features, emb_path)

    try:
        node_features = torch.load(emb_path)
        print("Loaded Node features of size {}.".format(node_features.shape[1]))
    except:
        print("Node embedding was not found continuing with standard node features.")
    node_features = min_max_scaler(node_features)
    # create model
    datamodule = CadDataModule(
        g=g, n_classes=n_classes, in_feats=node_features.shape[1],
        train_nid=train_nids, val_nid=val_nids, test_nid=test_nids,
        data_cpu=args.data_cpu, fan_out=fanouts, batch_size=config["batch_size"],
        num_workers=config["num_workers"])
    model = CadModelLightning(
        node_features=node_features, labels=labels,
        in_feats=datamodule.in_feats, n_hidden=config["num_hidden"],
        n_classes=datamodule.n_classes, n_layers=config["num_layers"],
        activation=F.relu, dropout=config["dropout"], lr=config["lr"],
        loss_weight=config["gamma"], ext_mode=config["ext_mode"], weight_decay=config["weight_decay"],
        adj_thresh=config["adjacency_threshold"])

    # Train
    dt = datetime.today()
    dt_str = "{}.{}.{}.{}.{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    checkpoint_callback = ModelCheckpoint(
        dirpath="./cad_checkpoints/GraphSMOTE-{}-{}".format(config["num_layers"], dt_str),
        monitor='val_fscore_epoch',
        mode="max",
        save_top_k=5,
        save_last=True,
        filename='{epoch}-{val_fscore_epoch:.2f}-{train_loss:.2f}'
    )
    # early_stopping = EarlyStopping('val_fscore', mode="max", patience=10)
    trainer = Trainer(gpus=args.num_gpus,
                      auto_select_gpus=True,
                      max_epochs=config["num_epochs"],
                      logger=WandbLogger(
                          project="Cad Learning",
                          group=config["dataset"],
                          job_type="GraphSMOTE+preprocessing+pos_enc+thresh",
                          name="Net-({})x{}_lr={:.04f}_bs={}_lw={:.04f}".format(
                              config["fan_out"], config["num_hidden"],
                              config["lr"], config["batch_size"], config["gamma"])
                      ),
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)

    # Test
    pred = trainer.predict(datamodule=datamodule)
    return pred



if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='Cadence Learning GraphSMOTE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("--num-gpus", type=int, default=4)
    argparser.add_argument("--dataset", type=str, default="cad_feature_quartets")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=256)
    argparser.add_argument('--num-layers', type=int, default=4)
    argparser.add_argument('--lr', type=float, default=0.006746163249465165)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=0.006879667132561756,
                           help="Weight for L2 loss")
    argparser.add_argument("--gamma", type=float, default=0.005611685767546064,
                           help="weight of decoder regularization loss.")
    argparser.add_argument("--ext-mode", type=str, default=None, choices=["None", "lstm", "attention"])
    argparser.add_argument("--fan-out", type=str, default='5,10,15,25')
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=256)
    argparser.add_argument("--adjacency_threshold", type=float, default=0.5)
    argparser.add_argument("--num-workers", type=int, default=10)
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
    argparser.add_argument("--preprocess", action="store_true", help="Train and store graph embedding")
    argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
    argparser.add_argument("--load-model", action="store_true", help="Load pretrained model.")
    argparser.add_argument("--eval", action="store_true", help="Preview Results on Validation set.")
    argparser.add_argument("--config-path", type=str, default="")
    # argparser.add_argument("--add_PE", action="store_true", help="Preview Results on Validation set.")
    args = argparser.parse_args()
    prediction = main(args)