import numpy as np
import time
import os, sys
from models import *

# Hyperparam Tuning and Logging
from ray import tune
from ray.tune.integration.wandb import wandb_mixin
import wandb

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import *

@wandb_mixin
def main(args):
    """
    Main Call for Node Classification with Gaug.

    """
    #--------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    config["num_layers"] = len(config["fan_out"])

    wandb.run.name = str("Gaug-" + str(config["num_layers"]) + "x" + str(
        config["num_hidden"]) + " --lr " + str(config["lr"]) + " --dropout " + str(config["dropout"]) + "-lr_scheduler")

    # --------------- Dataset Loading -------------------------
    if config["dataset"] == 'mps_onset':
        g, n_classes = load_and_save("mpgd_homo_onset", config["data_dir"], "MPGD_homo_onset")
    elif config["dataset"] == "toy":
        g, n_classes = load_and_save("toy_homo_onset", config["data_dir"])
    elif config["dataset"] == "toy01":
        g, n_classes = load_and_save("toy_01_homo", config["data_dir"])
    elif config["dataset"] == "toy02":
        g, n_classes = load_and_save("toy_02_homo", config["data_dir"])
    elif config["dataset"] == "cora":
        dataset = dgl.data.CoraGraphDataset()
        g = dataset[0]
        n_classes = dataset.num_classes
    elif config["dataset"] == "reddit":
        g, n_classes = load_reddit()
    else:
        raise ValueError()

    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']

    # Hack to track node idx in dataloader's subgraphs.
    train_g = g.subgraph(th.nonzero(train_mask)[:, 0])
    train_g.ndata['idx'] = th.tensor(range(train_g.number_of_nodes()))
    labels = train_g.ndata['label']
    eids = th.arange(train_g.number_of_edges())
    node_features = train_g.ndata['feat']

    # Validation and Testing
    val_g = g.subgraph(th.nonzero(val_mask)[:, 0])
    val_labels = val_g.ndata['label']
    test_g = g.subgraph(th.nonzero(test_mask)[:, 0])
    test_labels = test_g.ndata['label']

    if config["init_eweights"]:
        w = th.empty(train_g.num_edges())
        nn.init.uniform_(w)
        train_g.edata["w"] = w


    # check cuda
    use_cuda = config["gpu"] >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(config["gpu"])
        device = th.device('cuda:%d' % config["gpu"])
    else:
        device = th.device('cpu')

    # create model
    in_feats = node_features.shape[1]
    model = Gaug(in_feats,
                 config["num_hidden"],
                 n_classes,
                 n_layers=config["num_layers"],
                 activation=F.relu,
                 dropout=config["dropout"],
                 alpha = config["alpha"],
                 temperature=config["temperature"],
                 use_cuda=use_cuda)
    if use_cuda:
        model = model.cuda()

    if isinstance(config["fan_out"], str):
        fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    else :
        fanouts = config["fan_out"]

    # dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    # The edge dataloader returns a subgraph but iterates on the number of edges.
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        eids,
        sampler,
        batch_size = config["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=config["num_workers"],
        persistent_workers=config["num_workers"]>0)
    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = GaugLoss(config["beta"])

    # training loop
    print("start training...")
    dur = []
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(config["num_epochs"]):
        model.train()
        t0 = time.time()
        for step, (input_nodes, sub_g, blocks) in enumerate(train_dataloader):
            batch_inputs = node_features[input_nodes].to(device)
            # Hack to track the node idx for NodePred layer (SAGE) not the same as block or input nodes
            batch_labels = labels[sub_g.ndata['idx']].to(device)
            blocks = [block.int().to(device) for block in blocks]
            # The features for the loaded subgraph
            feat_inputs = sub_g.ndata["feat"].to(device)
            # The adjacency matrix of the subgraph
            if config["init_eweights"]:
                subgraph_shape = (sub_g.num_nodes(), sub_g.num_nodes())
                subgraph_indices = th.vstack(sub_g.edges())
                adj = th.sparse.FloatTensor(subgraph_indices, sub_g.edata["w"], subgraph_shape).to_dense().to(device)
            else:
                adj = sub_g.adj(ctx=device).to_dense()

            bce_weight = th.FloatTensor([float(adj.shape[0] ** 2 - adj.sum()) / adj.sum()]).to(device)
            # Prediction of the Gaug model
            logits = model(adj, blocks, batch_inputs, feat_inputs)
            # Combined loss
            loss = criterion(logits, batch_labels, adj.view(-1), model.ep.view(-1), bce_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 5 == 0:
                train_acc = th.sum(logits.argmax(dim=1) == batch_labels).item() / batch_labels.shape[0]
                print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f}".format(epoch, train_acc, loss.item()))

            t1 = time.time()
        if epoch > 5:
            dur.append(t1 - t0)
        model.eval()
        with th.no_grad():
            train_acc = th.sum(logits.argmax(dim=1) == batch_labels).item() / batch_labels.shape[0]
            pred = model.inference(val_g, device=device, batch_size=config["batch_size"], num_workers=config["num_workers"])
            val_loss = F.cross_entropy(pred, val_labels)
            val_acc = (th.argmax(pred, dim=1) == val_labels.long()).float().sum() / len(pred)
            scheduler.step(val_acc)
            tune.report(mean_loss=loss.item())
            wandb.log({"train_accuracy": train_acc.item(), "train_loss": loss.item(), "val_accuracy": val_acc, "val_loss": val_loss})
        print(
            "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Val Acc : {:.4f} | Val CE Loss: {:.4f}| Time: {:.4f}".
            format(epoch, train_acc, loss.item(), val_acc, val_loss, np.average(dur)))
    print()

    model.eval()
    with th.no_grad():
        pred = model.inference(test_g, device=device, batch_size=config["batch_size"], num_workers=config["num_workers"])
        test_loss = F.cross_entropy(pred, test_labels)
        test_acc = (th.argmax(pred, dim=1) == test_labels.long()).float().sum() / len(pred)
        wandb.log({"test_accuracy": test_acc, test_loss})
    print("Test Acc: {:.4f} | Test loss: {:.4f}| ".format(test_acc, test_loss.item()))
    print()

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", "--dataset", type=str, default="toy_01_homo")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--batch-size", type=int, default=512)
    argparser.add_argument("--num-workers", type=int, default=0)
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--init-eweights", type=int, default=0,
                           help="Initialize learnable graph weights. Use 1 for True and 0 for false")
    argparser.add_argument("--alpha", type=float, default=1)
    argparser.add_argument("--beta", type=float, default=0.5)
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("./data/"))

    args = argparser.parse_args()

    print(args)
    main(args)