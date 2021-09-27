import argparse
import numpy as np
import time
import os, sys
from models import *


def main(args):
    """
    Main Call for Node Classification with GraphSage.

    """

    config = args if isinstance(args, dict) else vars(args)

    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    n_classes = dataset.num_classes


    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    labels = g.ndata['label']


    # check cuda
    use_cuda = config["gpu"] >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(config["gpu"])
        g = g.to('cuda:%d' % config["gpu"])
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    node_features = g.ndata['feat']


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
        m = model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = GaugLoss(config["beta"])

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(config["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model(g, node_features)
        loss = criterion(logits[train_idx], labels[train_idx], model.adj, model.edge_pred)
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        with th.no_grad():
            model.eval()
            train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
            val_loss = criterion(logits[val_idx], labels[val_idx], model.adj, model.edge_pred)
            val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print(
            "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
            format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()

    model.eval()
    with th.no_grad():
        logits = model.forward(g, node_features)
        test_loss = criterion(logits[test_idx], labels[test_idx])
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}| ".format(test_acc, test_loss.item()))
    print()

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--alpha", type=float, default=1)
    argparser.add_argument("--beta", type=float, default=0.5)
    argparser.add_argument("--temperature", type=float, default=0.2)
    args = argparser.parse_args()

    print(args)
    main(args)