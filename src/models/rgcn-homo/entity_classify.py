"""Entity Classification for Music with Relational Graph Convolutional Networks

Author : Emmanouil Karystinaios

Reference repo : https://github.com/melkisedeath/musym-GDL
"""
import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import dgl
import dgl.nn as dglnn
from dgl import load_graphs
from dgl.data.utils import load_info

from models import SAGE, SGC


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import MPGD_homo_onset


def normalize(x, eps=1e-8):
    normed = (x - x.mean(axis=0) ) / (x.std(axis=0) + eps)
    return normed

def main(args):
    """
    Main Call for Node Classification with RGCN on Mozart Data.

    """

    # load graph data (from submodule repo)
    if args.dataset == 'mps_onset':
        print("Loading Mozart Sonatas For Cadence Detection")
        data_dir = os.path.abspath("./data")
        if os.path.exists(data_dir):
            name = "mpgd_homo_onset"
            # load processed data from directory `self.save_path`
            graph_path = os.path.join(data_dir, name, name + '_graph.bin')
            # Load the Homogeneous Graph
            g = load_graphs(graph_path)[0][0]
            info_path = os.path.join(data_dir, name, name + '_info.pkl')
            num_classes = load_info(info_path)['num_classes']
        else:
            dataset =  MPGD_homo_onset(save_path = data_dir) # select_piece = "K533-1"
            # Load the Homogeneous Graph
            g= dataset[0]
            num_classes = dataset.num_classes

    else:
        raise ValueError()

    # category = dataset.predict_category
    # num_classes = dataset.num_classes
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.ndata['labels']
    # Informative label balance to choose loss (if balance is relevantly balanced change to cross_entropy loss)
    label_balance = {u : th.count_nonzero(labels == u)/labels.shape[0] for u in th.unique(labels)}
    print("The label balance is :", label_balance)

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to('cuda:%d' % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    
    # Load the node features as a Dictionary to feed to the forward layer.
    if args.normalize:
        node_features = normalize(g.ndata['feature'])
    else:
        node_features = g.ndata['feature']

    if args.init_eweights:
        w = th.empty(g.num_edges())
        nn.init.uniform_(w)
        g.edata["w"] = w
    
    # create model
    in_feats = node_features.shape[1]
    model = SAGE(in_feats, args.n_hidden, num_classes,
        num_hidden_layers=args.n_layers - 2, aggr_type="pool")
    if use_cuda:
        model.cuda()
    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model(g, node_features)
        # loss = softmax_focal_loss(logits[train_idx], labels[train_idx]) 
        loss = F.cross_entropy(logits[train_idx], labels[train_idx]) 
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        # val_loss = softmax_focal_loss(logits[val_idx], labels[val_idx])
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward(g, node_features)
    # test_loss = softmax_focal_loss(logits[test_idx], labels[test_idx])
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}| " .format(test_acc, test_loss.item()))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--normalize", default=False, required=False, type=bool,
            help="Normalize with 0 mean and unit variance")
    parser.add_argument("--init_eweights", default=True, type=bool, help="Initialize edge weights")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    
    print(args)
    main(args)