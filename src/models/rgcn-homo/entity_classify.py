"""Entity Classification for Music with Graph Convolutional Networks

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
from models import SGC
from models import GraphSAGE as SAGE


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import MPGD_homo_onset, toy_homo_onset
from entity_classify_mp import load_reddit


def normalize(x, eps=1e-8):
    normed = (x - x.mean(axis=0) ) / (x.std(axis=0) + eps)
    return normed

def main(args):
    """
    Main Call for Node Classification with GraphSage.

    """

    # load graph data
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
            n_classes = load_info(info_path)['num_classes']
        else:
            dataset =  MPGD_homo_onset(save_path = data_dir) # select_piece = "K533-1"
            # Load the Homogeneous Graph
            g= dataset[0]
            n_classes = dataset.num_classes
    elif args.dataset == "toy":
        dataset =  toy_homo_onset()
        # dataset.visualize_graph(save_dir = os.path.join(os.path.dirname(__file__), "data"), name="toy_graph")
        g= dataset[0]
        n_classes = dataset.num_classes   
    elif args.dataset == "cora":
        dataset = dgl.data.CoraGraphDataset()
        g= dataset[0]
        n_classes = dataset.num_classes
    elif args.dataset == "reddit":
        g, n_classes = load_reddit()
    else:
        raise ValueError()


    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.ndata['label']
    # Informative label balance to choose loss (if balance is relevantly balanced change to cross_entropy loss)
    label_balance = {u : th.count_nonzero(labels == u)/labels.shape[0] for u in th.unique(labels)}
    print("The label balance is :", label_balance)

    # split dataset into train, validate, test
    
    if args.inductive:
        val_mask = g.ndata['val_mask']
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
    else:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to('cuda:%d' % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    
    # # Load the node features as a Dictionary to feed to the forward layer.
    # if args.normalize:
    #     node_features = (g.ndata['feat'])
    # else:
    #     node_features = g.ndata['feat']

    node_features = g.ndata.pop('feat')
    

    # if args.init_eweights:
    #     w = th.empty(g.num_edges())
    #     nn.init.uniform_(w)
    #     g.edata["w"] = w.to('cuda:%d' % args.gpu) if use_cuda else w
    
    # create model
    in_feats = node_features.shape[1]
    if args.gnn == "GraphSage" or args.gnn == "SAGE":
        model = SAGE(in_feats, args.num_hidden, n_classes, 
            n_layers=args.num_layers, activation=F.relu, dropout=args.dropout, aggregator_type=args.aggregator_type)
    elif args.gnn == "SGC":
        g = dgl.add_self_loop(g)
        model  = SGC(in_feats, args.num_hidden, n_classes,
             n_layers=args.num_layers, activation=F.relu, dropout=args.dropout)
    elif args.gnn == "GAT":
        g = dgl.add_self_loop(g)
        model  = SGC(in_feats, args.num_hidden, n_classes,
             n_layers=args.num_layers, activation=F.relu, dropout=args.dropout)
    else :
        raise AttributeError("The specified Graph Network is not implemented.")
    if use_cuda:
        model.cuda()
    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model(g, node_features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx]) 
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        with th.no_grad():
            model.eval()
            train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
            val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
            val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()
    # Saving Model for later
    # if args.model_path is not None:
    #     th.save(model.state_dict(), args.model_path)

    model.eval()
    with th.no_grad():
        logits = model.forward(g, node_features)

        test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}| " .format(test_acc, test_loss.item()))
    print()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", '--dataset', type=str, default='reddit')
    argparser.add_argument("-a", '--gnn', type=str, default='GraphSage')
    argparser.add_argument('--num-epochs', type=int, default=30)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    argparser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    # argparser.add_argument("--init_eweights", default=True, type=bool, help="Initialize edge weights")
    args = argparser.parse_args()
    
    print(args)
    main(args)