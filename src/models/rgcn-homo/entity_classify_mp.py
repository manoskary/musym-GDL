"""Entity Classification with Mini Batch sampling for Music with Graph Convolutional Networks

Author : Emmanouil Karystinaios

Reference repo : https://github.com/melkisedeath/musym-GDL
"""

import argparse, gc
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.multiprocessing as mp
from dgl.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from functools import partial
import pandas as pd

from models import SAGE
import tqdm
import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import MPGD_homo_onset, toy_homo_onset

from utils import StratifiedSampler

def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['val_mask'] | g.ndata['train_mask'])
    test_g = g
    return train_g, val_g, test_g

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    if args.sample_gpu:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])   
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=dataloader_device,
        batch_size=args.batch_size,
        # shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        sampler=StratifiedSampler(train_labels, args.batch_size))

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch>3:
                iter_tput.append(len(seeds) / (time.time() - tic_step + 1e-8))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
            tic_step = time.time()
            if epoch == args.num_epochs-1 :
                path = os.path.join(os.path.dirname(__file__), "artifacts", "last_epoch_results_block_"+str(step) + ".csv")
                if not os.path.exists(os.path.join(os.path.dirname(__file__), "artifacts")):
                    os.makedirs(os.path.join(os.path.dirname(__file__), "artifacts"))
                df = pd.DataFrame({"labels" : batch_labels.detach().numpy(), "predictions" : batch_pred.argmax(dim=1).detach().numpy()}).to_csv(path)
                

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
            test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device)
            print('Test Acc: {:.4f}'.format(test_acc))



    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=3)
    argparser.add_argument('--fan-out', type=str, default='5, 10, 25')
    argparser.add_argument('--batch-size', type=int, default=512)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

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
        # Load the Homogeneous Graph
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

    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('feat')
        val_nfeat = val_g.ndata.pop('feat')
        test_nfeat = test_g.ndata.pop('feat')
        train_labels = train_g.ndata.pop('label')
        val_labels = val_g.ndata.pop('label')
        test_labels = test_g.ndata.pop('label')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
        train_labels = val_labels = test_labels = g.ndata.pop('label')

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(args, device, data)