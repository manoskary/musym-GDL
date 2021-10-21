"""Entity Classification with Mini Batch sampling for Music with Graph Convolutional Networks

Author : Emmanouil Karystinaios
edited Mini Baching hyparams and added schedule lr
Reference repo : https://github.com/melkisedeath/musym-GDL
"""

import argparse
import numpy as np
import time
import os, sys
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import tqdm

import pandas as pd
import tqdm

from models import SAGE


# Hyperparam Tuning and Logging
from ray import tune
from ray.tune.integration.wandb import wandb_mixin
import wandb



PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import *




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
    train_g = g.subgraph(g.ndata['train_mask'].type(th.int64))
    val_g = g.subgraph(g.ndata['val_mask'].type(th.int64))
    test_g = g.subgraph(g.ndata['train_mask'].type(th.int64))
    return train_g, val_g, test_g

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device, config):
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
        pred = model.inference(g, nfeat, device, config["batch_size"], config["num_workers"])
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
@wandb_mixin
def run(config, device, data):
    # Unpack data
    n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
    val_nfeat, val_labels, test_nfeat, test_labels = data
    in_feats = train_nfeat.shape[1]
    train_nid = th.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = th.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = th.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    dataloader_device = th.device('cpu')
    if config["sample_gpu"]:
        train_nid = train_nid.to(device)
        # copy only the csc to the GPU
        train_g = train_g.formats(['csc'])
        train_g = train_g.to(device)
        dataloader_device = device

    # Create PyTorch DataLoader for constructing blocks
    if isinstance(config["fan_out"], str):
        fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    else :
        fanouts = config["fan_out"]
    graph_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # TODO fix correct sampler formulation
    if n_classes == 2:
        # For Imbalanced Binary Labels
        weights = th.ones(train_nfeat.shape[0])
        true_idx = th.nonzero(train_labels)
        false_idx = (train_labels == 0).nonzero()
        weights[true_idx] = true_idx.shape[0]/train_labels.shape[0] 
        weights[false_idx] = false_idx.shape[0]/train_labels.shape[0] 
        # sampler = th.utils.data.WeightedRandomSampler(weights)
    else :
        sampler = None

    # TODO create a new dataloader with weighted sampling or review random walk DRNE
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        graph_sampler,
        device=dataloader_device,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=config["num_workers"]
        )

    # Define model and optimizer
    model = SAGE(in_feats, config["num_hidden"], n_classes, config["num_layers"], F.relu, config["dropout"])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # Training loop
    avg = 0
    iter_tput = []
    wandb.watch(model, criterion, log="all", log_freq=10)
    for epoch in range(config["num_epochs"]):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(tqdm.tqdm(dataloader, position=0, leave=True)):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = criterion(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step + 1e-8))

        acc = compute_acc(batch_pred, batch_labels)
        gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
        print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))
        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))


        if epoch >= 5:
            avg += toc - tic

        eval_acc = evaluate(model, val_g, val_nfeat, val_labels, val_nid, device, config)
        print('Eval Acc {:.4f}'.format(eval_acc))
        tune.report(mean_loss=loss.item())
        wandb.log({"train_accuracy": acc.item(), "train_loss": loss.item(), "val_accuracy": eval_acc})

        scheduler.step(eval_acc)
        if epoch % 5 == 0 and epoch != 0:
            test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device, config)
            wandb.log({"test_accuracy": test_acc})
            print('Test Acc: {:.4f}'.format(test_acc))
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    test_acc = evaluate(model, test_g, test_nfeat, test_labels, test_nid, device, config)
    print('Test Acc: {:.4f}'.format(test_acc))

@wandb_mixin
def main(config):
    # Pass parameters to create experiment

    config["num_layers"] = len(config["fan_out"])

    wandb.run.name = str("SageMP-{}x{}-bs={}".format(config["num_layers"], config["num_hidden"],
                                                                            config["batch_size"]))


    use_cuda = config["gpu"] >= 0 and th.cuda.is_available()

    if use_cuda:
        device = th.device('cuda:%d' % config["gpu"])
    else:
        device = th.device('cpu')
    # load graph data
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

    if config["add_self_loop"]:
        g = dgl.add_self_loop(g)

    if config["init_eweights"]:
        w = th.empty(g.num_edges())
        nn.init.uniform_(w)
        g.edata["w"] = w

    # Hack to track node idx in dataloader's subgraphs.
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']
    train_g = g.subgraph(th.nonzero(train_mask)[:, 0])
    train_labels = train_g.ndata['label']
    eids = th.arange(train_g.number_of_edges())


    # Validation and Testing
    val_g = g.subgraph(th.nonzero(val_mask)[:, 0])
    val_labels = val_g.ndata['label']
    test_g = g.subgraph(th.nonzero(test_mask)[:, 0])
    test_labels = test_g.ndata['label']

    # Features
    train_nfeat = train_g.ndata.pop('feat')
    val_nfeat = val_g.ndata.pop('feat')
    test_nfeat = test_g.ndata.pop('feat')

    print("Number of total graph nodes : {} ".format(train_labels.shape[0]))

    if not config["data_cpu"]:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(config, device, data)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('-d', '--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='5, 5')
    argparser.add_argument('--batch-size', type=int, default=128)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1)
    argparser.add_argument('--lr', type=float, default=0.01)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--add-self-loop', action='store_true', help="Add a self loop to every node of the graph.")
    argparser.add_argument('--num-workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument("--init-eweights", type=int, default=0,
                           help="Initialize learnable graph weights. Use 1 for True and 0 for false")
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
    config = vars(args)

    use_cuda = config["gpu"] >= 0 and th.cuda.is_available()

    if use_cuda:
        device = th.device('cuda:%d' % config["gpu"])
    else:
        device = th.device('cpu')
    # load graph data
    if config["dataset"] == 'mps_onset':
        g, n_classes = load_and_save("mpgd_homo_onset", "MPGD_homo_onset")
    elif config["dataset"] =="toy":
        g, n_classes = load_and_save("toy_homo_onset")
    elif config["dataset"] == "toy01":
        g, n_classes = load_and_save("toy_01_homo")
    elif config["dataset"] == "toy02":
        g, n_classes = load_and_save("toy_02_homo")
    elif config["dataset"] == "cora":
        dataset = dgl.data.CoraGraphDataset()
        g= dataset[0]
        n_classes = dataset.num_classes
    elif config["dataset"] == "reddit":
        g, n_classes = load_reddit()
    else:
        raise ValueError()

    if config["add_self_loop"]:
        g = dgl.add_self_loop(g)

    if config["init_eweights"]:
        w = th.empty(g.num_edges())
        nn.init.uniform_(w)
        g.edata["w"] = w



    if config["inductive"]:
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

    print("Number of total graph nodes : {} ".format(train_labels.shape[0]))

    if not config["data_cpu"]:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)

    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
           val_nfeat, val_labels, test_nfeat, test_labels

    run(config, device, data)