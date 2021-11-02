import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import *

class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.gcn = gcn
        self.weight = nn.Linear(self.input_size if self.gcn else 2 * self.input_size, out_size)
        self.fc_self = nn.Linear(self.input_size, self.input_size, bias=False)
        self.init_params()
        # TODO add inference function

    def init_params(self):
        gain= nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.weight.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

    def aggregate(self, neigh_features):
        aggregate_feats = torch.zeros((len(neigh_features), self.input_size))
        for i, neigh in enumerate(neigh_features):
            aggregate_feats[i] = torch.max(self.fc_self(neigh), 0)[0]
        return aggregate_feats

    def forward(self, block, node_features):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if isinstance(block, tuple):
            src_idx, dst_idx = block
        nfeats = node_features[:src_idx]
        neigh_features = [node_features[nids] for nids in dst_idx]
        aggregate_feats = self.aggregate(neigh_features)
        if not self.gcn:
            combined = torch.cat([nfeats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = self.weight(combined)
        return combined


class GraphSage(nn.Module):
    """Simple GraphSage"""

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GraphSage, self).__init__()
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(SageLayer(in_feats, n_hidden, gcn=False))
            for i in range(1, n_layers - 1):
                self.layers.append(SageLayer(in_feats, n_classes, gcn=False))
            self.layers.append(SageLayer(n_hidden, n_classes, gcn=False))
        else:
            self.layers.append(SageLayer(in_feats, n_classes, gcn=False))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, node_features):
        h = node_features
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = F.normalize(h)
                h = self.dropout(h)
        return h


class MyDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, graph, labels, node_features, num_layers, k=5):
        'Initialization'
        self.g = graph
        self.adj = graph.adj().to_dense().type(torch.uint8)
        self.labels = labels
        self.list_IDs = torch.tensor(range(graph.num_nodes())).type(torch.int64)
        self.node_features = node_features
        self.num_layers = num_layers
        self.k = k


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # Load data and get label
        # blocks, batch_feats = self.get_blocks(ID)
        X = ID
        y = self.labels[ID]
        # return blocks, batch_feats, y
        return X, y

    # def get_blocks(self, idx):
    #     blocks = list()
    #     b_idx = copy.copy(idx)
    #     for i in range(self.num_layers - 1):
    #         # Neighbors for every layer
    #         neighs = torch.tensor(random.choices(torch.squeeze(torch.nonzero(self.adj[b_idx]), dim=1), k=self.k))
    #
    #         b_idx = torch.cat([torch.tensor([idx]), neighs])
    #         h = self.node_features[b_idx]
    #     return blocks, h

def fetch_neigh_features(batch_nids, adj, node_features, num_layers):
    blocks = list()
    inc = 0
    idx = list()
    layer = list()
    b_idx = copy.copy(batch_nids)
    # For every layer
    for i in range(num_layers):
        # Neighbors for every layer
        neighs = [torch.squeeze(torch.nonzero(row), dim=1) for row in adj[batch_nids]]
        src_pos = (inc, len(batch_nids)+inc)

        length_list = list(map(lambda x: len(x), neighs))
        dst_pos = [(src_pos[1] + inc + sum(length_list[:n]), src_pos[1] + inc + sum(length_list[:n + 1])) for n in
                    range(len(length_list))]
        inc = src_pos[1] + inc + sum(length_list)
        layer.append({"src": src_pos, "dst" : dst_pos})
        # TODO batch_nids is added multiple times
        idx += [batch_nids] + neighs
    neigh_ids, rev_idx = torch.unique(torch.cat(idx), return_inverse=True)
    # Put the input of every layer on top
    resorted_idx = torch.zeros(len(neigh_ids))
    resorted_idx[:len(batch_nids)] = neigh_ids[rev_idx[:len(batch_nids)]]
    resorted_idx[len(batch_nids):] = neigh_ids[torch.where(neigh_ids not in neigh_ids[rev_idx[:len(batch_nids)]])]

    for i in range(num_layers):
        src_pos = layer[i]["src"]
        dst_pos = layer[i]["dst"]
        n_src = rev_idx[src_pos[0] : src_pos[1]]
        n_dst = list(map(lambda x: rev_idx[x[0]:x[1]], dst_pos))
        blocks.append((n_src, n_dst))

    neigh_features = node_features[neigh_ids]
    return blocks, neigh_features


def get_sample_weights(labels):
    class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    sample_weights = torch.tensor([weight[t] for t in labels])
    return sample_weights


def get_blocks(batch_nids, adj, node_features, num_layers):
    blocks = list()
    b_idx = copy.copy(batch_nids)
    for i in range(num_layers - 1):
       # Neighbors for every layer
        neighs = [torch.squeeze(torch.nonzero(row), dim=1) for row in adj[b_idx]]
        inc = 0
        lens = torch.zeros((len(neighs), 2)).type(torch.int32)
        for j, neigh in enumerate(neighs):
            lens[j][0] = inc
            lens[j][1] = inc + len(neigh)
            inc += len(neigh)

        src_nodes = len(b_idx)
        nidx, revidx = torch.unique(torch.cat(neighs), return_inverse=True)

        blocks.append((src_nodes, [revidx[lens[i, 0]: lens[i, 1]] for i in range(len(neighs))]))
        idx = [batch_nids, nidx]
        b_idx = torch.cat(idx)
        h = node_features[b_idx]
    return blocks, h

def main(config):
    """Pass parameters to create experiment"""

    # --------------- Dataset Loading -------------------------
    dataset = dgl.data.CoraGraphDataset()
    n_classes = dataset.num_classes
    g = dataset[0]
    # g, n_classes = load_and_save("cad_basis_homo", "./data/")

    adj = g.adj().to_dense().type(torch.uint8)
    in_feats = g.ndata["feat"].shape[1]
    node_features = g.ndata["feat"]
    dataset = MyDataset(g, g.ndata["label"], node_features, num_layers=config["num_layers"])

    # --------------- Transfer to Devise ---------------
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:%d' % config["gpu"])
    else:
        device = torch.device('cpu')
    # ---------------- Sampler Definition ---------------
    # Balance Label Sampler
    label_weights = get_sample_weights(g.ndata["label"])
    # Torch Sampler
    sampler = torch.utils.data.sampler.WeightedRandomSampler(label_weights, len(label_weights))


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = config["batch_size"],
        shuffle=False,
        sampler=sampler,
    )

    # Define model and optimizer
    model = GraphSage(in_feats=in_feats, n_hidden=16, n_classes=n_classes, n_layers=config["num_layers"], activation=F.relu, dropout=0.5)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Loop over the dataloader to sample Node Ids
        # for step, (blocks, neigh_features, batch_labels) in enumerate(tqdm(dataloader, position=0, leave=True)):
        for step, (batch_nids, batch_labels) in enumerate(tqdm(dataloader, position=0, leave=True)):
            # Fetch the neighbor features
            blocks, neigh_features = get_blocks(batch_nids, adj, node_features, config["num_layers"])
            # TODO eventually every nodeID should have a list of k distant neighbors for every kth layer
            # TODO we can add sampling max(m) neighbors instead of fetching all of them similar to dgl.samplers/
            # Predict and loss
            # TODO move features to GPU option
            batch_pred = model(blocks, neigh_features)
            loss = criterion(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (torch.argmax(batch_pred, dim=1) == batch_labels.long()).float().sum() / len(batch_pred)
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), acc.item(), gpu_mem_alloc))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Weighted Sampling SAGE')
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('-d', '--dataset', type=str, default='toy01')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--batch-size', type=int, default=256)

    args = argparser.parse_args()
    config = vars(args)

    main(config)