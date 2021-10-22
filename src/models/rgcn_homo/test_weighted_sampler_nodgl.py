"""Entity Classification with Mini Batch sampling for Music with Graph Convolutional Networks

Author : Emmanouil Karystinaios
edited Mini Baching hyparams and added schedule lr
Reference repo : https://github.com/melkisedeath/musym-GDL
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import *
import dgl


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

    def init_params(self):
        gain= nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.weight.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

    def aggregate(self, neigh_features):
        aggregate_feats = torch.zeros((len(neigh_features), self.input_size))
        for i, neigh in enumerate(neigh_features):
            aggregate_feats[i] = torch.max(self.fc_self(neigh), 0)[0]
        return aggregate_feats

    def forward(self, nfeats, neigh_features, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        aggregate_feats = self.aggregate(neigh_features)
        if not self.gcn:
            combined = torch.cat([nfeats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = self.weight(combined)
        return combined


class GraphSage(nn.Module):
    """Simple GraphSage"""

    def __init__(self, in_feats, n_classes):
        super(GraphSage, self).__init__()
        self.conv = SageLayer(in_feats, n_classes, gcn=False)

    def forward(self, block_features, neigh_features):
        h = self.conv(block_features, neigh_features)
        return h


class MyDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = ID
        y = self.labels[ID]

        return X, y


def get_sample_weights(labels):
    class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
    weight = 1. / class_sample_count.float()
    sample_weights = torch.tensor([weight[t] for t in labels])
    return sample_weights

def main(config):
    """Pass parameters to create experiment"""


    # --------------- Dataset Loading -------------------------
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    adj = g.adj()
    n_classes = dataset.num_classes
    train_nid = torch.ones(g.num_nodes()).type(torch.int64)

    in_feats = g.ndata["feat"].shape[1]
    dataset = MyDataset(torch.tensor(range(g.num_nodes())).type(torch.int64), g.ndata["label"])
    node_features = g.ndata["feat"]

    # --------------- Transfer to Devise ---------------
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:%d' % config["gpu"])
        dataloader_device = device
    else:
        device = torch.device('cpu')
        train_nid = train_nid.to(device)

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
    model = GraphSage(in_feats=in_feats, n_classes=n_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
        for step, (batch_nids, batch_labels) in enumerate(tqdm(dataloader, position=0, leave=True)):
            # Load the input features as well as output labels
            neigh_features = [torch.squeeze(node_features[adj[i].coalesce().indices()]) for i in batch_nids]
            batch_features = node_features[batch_nids]
            print(step, batch_features.shape)
            # Predict and loss
            batch_pred = model(batch_features, neigh_features)
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
    argparser.add_argument('--batch-size', type=int, default=512)

    args = argparser.parse_args()
    config = vars(args)

    main(config)