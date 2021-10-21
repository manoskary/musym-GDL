"""Entity Classification with Mini Batch sampling for Music with Graph Convolutional Networks

Author : Emmanouil Karystinaios
edited Mini Baching hyparams and added schedule lr
Reference repo : https://github.com/melkisedeath/musym-GDL
"""

import argparse

import dgl.dataloading
import torch
import torch.nn as nn
import tqdm
from dgl.nn import SAGEConv, GraphConv


class MyModel(nn.Module):
    def __init__(self, in_feats, num_classes):
        super(MyModel, self).__init__()
        self.conv = SAGEConv(in_feats, num_classes, aggregator_type="gcn")

    def forward(self, mfgs, x):
        # Lines that are changed are marked with an arrow: "<---"
        h = self.conv(mfgs[0], x)  # <---  # <---
        return h

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
    g = dgl.add_self_loop(g)
    n_classes = dataset.num_classes
    train_nid = torch.tensor(range(g.num_nodes())).type(torch.int64)
    in_feats = g.ndata["feat"].shape[1]

    # --------------- Transfer to Devise ---------------
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:%d' % config["gpu"])
        dataloader_device = device
    else:
        device = torch.device('cpu')
        train_nid = train_nid.to(device)
        dataloader_device = device
    # ---------------- Sampler Definition ---------------

    # Graph Sampler takes all available neighbors
    graph_sampler = dgl.dataloading.MultiLayerNeighborSampler([5])

    # Balance Label Sampler
    label_weights = get_sample_weights(g.ndata["label"])
    # Torch Sampler
    sampler = torch.utils.data.sampler.WeightedRandomSampler(label_weights, len(label_weights))


    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        graph_sampler,
        device=dataloader_device,
        batch_size=config["batch_size"],
        drop_last=False,
        num_workers=0,
        sampler=sampler
        )

    # Define model and optimizer
    model = MyModel(in_feats, n_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(tqdm.tqdm(dataloader, position=0, leave=True)):
            # Load the input features as well as output labels
            batch_inputs = (blocks[0].srcdata["feat"], blocks[0].dstdata["feat"])
            batch_labels = blocks[-1].dstdata['label']
            print(step, blocks[0].num_nodes(), batch_labels.shape)
            # Predict and loss
            batch_pred = model(blocks, batch_inputs)
            loss = criterion(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Weighted Sampling SAGE')
    argparser.add_argument('--gpu', type=int, default=-1,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('-d', '--dataset', type=str, default='toy01')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--batch-size', type=int, default=32)

    args = argparser.parse_args()
    config = vars(args)

    main(config)