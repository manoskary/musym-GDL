import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from musym.benchmark import utils
from musym.models.rgcn_homo.models import SAGE


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, batch_size, device, num_workers):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, inputs, device, batch_size, num_workers)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid])


def load_subtensor(g, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_labels = g.ndata['labels'][seeds].to(device)
    return batch_inputs, batch_labels


@utils.benchmark('acc', 600)
@utils.parametrize('data', ['ogbn-products', "reddit"])
def track_acc(data):
    data = utils.process_data(data)
    device = utils.get_bench_device()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    in_feats = g.ndata['features'].shape[1]
    n_classes = data.num_classes

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()

    num_epochs = 20
    num_hidden = 16
    num_layers = 2
    fan_out = '5,10'
    batch_size = 1024
    lr = 0.003
    dropout = 0.5
    num_workers = 4

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # dry run one epoch
    for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
        # Load the input features as well as output labels
        blocks = [block.int().to(device) for block in blocks]
        batch_inputs = blocks[0].srcdata['features']
        batch_labels = blocks[-1].dstdata['labels']

        # Compute loss and prediction
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training loop
    for epoch in range(num_epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels']

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_g = g
    test_nid = torch.nonzero(
        ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    test_acc = evaluate(
        model, test_g, test_g.ndata['features'], test_g.ndata['labels'], test_nid, batch_size, device, num_workers)

    return test_acc.item()


if __name__ == '__main__':
    print(track_acc("reddit"))