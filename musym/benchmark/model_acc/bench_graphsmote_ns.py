import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from musym.benchmark import utils

from musym.models.rgcn_homo.GraphSMOTE.models import GraphSMOTE


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, labels, batch_size, device):
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
        pred = model.inference(g, device, batch_size)
    model.train()
    return compute_acc(pred, labels)

def apply_minority_sampling(labels, n_classes, imbalance_ratio=0.3):
    N = int(n_classes/2)
    label_weights = torch.full((len(labels), 1), 1 - imbalance_ratio).squeeze().type(torch.double)
    for i in range(N):
        label_weights = torch.where(labels == i, imbalance_ratio, label_weights)
    return label_weights

@utils.benchmark('acc', 600)
@utils.parametrize('data', ['ogbn-products', "reddit"])
def track_acc(data):
    data = utils.process_data(data)
    device = utils.get_bench_device()
    g = data[0]
    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()

    num_epochs = 50
    num_hidden = 32
    num_layers = 2
    fan_out = '5,10'
    batch_size = 1024
    lr = 0.003
    dropout = 0.5
    num_workers = 4

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    train_g = g.subgraph(train_nid)
    train_g.ndata['idx'] = torch.tensor(range(train_g.number_of_nodes()))
    eids = torch.arange(train_g.number_of_edges())

    # Create PyTorch DataLoader for constructing blocks
    graph_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in fan_out.split(',')])

    # label_weights = apply_minority_sampling(train_g.ndata["label"], n_classes, imbalance_ratio=0.3)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(label_weights, batch_size, replacement=False)

    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        eids,
        graph_sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        # sampler=sampler
    )

    # Define model and optimizer
    model = GraphSMOTE(in_feats, num_hidden, n_classes, num_layers, F.relu, dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # dry run one epoch
    for step, (input_nodes, sub_g, blocks) in enumerate(dataloader):
        # Load the input features as well as output labels
        blocks = [block.int().to(device) for block in blocks]
        batch_inputs = blocks[0].srcdata['feat']
        batch_labels = blocks[-1].dstdata['label']
        adj = sub_g.adj(ctx=device).to_dense()
        # Compute loss and prediction
        batch_pred, upsampl_lab, embed_loss = model(blocks, batch_inputs, adj, batch_labels)
        loss = loss_fcn(batch_pred, upsampl_lab) + embed_loss.to(device) * 0.000001
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training loop
    for epoch in range(num_epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, sub_g, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']
            adj = sub_g.adj(ctx=device).to_dense()

            # Compute loss and prediction
            pred, upsampl_lab, embed_loss = model(blocks, batch_inputs, adj, batch_labels)
            loss = loss_fcn(pred, upsampl_lab) + embed_loss.to(device) * 0.000001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_g = g.subgraph(torch.nonzero(~g.ndata["train_mask"], as_tuple=True)[0])
    test_acc = evaluate(
        model, test_g, test_g.ndata['label'], batch_size, device)

    return test_acc.item()


if __name__ == '__main__':
    print(track_acc("reddit"))