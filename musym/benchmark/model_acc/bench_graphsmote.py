import dgl
from dgl.nn.pytorch import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from musym.benchmark import utils
from musym.models.rgcn_homo.GraphSMOTE.models import GraphSMOTE



def evaluate(model, g, features, adj, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features, adj, labels)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels) * 100

@utils.benchmark('acc')
@utils.parametrize('data', ['cora', 'pubmed'])
def track_acc(data):
    data = utils.process_data(data)
    device = utils.get_bench_device()

    g = data[0].to(device)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    in_feats = features.shape[1]
    n_classes = data.num_classes

    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # create model
    model = GraphSMOTE(in_feats, 16, n_classes, 1, F.relu, 0.5)
    loss_fcn = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.train()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 weight_decay=5e-4)

    adj = g.adj(ctx=device).to_dense()
    for epoch in range(200):
        # Compute loss and prediction
        batch_pred, upsampl_lab, embed_loss = model(g, features, adj, labels)
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], upsampl_lab[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = evaluate(model, g, features, adj, labels, test_mask)
    return acc


if __name__ == '__main':
    print(track_acc("pubmed"))