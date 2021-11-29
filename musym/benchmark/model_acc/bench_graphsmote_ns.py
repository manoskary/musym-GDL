import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from musym.benchmark import utils
from sklearn.metrics import precision_recall_fscore_support
from musym.models.rgcn_homo.GraphSMOTE.models import GraphSMOTE
import wandb


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def compute_fscore(pred, labels):
    """
        Compute the accuracy of prediction given the labels.
        """
    y_true = labels.long().numpy()
    y_pred = torch.argmax(pred, dim=1).numpy()
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    return precision, recall, fscore

def compute_loss(pred, labels):
    return F.cross_entropy(pred, labels)

def evaluate(model, g, node_features, labels, batch_size, device, num_workers):
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
        pred = model.inference(g, node_features, labels, device, batch_size, num_workers)
    model.train()
    acc = compute_acc(pred, labels)
    precision, recall, fscore = compute_fscore(pred, labels)
    loss = compute_loss(pred, labels)
    return acc, precision, recall, fscore, loss


def apply_minority_sampling(labels, n_classes, imbalance_ratio=0.3):
    N = int(n_classes/2)
    label_weights = torch.full((len(labels), 1), 1 - imbalance_ratio).squeeze().type(torch.double)
    for i in range(N):
        label_weights = torch.where(labels == i, imbalance_ratio, label_weights)
    return label_weights

def partition_graph(g):
    train_mask = g.ndata.pop("train_mask").type(torch.bool)
    test_mask = ~train_mask
    train_g = g.subgraph(torch.nonzero(train_mask, as_tuple=True)[0])
    train_g.ndata['idx'] = torch.tensor(range(train_g.number_of_nodes()))
    train_nfeat = train_g.ndata.pop("feat")
    train_labels = train_g.ndata.pop("label")

    test_g = g.subgraph(torch.nonzero(test_mask, as_tuple=True)[0])
    test_nfeat = test_g.ndata.pop("feat")
    test_labels = test_g.ndata.pop("label")
    return train_g, train_nfeat, train_labels, test_g, test_nfeat, test_labels


@utils.benchmark('acc', 600)
@utils.parametrize('data', ['CoraImbalanced', "BlogCatalog"])
def track_acc(data, config=None):

    data = utils.process_data(data)
    device = utils.get_bench_device()
    g = data[0]
    in_feats = g.ndata['feat'].shape[1]
    n_classes = data.num_classes

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()

    if not config:
        config = {
            "num_epochs" : 50,
            "num_hidden" : 32,
            "num_layers" : 2,
            "fan_out" : '5,10',
            "batch_size" : 512,
            "lr" : 0.003,
            "dropout" : 0.5,
            "num_workers" : 4,
        }

    # ------------------------------- Logging INIT ----------------------------------
    wandb.init(project="bench-GNN", group=data, job_type="GraphSMOTE_sampling", config=config, reinit=True)


    train_g, train_nfeat, train_labels, test_g, test_nfeat, test_labels = partition_graph(g)

    # Create PyTorch DataLoader for constructing blocks
    if isinstance(config["fan_out"], str):
        fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    else:
        fanouts = config["fan_out"]

    graph_sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)

    # label_weights = apply_minority_sampling(train_g.ndata["label"], n_classes, imbalance_ratio=0.3)
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(label_weights, batch_size, replacement=False)

    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        torch.arange(train_g.number_of_edges()),
        graph_sampler,
        device=device,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
        # sampler=sampler
    )

    # Define model and optimizer
    model = GraphSMOTE(in_feats, config["num_hidden"], n_classes, config["num_layers"], F.relu, config["dropout"])
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # dry run one epoch
    for step, (input_nodes, sub_g, blocks) in enumerate(dataloader):
        # Load the input features as well as output labels
        batch_inputs = train_nfeat[input_nodes].to(device)
        batch_labels = train_labels[sub_g.ndata["idx"]].to(device)
        adj = sub_g.adj(ctx=device).to_dense().to(device)
        # Compute loss and prediction
        batch_pred, upsampl_lab, embed_loss = model(blocks, batch_inputs, adj, batch_labels)
        loss = loss_fcn(batch_pred, upsampl_lab) + embed_loss.to(device) * 0.000001
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Training loop
    for epoch in range(config["num_epochs"]):
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, sub_g, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs = train_nfeat[input_nodes].to(device)
            batch_labels = train_labels[sub_g.ndata["idx"]].to(device)
            adj = sub_g.adj(ctx=device).to_dense().to(device)

            # Compute loss and prediction
            pred, upsampl_lab, embed_loss = model(blocks, batch_inputs, adj, batch_labels)
            loss = loss_fcn(pred, upsampl_lab) + embed_loss.to(device) * 0.000001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_acc, test_precision, test_recall, test_fscore, test_loss = evaluate(model, test_g, test_nfeat,
                                                                                 test_labels,
                                                                                 config["batch_size"], device,
                                                                                 config["num_workers"])
        print("Test Acc: {:.4f} | Test loss: {:.4f}| Test Precision: {:.4f}| Test Recall : {:.4f}| Test Weighted f1 score: {:.4f}|".format(
                test_acc.item(), test_loss, test_precision, test_recall, test_fscore)
        )
        log_dict = {
            "Accuracy": test_acc,
            "Precision": test_precision,
            "Recall": test_recall,
            "Fscore" : test_fscore,
            "Loss" : test_loss
        }
        wandb.log(log_dict)

    return test_acc.item()


if __name__ == '__main__':
    wandb.login()
    print(track_acc("CoraImbalanced"))