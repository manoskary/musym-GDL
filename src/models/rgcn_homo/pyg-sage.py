from pytorch_geometric.data import CoraFull

import torch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader




class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        return x_j

    def update(self, aggr_out, x):
        new_embedding = torch.cat([aggr_out, x], dim=1)
        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)
        return new_embedding


class SAGE(torch.nn.Module):
    def __init__(self, in_feats, num_hidden, num_classes, num_layers=1):
        super(Net, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.pool_layers = torch.nn.ModuleList()
        self.layers.append = SAGEConv(in_feats, num_hidden)
        self.pool_layers.append(TopKPooling(num_hidden, ratio=0.8))
        for i in range(num_layers-1):
            self.layers.append = SAGEConv(num_hidden, num_hidden)

        self.linlayer = torch.nn.Linear(num_hidden, num_hidden)
        self.linout = torch.nn.Linear(num_hidden, num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        lin = list()
        for conv, pool in zip(self.layers, self.pool_layers):
            x = F.relu(conv(x, edge_index))
            x, edge_index, _, batch, _ = pool(x, edge_index, None, batch)
            lin.append(torch.cat([gmp(x, batch), gap(x, batch)], dim=1))
        x = sum(lin)
        x = torch.nn.ReLU(self.linlayer(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.sigmoid(self.linout(x)).squeeze(1)
        return x


def train(model, loader, optimizer, criterion, len_train, device):
    model.train()
    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = criterion(output, label)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len_train


def evaluate(model, loader):
    model.eval()

    predictions = []
    labels = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()

            label = data.y.detach().cpu().numpy()
            predictions.append(pred)
            labels.append(label)

    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    return roc_auc_score(labels, predictions)



def main():


    train_dataset = CoraFull()
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device('cuda')
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.BCELoss()

    for epoch in range(1):
        loss = train(model, train_loader, optimizer, criterion, len(train_dataset), device)
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)
        test_acc = evaluate(model, test_loader)
        print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.
              format(epoch, loss, train_acc, val_acc, test_acc))