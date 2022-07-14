import torch
import torch.nn as nn
from .model import HGNN
from tqdm import tqdm


def main():
    graphs = list()
    in_feats = graphs[0].x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HGNN(in_feats, 16, in_feats, 3)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in tqdm(range(50), position=0, desc="Epoch", leave=True):
        for graph in tqdm(graphs[:200], position=1, desc="Graph", leave=True):
            x, edge_index = graph.x, graph.edge_index
            x = x.squeeze().to(device)
            edge_index = edge_index.to(device)
            optimizer.zero_grad()
            out, mu, log_var = model(x, edge_index)
            loss = criterion(out, x, mu, log_var)
            loss.backward()
            optimizer.step()
        print("Epoch {:3d} | Loss {:.4f}".format(epoch, loss.item()))