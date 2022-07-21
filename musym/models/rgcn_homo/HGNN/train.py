import torch
import torch.nn as nn
from model import HGNN
from tqdm import tqdm
from dgl.data import CoraGraphDataset

def main():
    dataset = CoraGraphDataset()
    g = dataset[0]
    in_feats = g.ndata["feat"].shape[1]
    n_classes = dataset.num_classes
    model = HGNN(in_feats, 16, n_classes, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in tqdm(range(50), position=0, desc="Epoch", leave=True):
        x = g.ndata["feat"]
        y = g.ndata["label"]
        adj = g.adj().to_dense()
        optimizer.zero_grad()
        pred = model(x, adj)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        print("Epoch {:3d} | Loss {:.4f}".format(epoch, loss.item()))


if __name__ == '__main__':
    main()