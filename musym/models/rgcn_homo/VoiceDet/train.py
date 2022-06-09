import argparse
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import time
import numpy as np
import tqdm
from musym.models.rgcn_homo.VoiceDet.model import VoicePredSage
from musym.models.rgcn_homo.VoiceDet.utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--pure-gpu', action='store_true',
                    help='Perform both sampling and training on GPU.')
args = parser.parse_args()

device = 'cuda'


def compute_mrr(model, node_emb, src, dst, neg_dst, device, batch_size=500):
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predict(h_src, h_dst).squeeze(-1)
        relevance = torch.zeros(*pred.shape, dtype=torch.bool)
        relevance[:, 0] = True
        rr[start:end] = MF.retrieval_reciprocal_rank(pred, relevance)
    return rr.mean()


def main():
    load_dir = "/home/manos/Desktop/JKU/data/asap_graphs/"
    graphs = [load_score_graph(load_dir, fn) for fn in os.listdir(load_dir)]
    model = VoicePredSage(in_feats=12, n_hidden=128, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        model.train()
        for graph in graphs:
            print(graph.name)
            adj = graph.adj()
            row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
            col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
            edge_index = torch.stack([row, col], dim=0)
            val = torch.from_numpy(adj.data.astype(int)).to(int)
            adj = torch.sparse.FloatTensor(edge_index, val, torch.Size(adj.shape)).to_dense().float()

            edges = torch.Tensor(graph.edge_index).long()
            x = torch.Tensor(graph.x)
            y = torch.Tensor(graph.y).long()
            labels = torch.all(y[edges[0]] == y[edges[1]], dim=1)

            pos_edges = edges.t()[labels].t()
            neg_edges = edges.t()[torch.logical_not(labels)].t()
            pos_score, neg_score = model(pos_edges, neg_edges, adj, x)
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)

if __name__ == '__main__':
    main()