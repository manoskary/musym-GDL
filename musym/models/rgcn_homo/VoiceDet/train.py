import argparse
import torch
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import time
import numpy as np
import tqdm
from .model import VoicePredSage

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
    graphs = score_graphs()
    model = VoicePredSage(in_feats=135, n_hidden=256, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):
        model.train()
        t0 = time.time()
        for graph in graphs:
            adj = graph.adj()
            edges = graph.edge_index
            x = graph.x
            y = graph.y
            labels = torch.nonzero(y[edges[0]] == y[edges[1]]).squeeze()
            non_labels = torch.nonzero(y[edges[0]] != y[edges[1]]).squeeze()
            pos_edges = edges[:, labels.int()]
            neg_edges = edges[:, non_labels.int()]
            pos_score, neg_score = model(pos_edges, neg_edges, adj, x)
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            score = torch.cat([pos_score, neg_score])
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
