import torch.nn as nn
import torch
import torch.nn.functional as F


class SageConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(SageConvLayer, self).__init__()
        self.neigh_linear = nn.Linear(in_features, in_features, bias=bias)
        self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=nn.init.calculate_gain('relu'))
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.neigh_linear.bias is not None:
            nn.init.constant_(self.neigh_linear.bias, 0.)

    def forward(self, features, adj, neigh_feats=None):
        if neigh_feats == None:
            neigh_feats = features
        h = self.neigh_linear(neigh_feats)
        if not isinstance(adj, torch.sparse.FloatTensor):
            if len(adj.shape) == 3:
                h = torch.bmm(adj, h) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                h = torch.mm(adj, h) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            h = torch.mm(adj, h) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)
        z = self.linear(torch.cat([features, h], dim=-1))
        return z


class VoicePredSage(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = nn.BatchNorm1d()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(SageConvLayer(in_feats, n_hidden))
        for i in range(n_layers):
            self.layers.append(SageConvLayer(n_hidden, n_hidden))
        self.predictor = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1))

    def predict(self, h_src, h_dst):
        return self.predictor(h_src * h_dst)

    def forward(self, pos_edges, neg_edges, adj, x):
        h = x
        for l, layer in enumerate(zip(self.layers)):
            h = layer(h, adj)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.normalize(h)
                h = self.dropout(h)

        pos_src, pos_dst = pos_edges
        neg_src, neg_dst = neg_edges
        h_pos = self.predict(h[pos_src], h[pos_dst])
        h_neg = self.predict(h[neg_src], h[neg_dst])
        return h_pos, h_neg

