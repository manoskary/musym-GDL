import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *




class HyperbolicSageConv(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(self, HyperbolicSageConv).__init__()
        self.bias = bias
        self.embed = torch.zeros((in_feats, in_feats), requires_grad=True)
        self.layer = torch.zeros((2*in_feats, out_feats), requires_grad=True)
        self.reset_parameters()
        self.embed = nn.Parameter(self.embed)
        self.layer = nn.Parameter(self.layer)
        if self.bias:
            self.embed_bias = torch.zeros((in_feats, 1), requires_grad=True)
            self.layer_bias = torch.zeros((out_feats, 1), requires_grad=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer, gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.constant(self.embed_bias, 0.)
            nn.init.constant(self.layer_bias, 0.)

    def forward(self, x, adj):
        h = h_mul(self.embed_bias, x)
        if self.bias:
            h = h_add(h, self.embed_bias)
        neigh = exp_map_zero(torch.mm(adj, log_map_zero(h)).sum(dim=1))
        h = h_mul(self.layer, torch.cat((x, neigh)))
        if self.bias:
            h = h_add(h, self.layer_bias)
        return h


class HGNN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation=F.relu, dropout=0.5):
        super(HGNN, self).__init__()
        self.init_layer = exp_map_zero
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(HyperbolicSageConv(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(HyperbolicSageConv(n_hidden, n_hidden))
        self.layers.append(HyperbolicSageConv(n_hidden, out_feats))

    def forward(self, x, adj):
        # 0-projection to Poincare Hyperbolic space.
        h = self.init_layer(x)
        for i, layer in enumerate(self.layers):
            h = layer(x, adj)
            if i != len(layer)-1:
                h = exp_map_zero(self.activation(log_map_zero(h)))
                h = self.dropout(h)
        return h
