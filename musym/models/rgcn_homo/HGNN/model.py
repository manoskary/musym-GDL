import torch.nn as nn
from .utils import *


class CentroidDistance(nn.Module):
    """
    Implement a model that calculates the pairwise distances between node representations
    and centroids
    """
    def __init__(self, args, logger, manifold):
        super(CentroidDistance, self).__init__()
        self.args = args
        self.logger = logger
        self.manifold = manifold

        # centroid embedding
        self.centroid_embedding = nn.Embedding(
            args.num_centroid, args.embed_size,
            sparse=False,
            scale_grad_by_freq=False)
        self.init_embed()

    def init_embed(self, irange=1e-2):
        self.centroid_embeddin.weight.data.uniform_(-irange, irange)
        self.centroid_embeddin.weight.data.copy_(self.normalize(self.centroid_embeddin.weight.data))

    def forward(self, node_repr, mask):
        """
        Args:
            node_repr: [node_num, embed_size]
            mask: [node_num, 1] 1 denote real node, 0 padded node
        return:
            graph_centroid_dist: [1, num_centroid]
            node_centroid_dist: [1, node_num, num_centroid]
        """
        node_num = node_repr.size(0)

        # broadcast and reshape node_repr to [node_num * num_centroid, embed_size]
        node_repr = node_repr.unsqueeze(1).expand(
                                                -1,
                                                self.args.num_centroid,
                                                -1).contiguous().view(-1, self.args.embed_size)

        # broadcast and reshape centroid embeddings to [node_num * num_centroid, embed_size]
        if self.args.embed_manifold == 'hyperbolic':
            centroid_repr = self.centroid_embedding(torch.arange(self.args.num_centroid).cuda())
        else:
            centroid_repr = self.manifold.exp_map_zero(
                self.centroid_embedding(torch.arange(self.args.num_centroid).cuda()))
        centroid_repr = centroid_repr.unsqueeze(0).expand(
                                                node_num,
                                                -1,
                                                -1).contiguous().view(-1, self.args.embed_size)
        # get distance
        node_centroid_dist = self.manifold.distance(node_repr, centroid_repr)
        node_centroid_dist = node_centroid_dist.view(1, node_num, self.args.num_centroid) * mask
        # average pooling over nodes
        graph_centroid_dist = torch.sum(node_centroid_dist, dim=1) / torch.sum(mask)
        return graph_centroid_dist, node_centroid_dist


class HyperbolicSageConv(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True):
        super(HyperbolicSageConv, self).__init__()
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
