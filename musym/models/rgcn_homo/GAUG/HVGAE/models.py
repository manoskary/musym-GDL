import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import coalesce
from torch_geometric.utils import softmax


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

    def forward(self, adj, features, neigh_feats=None):
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


class SageEncoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.1):
        super(SageEncoder, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

    def forward(self, adj, inputs):
        h = inputs
        for l, conv in enumerate(self.layers):
            h = conv(adj, h)
            h = self.activation(F.normalize(h))
            h = self.dropout(h)
        return h


class EdgePooling(torch.nn.Module):
    r"""The edge pooling operator from the `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ papers.
    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.
    To duplicate the configuration from the "Towards Graph Pooling by Edge
    Contraction" paper, use either
    :func:`EdgePooling.compute_edge_score_softmax`
    or :func:`EdgePooling.compute_edge_score_tanh`, and set
    :obj:`add_to_edge_score` to :obj:`0`.
    To duplicate the configuration from the "Edge Contraction Pooling for
    Graph Neural Networks" paper, set :obj:`dropout` to :obj:`0.2`.
    Args:
        in_channels (int): Size of each input sample.
        edge_score_method (function, optional): The function to apply
            to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`EdgePooling.compute_edge_score_softmax`,
            :func:`EdgePooling.compute_edge_score_tanh`, and
            :func:`EdgePooling.compute_edge_score_sigmoid`.
            (default: :func:`EdgePooling.compute_edge_score_softmax`)
        dropout (float, optional): The probability with
            which to drop edge scores during training. (default: :obj:`0`)
        add_to_edge_score (float, optional): This is added to each
            computed edge score. Adding this greatly helps with unpool
            stability. (default: :obj:`0.5`)
    """

    unpool_description = namedtuple(
        "UnpoolDescription",
        ["edge_index", "cluster", "batch", "new_edge_score"])

    def __init__(self, in_channels, edge_score_method=None, dropout=0,
                 add_to_edge_score=0.5):
        super(EdgePooling, self).__init__()
        self.in_channels = in_channels
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(raw_edge_score, edge_index, num_nodes):
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(raw_edge_score, edge_index, num_nodes):
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.
        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
        Return types:
            * **x** *(Tensor)* - The pooled node features.
            * **edge_index** *(LongTensor)* - The coarsened edge indices.
            * **batch** *(LongTensor)* - The coarsened batch vector.
            * **unpool_info** *(unpool_description)* - Information that is
              consumed by :func:`EdgePooling.unpool` for unpooling.
        """
        # Run nodes on each edge through a linear layer
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        # Edge score is the cosine similarity between respective nodes
        # e = F.cosine_similarity(x[edge_index[0]], x[edge_index[1]])

        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        x, edge_index, batch, unpool_info = self.__merge_edges__(
            x, edge_index, batch, e)

        return x, edge_index, batch, unpool_info

    def __merge_edges__(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(x.size(0)))

        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        # edge_argsort = torch.argsort(edge_score, descending=True)
        edge_argsort = edge_score.detach().cpu().numpy().argsort(kind='stable')[::-1]  # Use stable sort

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            # print(edge_score[edge_idx], edge_index[:,edge_idx])
            source = edge_index_cpu[0, edge_idx].item()
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            if target not in nodes_remaining:
                continue

            new_edge_indices.append(edge_idx)

            cluster[source] = i
            nodes_remaining.remove(source)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices), ))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)

        N = new_x.size(0)
        new_edge_index, _ = coalesce(cluster[edge_index], None, N, N)

        new_batch = x.new_empty(new_x.size(0), dtype=torch.long)
        new_batch = new_batch.scatter_(0, cluster, batch)

        unpool_info = self.unpool_description(edge_index=edge_index,
                                              cluster=cluster, batch=batch,
                                              new_edge_score=new_edge_score)

        return new_x, new_edge_index, new_batch, unpool_info

    def unpool(self, x, unpool_info):
        r"""Unpools a previous edge pooling step.
        For unpooling, :obj:`x` should be of same shape as those produced by
        this layer's :func:`forward` function. Then, it will produce an
        unpooled :obj:`x` in addition to :obj:`edge_index` and :obj:`batch`.
        Args:
            x (Tensor): The node features.
            unpool_info (unpool_description): Information that has
                been produced by :func:`EdgePooling.forward`.
        Return types:
            * **x** *(Tensor)* - The unpooled node features.
            * **edge_index** *(LongTensor)* - The new edge indices.
            * **batch** *(LongTensor)* - The new batch vector.
        """

        new_x = x / unpool_info.new_edge_score.view(-1, 1)
        new_x = new_x[unpool_info.cluster]
        return new_x, unpool_info.edge_index, unpool_info.batch

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.in_channels)


class HierarchicalGraphNet(torch.nn.Module):
    """The Hierarchical GraphNet
    TODO: update docstring
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 dropout_ratio, normalize, activation=F.relu):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.activation = activation
        self.dropout_ratio = dropout_ratio
        self.normalize = normalize
        assert inter_connect in ('sum', 'concat', 'edge', 'addnode'),\
            f"Unknown inter-layer connection type: {inter_connect}"
        self.inter_connect = inter_connect

        channels = hidden_channels
        norm_class = torch.nn.BatchNorm1d

        # Pooling and Convolutions going UP the hierarchy towards coarsest level
        self.node_preembedder = torch.nn.Linear(in_channels, channels)
        self.up_convs = torch.nn.ModuleList()
        self.up_norms = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        self.up_convs.append(SageConvLayer(in_channels, channels))
        self.up_norms.append(norm_class(channels))
        for _ in range(depth):
            self.pools.append(EdgePooling(channels, dropout=0))
            self.up_convs.append(SageConvLayer(channels, channels))
            self.up_norms.append(norm_class(channels))
        if self.no_up_convs:  # wipe
            self.up_convs = torch.nn.ModuleList()

        # Convolutions going back DOWN the hierarchy from coarsest to finest level
        in_channels = 2 * channels if inter_connect == 'concat' else channels
        self.down_convs = torch.nn.ModuleList()
        self.down_norms = torch.nn.ModuleList()
        self.down_convs.append(SageConvLayer(in_channels, out_channels))
        self.down_norms.append(norm_class(out_channels))
        for _ in range(depth - 1):
            self.down_convs.append(SageConvLayer(in_channels, channels))
            self.down_norms.append(norm_class(channels))
        if not self.normalize:  # wipe
            self.up_norms = torch.nn.ModuleList()
            self.down_norms = torch.nn.ModuleList()


    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        # TODO: support edge weights (need to augment the pooling)
        # edge_weight = x.new_ones(edge_index.size(1))

        # x = self.up_convs[0](x, edge_index, edge_weight)

        xs = [x]
        edge_indices = [edge_index]
        # edge_weights = [edge_weight]
        unpool_infos = []

        for level in range(1, self.depth + 1):
            x, edge_index, batch, unpool_info = self.pools[level - 1](
                x, edge_index, batch)

            if not self.no_up_convs or level == self.depth:
                x = self.up_convs[level](x, edge_index)
                x = self.activation(x)
                x = F.dropout(x, self.dropout_ratio, training=self.training)

            if level < self.depth:
                xs += [x]
                edge_indices += [edge_index]
            unpool_infos += [unpool_info]

        for level in reversed(range(self.depth)):
            res = xs[level]

            unpool_info = unpool_infos[level]

            unpooled, edge_index, batch = self.pools[level].unpool(x, unpool_info)

            x = torch.cat((res, unpooled), dim=-1)

            if level > 0:
                x = self.act(x)
                x = F.dropout(x, self.dropout_ratio, training=self.training)

        return x

    def __repr__(self):
        rep = '{}({}, {}, {}, depth={}, inter_connect={})'.format(
            self.__class__.__name__, self.in_channels, self.hidden_channels,
            self.out_channels, self.depth, self.inter_connect)
        rep += '\n'
        rep += super().__repr__()
        return rep
