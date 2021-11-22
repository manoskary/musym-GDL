import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl
import tqdm
import dgl.function as fn
import pyro


# ------------------------ GAE MODEL FOR LINK PREDICTIVE AUGMENTATION ----------------------------

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)
        self.gcn_msg = fn.copy_src(src='h', out='m')
        self.gcn_reduce = fn.sum(msg='m', out='h')

    def forward(self, g, feature):
        with g.local_scope():
            # sum aggregation
            g.ndata['h'] = feature
            g.update_all(self.gcn_msg, self.gcn_reduce)
            g.apply_nodes(func=self.apply_mod)
            h = g.ndata.pop('h')
            return h


class GAE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(GAE, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # Probably should change nework to GraphSAGE
        self.layers.append(dglnn.GraphConv(self.in_feats, self.n_hidden, allow_zero_in_degree=True))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.GraphConv(self.n_hidden, self.n_hidden, allow_zero_in_degree=True))

    def forward(self, blocks, inputs):
        h = self.encode(blocks, inputs)
        adj_rec = self.decode(h)
        return adj_rec

    def encode(self, blocks, inputs):
        h = inputs
        for l, (conv, block) in enumerate(zip(self.layers, blocks)):
            h = conv(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def decode(self, h):
        h = self.dropout(h)
        adj = th.matmul(h, h.t())
        return adj


class NonDglSAGELayer(nn.Module):
    """ one layer of GraphSAGE with gcn aggregator """

    def __init__(self, in_feats, out_feats):
        super(NonDglSAGELayer, self).__init__()
        self.linear_neigh = nn.Linear(in_feats, out_feats, bias=False)
        self.bias = nn.Parameter(th.zeros(out_feats))
        self.init_params()

    def init_params(self):
        """ Initialize weights with xavier uniform and biases with all zeros """
        for param in self.parameters():
            if len(param.size()) == 2:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        # using GCN aggregator
        y = adj @ h
        y = self.linear_neigh(y)
        y = y + self.bias
        return y


class NonDglSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(NonDglSAGE, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(NonDglSAGELayer(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(NonDglSAGELayer(n_hidden, n_hidden))
        self.layers.append(NonDglSAGELayer(n_hidden, n_classes))

    def forward(self, adj, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(adj, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = F.normalize(h)
                h = self.dropout(h)
        return h


class GaugLoss(nn.Module):
    def __init__(self, beta, weight=None):
        super(GaugLoss, self).__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(weight=weight)
        # self.bce = nn.BCELoss()

    def forward(self, output, target, adj_true, adj_pred, weight):
        norm_w = adj_true.shape[0] ** 2 / float((adj_true.shape[0] ** 2 - adj_true.sum()) * 2)
        ce_loss = self.ce(output, target)
        # bce_loss = self.bce(adj_pred, adj_true)
        bce_loss = norm_w * F.binary_cross_entropy_with_logits(th.sigmoid(adj_pred), adj_true, pos_weight=weight)
        return ce_loss + self.beta * bce_loss


class Gaug(nn.Module):
    """
    Gaug end-to-end trainable edge augmentation and Node Classification model.

    Parameters
    ---------
    in_feats: int
    n_hidden : int
    n_classes : int
    n_layers : int
    activation : nn.Module
    dropout : float
    alpha : float
    beta : float
    temperature : float
    """

    def __init__(self, in_feats, n_hidden, n_classes,
                 n_layers, activation=F.relu, dropout=0.5, alpha=1, temperature=0.2):
        super(Gaug, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.gae = GAE(in_feats, n_hidden, n_layers, activation=activation, dropout=dropout)
        self.sage = NonDglSAGE(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            activation=activation,
            dropout=dropout)
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, adj, blocks, inputs, feat_inputs):
        self.ep = self.gae(blocks, inputs)
        P = self.interpolate(adj, self.ep)
        A_new = self.sampling(P)
        # The next two lines are computationally redundant. Maybe should compute sage directly from adjacency.
        A_new = self.normalize_adj(A_new).fill_diagonal_(1)
        # A_new = sp.coo_matrix(self.normalize_adj(A_new).detach().cpu())
        # g_new = numpy_to_graph(A_new.toarray()).to(device='cuda') if self.use_cuda else numpy_to_graph(A_new.toarray())
        h = self.sage(A_new, feat_inputs)
        return h

    def interpolate(self, A, M):
        """Interpolate the original adjacency with the predicted from GAE"""
        M = M / th.max(M)
        P = self.alpha * M + (1 - self.alpha) * A
        return P

    def sampling(self, P):
        """Sample from a Bernoulli"""
        # sampling
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature,
                                                                         logits=P).rsample()
        # making adj_sampled symmetric
        adj_sampled = adj_sampled.triu(1)
        adj_sampled = adj_sampled + adj_sampled.T
        return adj_sampled

    def normalize_adj(self, adj):
        """Normalize an Adjacency matrix"""
        adj.fill_diagonal_(1)
        adj = F.normalize(adj, p=1, dim=1)
        return adj

    def inference(self, g, device, batch_size, num_workers=0):
        """
        Perform Evaluation using Mini-Batching.

        Parameters
        ----------
        g : DGLGraph
        device : str
        batch_size : int
        num_workers : int

        Returns
        -------
        y : Tensor
            The predictions for the nodes of g.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        g.ndata['idx'] = th.tensor(range(g.number_of_nodes()))
        node_features = g.ndata['feat']
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        dataloader = dgl.dataloading.EdgeDataLoader(
            g,
            th.arange(g.number_of_edges()),
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers)
        y = th.zeros(g.num_nodes(), self.n_classes)
        for input_nodes, sub_g, blocks in tqdm.tqdm(dataloader, position=0, leave=True):
            blocks = [block.int().to(device) for block in blocks]
            batch_inputs = node_features[input_nodes].to(device)
            feat_inputs = sub_g.ndata["feat"].to(device)
            adj = sub_g.adj(ctx=device).to_dense()
            h = self.forward(adj, blocks, batch_inputs, feat_inputs)
            # TODO prediction may replace values because Edge dataloder repeats nodes, maybe take average or addition.
            y[sub_g.ndata['idx']] = h.cpu()
        return y
