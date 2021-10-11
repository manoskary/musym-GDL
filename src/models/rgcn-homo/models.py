import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl
import tqdm
import dgl.function as fn
import pyro

class mySAGEConv(nn.Module):

    def __init__(self, in_feats, out_feats, aggretype="maxpool"):
        super(mySAGEConv, self).__init__()
        self.in_feats = in_feats 
        self.out_feats = out_feats
        self.aggretype = aggretype
        # Trainable weights for Linear Transformation
        self.linear = nn.Linear(in_feats, out_feats)
        self.bias = nn.Parameter(th.zeros(out_feats))
        gain = nn.init.calculate_gain('relu')
        
        if self.aggretype == "maxpool":
            # Call max pooling init
            self.pool = nn.Linear(self.in_feats, self.in_feats, bias=False)
            self.pool_bias = nn.Parameter(th.zeros(in_feats))
            # Init glorot transform to aggre layer and linear layer
            nn.init.xavier_uniform_(self.pool.weight, gain=gain)        
        elif self.aggretype == "lstm":
            self.lstm = nn.LSTM(self.in_feats, self.in_feats, batch_first=True)
            self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)


    def _lstm_reducer(self, nodes):
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self.in_feats)),
             m.new_zeros((1, batch_size, self.in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'h_new': rst.squeeze(0)}

        
    def aggregate(self, g, node_feats, e_weight):
        # Search for edge weights
        msg_fn = fn.u_mul_e('h', 'w', 'm') if e_weight != None else fn.copy_src('h', 'm')
        # Update message functions
        with g.local_scope():    
            if self.aggretype == "maxpool":
                g.srcdata['h'] = F.relu(self.pool(node_feats) + self.pool_bias)
                g.update_all(msg_fn, fn.max('m', 'h_new'))
            elif self.aggretype == "lstm":
                g.srcdata['h'] = node_feats
                g.update_all(msg_fn, self._lstm_reducer)
            aggregated = g.dstdata['h_new']
            return aggregated

    def forward(self, g, node_feats, e_weight=None):
        """
        Forward Layer.

        In the literature after forward opetation a activation and a normalization should follow.

        Parameters
        ----------
        g : graph
            For the moment only considers homo graphs.
        node_feats : tensor
            The node features if hetero then dict of tensors
        e_weights : tensor
            The weight features if hetero then dict of tensors

        Returns
        -------
        h : tensor
            A learned representation for every node of the graph.
        """
        # Aggregate
        h = self.aggregate(g, node_feats, e_weight)
        # Apply Learnable Weight
        h = self.linear(h)
        # bias term
        h = h + self.bias
        return h


class mylstmSAGEConv(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(mylstmSAGEConv, self).__init__()
        self.in_feats = in_feats 
        self.out_feats = out_feats
        # Trainable weights for Linear Transformation
        self.layer = nn.LSTM(input_size=in_feats, hidden_size=out_feats)
        self.linear = nn.Linear(out_feats, out_feats)

        self.bias = nn.Parameter(th.zeros(out_feats))
        self.lstm = nn.LSTM(self.in_feats, self.in_feats, batch_first=True)
        self.layer.reset_parameters()
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))


    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self.in_feats)),
             m.new_zeros((1, batch_size, self.in_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'h_new': rst.squeeze(0)}


    def aggregate(self, g, node_feats, e_weight):
        # Search for edge weights
        msg_fn = fn.u_mul_e('h', 'w', 'm') if e_weight != None else fn.copy_src('h', 'm')
        # Update message functions
        with g.local_scope():    
            g.srcdata['h'] = node_feats
            g.update_all(msg_fn, self._lstm_reducer)
            aggregated = g.dstdata['h_new']
            return aggregated

    def forward(self, g, node_feats, e_weight=None):
        # Aggregate
        h = self.aggregate(g, node_feats, e_weight)
        # Apply Learnable Weight
        h = th.squeeze(self.layer(h)[0])
        h = self.linear(h)
        # bias term
        h = h + self.bias
        return h


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(mySAGEConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(mySAGEConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(mySAGEConv(n_hidden, n_classes)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = F.normalize(h)
                h = self.dropout(h)
        return h

class LSTMGraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(LSTMGraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(mylstmSAGEConv(in_feats, n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(mylstmSAGEConv(n_hidden, n_hidden))
        # output layer
        self.layers.append(mylstmSAGEConv(n_hidden, n_classes)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        # h = F.normalize(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = F.normalize(h)
                h = self.dropout(h)
        return h

# Define a Homogeneous Conv model
class SGC(nn.Module):
    """
    The SAGE object produces a Relational Graph Convolutional Network for homogeneous graphs.

    Parameters
    ----------
    in_feats : int
        The size of the Node Feature Vector
    hid_feats : int
        The size of the Latent Node Feature Representation of the hidden layers
    out_feats : int
        The number of the node classes we want to predict.
    rel_names : list
        The graph edge types
    num_hidden_layers : int
        The number of Hidden layers. 

    Attributes
    ----------
    layers : nn.ModuleList()
        The number of layers and information about inputs and outputs
    num_hidden_layers : int
        The number of hidden layers

    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
            
        # i2h
        self.layers.append(dglnn.SGConv(
            in_feats=in_feats, out_feats=n_hidden))
        # h2h
        for i in range(n_layers - 1):
            self.layers.append(dglnn.SGConv(
            in_feats=n_hidden, out_feats=n_hidden))
        # h2o
        self.layers.append(dglnn.SGConv(
            in_feats=n_hidden, out_feats=n_classes))


    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        # h = F.normalize(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class GAT(nn.Module):
    """
    The SAGE object produces a Relational Graph Convolutional Network for homogeneous graphs.

    Parameters
    ----------
    in_feats : int
        The size of the Node Feature Vector
    hid_feats : int
        The size of the Latent Node Feature Representation of the hidden layers
    out_feats : int
        The number of the node classes we want to predict.
    rel_names : list
        The graph edge types
    num_hidden_layers : int
        The number of Hidden layers. 

    Attributes
    ----------
    layers : nn.ModuleList()
        The number of layers and information about inputs and outputs
    num_hidden_layers : int
        The number of hidden layers

    """
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
            
        # i2h
        self.layers.append(dglnn.GATConv(
            in_feats=in_feats, out_feats=n_hidden, num_heads=3))
        # h2h
        for i in range(n_layers - 1):
            self.layers.append(dglnn.GATConv(
            in_feats=n_hidden, out_feats=n_hidden, num_heads=3))
        # h2o
        self.layers.append(dglnn.GATConv(
            in_feats=n_hidden, out_feats=n_classes, num_heads=3))


    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        # h = F.normalize(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h



class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'pool'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'pool'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'pool'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'pool'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = F.normalize(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y



def expand_as_pair(input_, g=None):
    """Return a pair of same element if the input is not a pair.
    If the graph is a block, obtain the feature of destination nodes from the source nodes.
    Parameters
    ----------
    input_ : Tensor, dict[str, Tensor], or their pairs
        The input features
    g : DGLHeteroGraph or DGLGraph or None
        The graph.
        If None, skip checking if the graph is a block.
    Returns
    -------
    tuple[Tensor, Tensor] or tuple[dict[str, Tensor], dict[str, Tensor]]
        The features for input and output nodes
    """
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {
                k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                for k, v in input_.items()}
        else:
            input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_



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
        x = adj @ h
        x = self.linear_neigh(x)
        x = x + self.bias
        return x

class NonDglSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(NonDglSAGE, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(NonDglSAGELayer(in_feats, n_hidden))
        for i in range(n_layers-1):
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
    def __init__(self, beta):
        super(GaugLoss, self).__init__()
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
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
        adj_sampled = pyro.distributions.RelaxedBernoulliStraightThrough(temperature=self.temperature, logits=P).rsample()
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

