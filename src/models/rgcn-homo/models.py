import dgl.nn as dglnn
from dgl import add_self_loop
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import dgl
import tqdm
from dgl.nn.pytorch.conv import SAGEConv
from dgl.utils import check_eq_shape
import dgl.function as fn

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
        h = torch.squeeze(self.layer(h)[0])
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
        # h = F.normalize(h)
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




                    