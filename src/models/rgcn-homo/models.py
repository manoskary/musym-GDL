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

    def __init__(self, in_feats, out_feats):
        super(mySAGEConv, self).__init__()
        self.in_feats = in_feats 
        self.out_feats = out_feats
        # Call max pooling init
        self.pool = nn.Linear(self.in_feats, self.in_feats, bias=False)
        self.pool_bias = nn.Parameter(th.zeros(in_feats))
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.pool.weight, gain=gain)
        self.linear = nn.Linear(in_feats, out_feats)
        self.bias = nn.Parameter(th.zeros(out_feats))

    def forward(self, g, node_feats, e_weight=None):
        # Check if graph has weights        
        with g.local_scope():    
            g.srcdata['h'] = F.relu(self.pool(node_feats) + self.pool_bias)
            if e_weight != None:
                g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.max('m', 'h_new'))
            else:
                g.update_all(fn.copy_src('h', 'm'), fn.max('m', 'h_new'))
            out = self.linear(g.dstdata['h_new'])
        # bias term
        out = out + self.bias
        return out

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
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
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




                    