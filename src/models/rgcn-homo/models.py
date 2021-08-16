import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


# Define a Homogeneous Conv model
class SAGE(nn.Module):
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
    def __init__(self, in_feats, hid_feats, out_feats, num_hidden_layers=2, aggr_type="mean"):
        super().__init__()
        self.activation = True
        self.layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
            
        # i2h
        self.layers.append(dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type=aggr_type))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=hid_feats, aggregator_type='gcn'))
        # h2o
        self.layers.append(dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type=aggr_type))


    def forward(self, graph, inputs):
        """
        Forward Funtion

        Parameters
        ----------
        graph : dgl object
            A heterogenous graph
        inputs : dict
            A dictionary with the predict category node type features.
        """
        # inputs are features of nodes
        h = F.normalize(inputs)
        # h = inputs
        for i, conv_l in enumerate(self.layers):
            h = conv_l(graph, h)
            if i == len(self.layers)-1:
                self.activation = False
            if self.activation:
                h = F.relu(h)
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
    def __init__(self, in_feats, hid_feats, out_feats, num_hidden_layers=2,):
        super().__init__()
        self.activation = True
        self.layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
            
        # i2h
        self.layers.append(dglnn.SGConv(
            in_feats=in_feats, out_feats=hid_feats))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(dglnn.SGConv(
            in_feats=hid_feats, out_feats=hid_feats))
        # h2o
        self.layers.append(dglnn.SGConv(
            in_feats=hid_feats, out_feats=out_feats))


    def forward(self, graph, inputs):
        """
        Forward Funtion

        Parameters
        ----------
        graph : dgl object
            A heterogenous graph
        inputs : dict
            A dictionary with the predict category node type features.
        """
        # inputs are features of nodes
        graph = dgl.add_self_loop(graph)
        h = F.normalize(inputs)
        # h = inputs
        for i, conv_l in enumerate(self.layers):
            h = conv_l(graph, h)
            if i == len(self.layers)-1:
                self.activation = False
            if self.activation:
                h = F.relu(h)
        return h