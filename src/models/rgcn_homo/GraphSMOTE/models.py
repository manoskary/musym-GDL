import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import dgl.nn as dglnn

# Graphsage layer
class SageConvLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=False):
		super(SageConvLayer, self).__init__()
		self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
		if self.linear.bias is not None:
			nn.init.constant_(self.linear.bias, 0.)

	def forward(self, adj, features):
		"""

		Parameters
		----------
		adj : torch sparse or dense
			Can be Dense or Sparse Adjacency Matrix.
		features : torch.tensor
			A float tensor with the node attributes.
		Returns
		-------
		combined : torch.tensor
			An embeded feature tensor.
		"""
		if not isinstance(adj, torch.sparse.FloatTensor):
			if len(adj.shape) == 3:
				neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
			else:
				neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
		# For Sparse Adjacency Matrices
		else:
			neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

		# perform conv
		data = torch.cat([features, neigh_feature], dim=-1)
		combined = self.linear(data)
		return combined


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


class SageClassifier(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers=1, activation=F.relu, dropout=0.1):
		super(SageClassifier, self).__init__()
		self.in_feats = in_feats
		self.n_hidden = n_hidden
		self.n_layers = n_layers
		self.activation = activation
		self.dropout = nn.Dropout(dropout)
		self.clf = nn.Linear(n_hidden, n_classes)
		self.layers = nn.ModuleList()
		self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
		for i in range(n_layers - 1):
			self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.clf.weight, gain=nn.init.calculate_gain('relu'))
		if self.clf.bias is not None:
			nn.init.constant_(self.clf.bias, 0.)

	def forward(self, adj, inputs):
		h = inputs
		for l, conv in enumerate(self.layers):
			h = conv(adj, h)
			if l != self.n_layers-1:
				h = self.activation(h)
				h = self.dropout(h)
		h = self.clf(h)
		return h


class SageDecoder(nn.Module):
	def __init__(self, num_hidden, dropout=0.1):
		super(SageDecoder, self).__init__()
		self.dropout = dropout
		self.de_weight = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.de_weight.size(1))
		self.de_weight.data.uniform_(-stdv, stdv)

	def forward(self, encoded_features):
		out = F.linear(encoded_features, self.de_weight)
		adj_out = torch.sigmoid(torch.mm(out, out.transpose(-1, -2)))
		return adj_out


class EdgeLoss(nn.Module):
	def __init__(self):
		super(EdgeLoss, self).__init__()

	def forward(self, adj_rec, adj_tgt, adj_mask = None):
		adj_rec = adj_rec[:len(adj_tgt), :len(adj_tgt)]
		edge_num = adj_tgt.nonzero().shape[0]
		total_num = adj_tgt.shape[0] ** 2
		neg_weight = edge_num / (total_num - edge_num)
		weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
		weight_matrix[adj_tgt == 0] = neg_weight
		loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)
		return loss



class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # Probably should change nework to GraphSAGE
        self.layers.append(dglnn.SAGEConv(self.in_feats, self.n_hidden, aggregator_type="pool"))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.SAGEConv(self.n_hidden, self.n_hidden, aggregator_type="pool"))

    def forward(self, blocks, inputs):
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


