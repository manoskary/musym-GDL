import torch
torch.manual_seed(42)
import random
random.seed(42)
import numpy as np
np.random.seed(42)

from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import dgl
from models import NonDglSAGE
from imblearn.over_sampling import ADASYN, SMOTE
import argparse


# Graphsage layer
class SageConv(nn.Module):
	def __init__(self, in_features, out_features, bias=False):
		super(SageConv, self).__init__()
		self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.linear.weight)
		if self.linear.bias is not None:
			nn.init.constant_(self.linear.bias, 0.)

	def forward(self, adj, features):
		"""
        Args:
            adj: can be sparse or dense matrix.
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
		self.layers.append(SageConv(self.in_feats, self.n_hidden))
		for i in range(n_layers - 1):
			self.layers.append(SageConv(self.n_hidden, self.n_hidden))

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
		self.layers = nn.ModuleList()
		self.layers.append(SageConv(self.in_feats, self.n_hidden))
		for i in range(n_layers - 1):
			self.layers.append(SageConv(self.n_hidden, self.n_hidden))

	def forward(self, adj, inputs):
		h = inputs
		for l, conv in enumerate(self.layers):
			h = conv(adj, h)
			if l != self.n_layers-1:
				h = self.activation(h)
				h = self.dropout(h)
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

	def forward(self, node_embed):
		combine = F.linear(node_embed, self.de_weight)
		adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))
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


def main(config):
	"""Pass parameters to create experiment"""

	# --------------- Dataset Loading -------------------------
	dataset = dgl.data.CoraGraphDataset()
	g = dataset[0]
	n_classes = dataset.num_classes
	adj = g.adj().to_dense()
	train_nid = torch.tensor(range(g.num_nodes())).type(torch.int64)
	in_feats = g.ndata["feat"].shape[1]

	# --------------- Transfer to Devise ---------------
	use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
	if use_cuda:
		device = torch.device('cuda:%d' % config["gpu"])
		dataloader_device = device
	else:
		device = torch.device('cpu')
		train_nid = train_nid.to(device)
		dataloader_device = device

	# Define model and optimizer
	encoder = SageEncoder(in_feats, 64, config["num_layers"])
	decoder = SageDecoder(64)
	classifier = SageClassifier(64, 64, n_classes, config["num_layers"])
	dloss = EdgeLoss()
	criterion = nn.CrossEntropyLoss()
	optimizer_en = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=5e-4)
	optimizer_cls = optim.Adam(classifier.parameters(), lr=0.001, weight_decay=5e-4)
	optimizer_de = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=5e-4)

	# Training loop
	node_features = g.ndata["feat"]
	labels = g.ndata["label"]

	sm = SMOTE()

	for epoch in tqdm(range(config["num_epochs"]), desc='epoch'):
		# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
		# Predict and loss
		embed = encoder(adj, node_features)
		upsampl_embed, upsampl_lab = map(lambda x : torch.tensor(x), SMOTE().fit_resample(embed.detach(), labels))
		pred_adj = decoder(upsampl_embed).type(torch.double)
		embed_loss = dloss(pred_adj, adj)
		pred_adj = torch.where(pred_adj>=0.5, pred_adj, 0.).type(torch.float32)
		pred = classifier(pred_adj, upsampl_embed)
		loss = criterion(pred, upsampl_lab) + embed_loss*0.000001
		acc = (torch.argmax(pred, dim=1) == upsampl_lab).float().sum() / len(pred)
		optimizer_en.zero_grad()
		optimizer_cls.zero_grad()
		optimizer_de.zero_grad()
		loss.backward()
		optimizer_cls.step()
		optimizer_en.step()
		optimizer_de.step()
		if epoch%100 == 0 and epoch != 0:
			print('Epoch {:04d} | Loss {:.4f} | Accuracy {:.4f} |'.format(epoch, loss.item(), acc))

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description='Weighted Sampling SAGE')
	argparser.add_argument('--gpu', type=int, default=-1,
						   help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('-d', '--dataset', type=str, default='toy01')
	argparser.add_argument('--num-epochs', type=int, default=1000)
	argparser.add_argument('--batch-size', type=int, default=1024)
	argparser.add_argument("--num-layers", type=int, default=1)
	argparser.add_argument("--lambda", type=float, default=1e-4)

	args = argparser.parse_args()
	config = vars(args)

	main(config)
