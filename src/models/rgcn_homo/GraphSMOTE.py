import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
from models import NonDglSAGE
from imblearn.over_sampling import ADASYN
import argparse


class VGAE(nn.Module):
	def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, delta=0.5):
		super(VGAE, self).__init__()
		self.in_feats = in_feats
		self.n_hidden = n_hidden
		self.activation = activation
		self.delta = delta
		self.dropout = nn.Dropout(dropout)
		self.weight = nn.Linear(n_hidden, n_hidden)
		self.layers = nn.ModuleList()
		self.oversample = ADASYN()
		# Probably should change nework to GraphSAGE
		self.layers.append(SAGEConv(self.in_feats, self.n_hidden, aggregator_type="mean"))
		for i in range(n_layers - 1):
			self.layers.append(SAGEConv(self.n_hidden, self.n_hidden, aggregator_type="mean"))

	def forward(self, blocks, inputs, labels):
		h = self.encode(blocks, inputs)
		h, labels = self.create_nodes(h, labels)
		adj_rec, h = self.decode(h)
		return adj_rec, h, labels

	def encode(self, g, inputs):
		h = inputs
		for l, conv in enumerate(self.layers):
			h = conv(g, h)
			if l != len(self.layers) - 1:
				h = self.activation(h)
				h = self.dropout(h)
		return h

	def decode(self, h):
		h = self.dropout(h)
		adj = torch.matmul(self.weight(h), h.t())
		adj = F.softmax(F.relu(adj), dim=1)
		return adj, h

	def create_nodes(self, h, labels):
		h, labels = self.oversample.fit_resample(h.detach(), labels)
		return torch.tensor(h), torch.tensor(labels)

class EdgeLoss(nn.Module):
	def __init__(self):
		super(EdgeLoss, self).__init__()

	def forward(self, pred_edges, org_adj):
		h = pred_edges[:len(org_adj), :len(org_adj)]
		return F.binary_cross_entropy_with_logits(h, org_adj)


class GraphSMOTE(nn.Module):
	def __init__(self, in_feats, num_hidden, num_classes, num_layers, activation=F.relu, dropout=0.5, eta=0.5, delta=0.5):
		super(GraphSMOTE, self).__init__()
		self.generator = VGAE(in_feats=in_feats, n_hidden=num_hidden, n_layers=num_layers, activation=activation, dropout=dropout, delta=delta)
		self.gnn = NonDglSAGE(in_feats=num_hidden, n_hidden=num_hidden, n_classes=num_classes, n_layers=num_layers, activation=activation, dropout=dropout)
		self.in_feats = in_feats
		self.loss = EdgeLoss()
		self.eta = eta

	def forward(self, blocks, inputs, labels):
		adj, h, labels = self.generator(blocks, inputs, labels)
		adj = F.threshold(adj, self.eta, 0)
		self.pred = self.gnn(adj, h)
		return self.pred, labels

	def edge_loss(self, blocks):
		org_adj = blocks.adj().to_dense()
		return self.loss(self.pred, org_adj)



def main(config):
	"""Pass parameters to create experiment"""

	# --------------- Dataset Loading -------------------------
	dataset = dgl.data.CoraGraphDataset()
	g = dataset[0]
	n_classes = dataset.num_classes
	g = dgl.add_self_loop(g)

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
	model = GraphSMOTE(in_feats, 16, n_classes, config["num_layers"], F.relu, 0.5)
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	# Training loop

	node_features = g.ndata["feat"]
	labels = g.ndata["label"]
	for epoch in tqdm(range(config["num_epochs"]), desc='epoch'):
		model.train()
		# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
		# Predict and loss
		batch_pred, batch_labels = model(g, node_features, labels)
		loss = criterion(batch_pred, batch_labels) + model.edge_loss(g)
		acc = (torch.argmax(batch_pred, dim=1) == batch_labels).float().sum() / len(batch_pred)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		print('Epoch {:04d} | Loss {:.4f} | Accuracy {:.4f} |'.format(epoch, loss.item(), acc))
		print()

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description='Weighted Sampling SAGE')
	argparser.add_argument('--gpu', type=int, default=-1,
						   help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('-d', '--dataset', type=str, default='toy01')
	argparser.add_argument('--num-epochs', type=int, default=20)
	argparser.add_argument('--batch-size', type=int, default=1024)
	argparser.add_argument("--num-layers", type=int, default=1)

	args = argparser.parse_args()
	config = vars(args)

	main(config)
