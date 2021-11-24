import torch
import torch.nn as nn
import math
from tqdm import tqdm
import torch.nn.functional as F
import dgl.nn as dglnn
from random import randint
import random
import dgl

class SMOTE(object):
	"""
	Minority Sampling with SMOTE.
	"""
	def __init__(self, distance='euclidian', dims=512, k=5):
		super(SMOTE, self).__init__()
		self.newindex = 0
		self.k = k
		self.dims = dims
		self.distance_measure = distance

	def populate(self, N, i, nnarray, min_samples, k):
		while N:
			nn = randint(0, k - 2)

			diff = min_samples[nnarray[nn]] - min_samples[i]
			gap = random.uniform(0, 1)

			self.synthetic_arr[self.newindex, :] = min_samples[i] + gap * diff

			self.newindex += 1

			N -= 1

	def k_neighbors(self, euclid_distance, k):
		nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64)

		idxs = torch.argsort(euclid_distance, dim=1)
		nearest_idx[:, :] = idxs

		return nearest_idx[:, 1:k]

	def find_k(self, X, k):
		euclid_distance = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32)

		for i in range(len(X)):
			dif = (X - X[i]) ** 2
			dist = torch.sqrt(dif.sum(axis=1))
			euclid_distance[i] = dist

		return self.k_neighbors(euclid_distance, k)

	def generate(self, min_samples, N, k):
		"""
		Returns (N/100) * n_minority_samples synthetic minority samples.
		Parameters
		----------
		min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
			Holds the minority samples
		N : percetange of new synthetic samples:
			n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
		k : int. Number of nearest neighbours.
		Returns
		-------
		S : Synthetic samples. array,
			shape = [(N/100) * n_minority_samples, n_features].
		"""
		T = min_samples.shape[0]
		self.synthetic_arr = torch.zeros(int(N / 100) * T, self.dims)
		N = int(N / 100)
		if self.distance_measure == 'euclidian':
			indices = self.find_k(min_samples, k)
		for i in range(indices.shape[0]):
			self.populate(N, i, indices[i], min_samples, k)
		self.newindex = 0
		return self.synthetic_arr

	def fit_generate(self, X, y):
		# get occurence of each class
		occ = torch.eye(int(y.max() + 1), int(y.max() + 1))[y].sum(axis=0)
		# get the dominant class
		dominant_class = torch.argmax(occ)
		# get occurence of the dominant class
		n_occ = int(occ[dominant_class].item())
		for i in range(len(occ)):
			if i != dominant_class:
				# calculate the amount of synthetic data to generate
				N = (n_occ - occ[i]) * 100 / occ[i]
				if N != 0:
					candidates = X[y == i]
					xs = self.generate(candidates, N, self.k)
					X = torch.cat((X, xs.to(X.get_device())))
					ys = torch.ones(xs.shape[0]) * i
					y = torch.cat((y, ys.to(y.get_device())))
		return X, y

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
				h = F.normalize(h)
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
				h = F.normalize(h)
				h = self.dropout(h)
		return h

	def decode(self, h):
		h = self.dropout(h)
		adj = torch.matmul(h, h.t())
		return adj


class GraphSMOTE(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.1):
		super(GraphSMOTE, self).__init__()
		self.n_layers = n_layers
		self.n_classes = n_classes
		self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout)
		self.decoder = SageDecoder(n_hidden, dropout)
		self.classifier = SageClassifier(n_hidden, n_hidden, n_classes, n_layers=1, activation=activation, dropout=dropout)
		self.smote = SMOTE(dims=n_hidden, k=2)
		self.decoder_loss = EdgeLoss()

	def forward(self, blocks, input_feats, adj, batch_labels):
		x = input_feats
		x = self.encoder(blocks, x)
		x, y = self.smote.fit_generate(x, batch_labels)
		pred_adj = self.decoder(x)
		loss = self.decoder_loss(pred_adj, adj)
		pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype).to(pred_adj.get_device()))
		x = self.classifier(pred_adj, x)
		return x, y.type(torch.long), loss

	#TODO fix that.
	def inference(self, g, device, batch_size, num_workers=0):
		g.ndata['idx'] = torch.tensor(range(g.number_of_nodes()))
		node_features = g.ndata['feat']
		labels = g.ndata["label"]
		sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
		dataloader = dgl.dataloading.EdgeDataLoader(
			g,
			torch.arange(g.number_of_edges()),
			sampler,
			batch_size=batch_size,
			shuffle=False,
			drop_last=False,
			num_workers=num_workers)
		y = torch.zeros(g.num_nodes(), self.n_classes)

		for input_nodes, sub_g, blocks in tqdm(dataloader, position=0, leave=True):
			blocks = [block.int().to(device) for block in blocks]
			batch_inputs = node_features[input_nodes].to(device)
			batch_labels = labels[input_nodes].to(device)

			# feat_inputs = sub_g.ndata["feat"].to(device)

			adj = sub_g.adj(ctx=device).to_dense()
			h = self.encoder(blocks, batch_inputs)
			pred_adj = self.decoder(h)
			pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype).to(h.get_device()))
			h = self.classifier(pred_adj, h)
			# TODO prediction may replace values because Edge dataloder repeats nodes, maybe take average or addition.
			y[sub_g.ndata['idx']] = h.cpu()[:len(batch_labels)]
		return y