import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import dgl.nn as dglnn
from random import randint
import random


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
			# For Mini-Batch Training exclude examples with less than k occurances in the mini banch.
			if i != dominant_class and occ[i] >= self.k:
				# calculate the amount of synthetic data to generate
				N = (n_occ - occ[i]) * 100 / occ[i]
				if N != 0:
					candidates = X[y == i]
					xs = self.generate(candidates, N, self.k)
					# TODO Possibility to add Gaussian noise here for ADASYN approach, important for mini-batch training with respect to the max euclidian distance.
					X = torch.cat((X, xs.to(X.get_device()))) if X.get_device() >= 0 else torch.cat((X, xs))
					ys = torch.ones(xs.shape[0]) * i
					y = torch.cat((y, ys.to(y.get_device()))) if y.get_device() >= 0 else torch.cat((y, ys))
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

	def forward(self, adj, features, neigh_feats):
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
			if len(neigh_feats) != len(features):
				adj.fill_diagonal_(1)
			if len(adj.shape) == 3:
				neigh_feature = torch.bmm(adj, neigh_feats) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
			else:
				neigh_feature = torch.mm(adj, neigh_feats) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
		# For Sparse Adjacency Matrices
		else:
			neigh_feature = torch.spmm(adj, neigh_feats) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

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
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.clf.weight, gain=nn.init.calculate_gain('relu'))
		if self.clf.bias is not None:
			nn.init.constant_(self.clf.bias, 0.)

	def forward(self, adj, inputs, neigh_feats):
		h = inputs
		for l, conv in enumerate(self.layers):
			h = conv(adj, h, neigh_feats)
			if l != self.n_layers-1:
				h = self.activation(h)
				h = F.normalize(h)
				h = self.dropout(h)
		h = self.clf(h)
		return h


class SageDecoder(nn.Module):
	def __init__(self, num_hidden, num_feats, dropout=0.1):
		super(SageDecoder, self).__init__()
		self.dropout = dropout
		self.de_weight = nn.Parameter(torch.FloatTensor(num_hidden, num_hidden))
		self.enc_weight = nn.Parameter(torch.FloatTensor(num_feats, num_hidden))
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.de_weight.size(1))
		self.de_weight.data.uniform_(-stdv, stdv)
		stdv = 1. / math.sqrt(self.enc_weight.size(1))
		self.enc_weight.data.uniform_(-stdv, stdv)

	def forward(self, encoded_features, previous_features):
		out = F.linear(encoded_features, self.de_weight)
		out_prev = F.linear(previous_features, self.enc_weight)
		adj_out = torch.sigmoid(torch.mm(out, out_prev.transpose(-1, -2)))
		return adj_out


class EdgeLoss(nn.Module):
	def __init__(self):
		super(EdgeLoss, self).__init__()

	def forward(self, adj_rec, adj_tgt, adj_mask = None):
		adj_rec = adj_rec[:adj_tgt.shape[0], :adj_tgt.shape[1]]
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
		self.layers.append(dglnn.SAGEConv(self.in_feats, self.n_hidden, aggregator_type="pool"))
		for i in range(n_layers - 1):
			self.layers.append(dglnn.SAGEConv(self.n_hidden, self.n_hidden, aggregator_type="pool"))

	def forward(self, blocks, inputs):
		h = inputs
		enc_feat_input = h
		for l, (conv, block) in enumerate(zip(self.layers, blocks)):
			h = conv(block, h)
			if l != len(self.layers) - 1:
				h = self.activation(h)
				h = F.normalize(h)
				h = self.dropout(h)
			if l == len(self.layers) - 2 and len(self.layers) > 1:
				enc_feat_input = h
		return h, enc_feat_input

	def decode(self, h):
		h = self.dropout(h)
		adj = torch.matmul(h, h.t())
		return adj


class LMCL_loss(nn.Module):
    """
        Refer to paper:
        Hao Wang, Yitong Wang, Zheng Zhou, Xing Ji, Dihong Gong, Jingchao Zhou,Zhifeng Li, and Wei Liu
        CosFace: Large Margin Cosine Loss for Deep Face Recognition. CVPR2018
        re-implement by yirong mao
        2018 07/02
    """

    def __init__(self, num_classes, feat_dim, s=7.00, m=0.2):
        super(LMCL_loss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)
        logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

        y_onehot = torch.FloatTensor(batch_size, self.num_classes)
        y_onehot.zero_()
        y_onehot = torch.autograd.Variable(y_onehot).cuda()
        y_onehot.scatter_(1, torch.unsqueeze(label, dim=-1), self.m)
        margin_logits = self.s * (logits - y_onehot)

        return logits, margin_logits


class ContrastiveGraphSMOTE(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.1):
		super(ContrastiveGraphSMOTE, self).__init__()
		self.n_layers = n_layers
		self.n_classes = n_classes
		self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout)
		if n_layers > 1:
			dec_feats = n_hidden
		else:
			dec_feats = in_feats
		self.decoder = SageDecoder(n_hidden, dec_feats, dropout)
		self.proj_head = nn.Linear(n_hidden, int(n_hidden/4))
		self.smote = SMOTE(dims=n_hidden, k=3)
		self.decoder_loss = EdgeLoss()
		nn.init.xavier_uniform_(self.proj_head.weight, gain=nn.init.calculate_gain('relu'))

	def forward(self, blocks, input_feats, adj, batch_labels):
		x = input_feats
		x, prev_feats = self.encoder(blocks, x)
		x, y = self.smote.fit_generate(x, batch_labels)
		pred_adj = self.decoder(x, prev_feats)
		x = self.proj_head(x)
		loss = self.decoder_loss(pred_adj, adj)
		return x, y.type(torch.long), loss


