import torch
import torch.nn as nn
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
	def __init__(self, distance='euclidian', dims=512, k=2):
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


class MBSMOTE(object):
	"""
		Minority Sampling with SMOTE.
	"""

	def __init__(self, n_classes, distance='euclidian', dims=512, k=2, epsilon=1):
		super(MBSMOTE, self).__init__()
		self.newindex = 0
		self.k = k
		self.dims = dims
		self.distance_measure = distance
		self.n_classes = n_classes
		# This could be a Linear Layer
		self.linear = nn.Linear(dims*2, dims)
		nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
		self.epsilon = epsilon
		self.centers = torch.zeros((n_classes, self.dims))

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
		device = X.get_device()
		if device < 0:
			device = "cpu"
		# get the dominant class
		dominant_class = torch.argmax(occ)
		# get occurence of the dominant class
		n_occ = int(occ[dominant_class].item())
		for i in range(len(occ)):
			# For Mini-Batch Training exclude examples with less than k occurances in the mini banch.
			if i != dominant_class and occ[i] > 0:
				center = self.update_centers(X, y, i, device)
				# calculate the amount of synthetic data to generate
				N = (n_occ - occ[i]) * 100 / occ[i]
				if N != 0:
					candidates = torch.cat((X[y==i], center.unsqueeze(0).to(device)))
					xs = self.generate(candidates, N, self.k)
					X = torch.cat((X, xs.to(device)))
					ys = torch.ones(xs.shape[0]) * i
					y = torch.cat((y, ys.to(device)))
		return X, y

	def update_centers(self, X, y, i, device):
		if torch.all(self.centers[i] == 0):
			center = self.barycenter(X[y==i]).to(device)
		else:
			#TODO Maybe add mean Euclidean Distance instead
			center = self.linear(torch.cat((self.barycenter(X[y == i]), self.centers[i])).to(device))
			# center = (self.barycenter(X[y==i]) + self.centers[i])/2
		return center

	def barycenter(self, x, y=None):
		return x.mean(0)

# Graphsage layer
class SageConvLayer(nn.Module):
	def __init__(self, in_features, out_features, bias=False):
		super(SageConvLayer, self).__init__()
		self.neigh_linear = nn.Linear(in_features, in_features, bias=bias)
		self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.neigh_linear.weight, gain=nn.init.calculate_gain('relu'))
		if self.linear.bias is not None:
			nn.init.constant_(self.linear.bias, 0.)
		if self.neigh_linear.bias is not None:
			nn.init.constant_(self.neigh_linear.bias, 0.)

	def forward(self, adj, features, neigh_feats=None):
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
		if not neigh_feats:
			neigh_feats = features
		h = self.neigh_linear(neigh_feats)
		if not isinstance(adj, torch.sparse.FloatTensor):
			# NOTE: Diagonal with a rectangular adjacency doesn't make sense and raises a backward pass error.
			if len(adj.shape) == 3:
				h = torch.bmm(adj, h) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
			else:
				h = torch.mm(adj, h) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
		# For Sparse Adjacency Matrices
		else:
			h = torch.mm(adj, h) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

		# perform conv
		z = self.linear(torch.cat([features, h], dim=-1))
		return z


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
		self.linear = nn.Linear(n_hidden, n_hidden)
		self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
		for i in range(n_layers - 1):
			self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.clf.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
		if self.clf.bias is not None:
			nn.init.constant_(self.clf.bias, 0.)

	def forward(self, adj, inputs, neigh_feats):
		h = inputs
		for l, conv in enumerate(self.layers):
			h = conv(adj, h, neigh_feats)
			# Is activation on the correct position?
			if l != self.n_layers - 1:
				h = self.activation(h)
			h = F.normalize(h)
			h = self.dropout(h)
		h = self.clf(h)
		# Added softmax
		# return F.softmax(h, dim=1)
		return h


class SageDecoder(nn.Module):
	def __init__(self, num_hidden, num_feats, dropout=0.1):
		super(SageDecoder, self).__init__()
		self.dropout = dropout
		self.layer_1 = nn.Linear(num_hidden, num_hidden)
		self.layer_2 = nn.Linear(num_feats, num_hidden)
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.layer_1.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_uniform_(self.layer_2.weight, gain=nn.init.calculate_gain('relu'))

	def forward(self, encoded_features, previous_features):
		out = self.layer_1(encoded_features)
		out_prev = self.layer_2(previous_features)
		adj_out = torch.mm(out, out_prev.T)
		adj_out = torch.sigmoid(adj_out)
		return adj_out


class EdgeLoss(nn.Module):
	def __init__(self):
		super(EdgeLoss, self).__init__()

	def forward(self, adj_rec, adj_tgt, adj_mask = None):
		if adj_tgt.is_sparse:
			shape = adj_tgt.size()
			indices = adj_tgt._indices().T
		else:
			shape = adj_tgt.shape
			indices = adj_tgt.nonzero()

		edge_num = indices.shape[0]
		total_num = shape[0] * shape[1]

		new_adj = torch.transpose(adj_rec[:shape[1], :shape[0]], 0, 1)
		neg_weight = edge_num / (total_num - edge_num)
		# weight_matrix = new_adj.new(shape).fill_(neg_weight)
		weight_matrix = torch.empty_like(new_adj).fill_(neg_weight)
		weight_matrix[indices] = 1.0
		loss = torch.sum(weight_matrix * torch.subtract(new_adj, adj_tgt) ** 2)
		return loss


class GaugLoss(nn.Module):
	def __init__(self):
		super(GaugLoss, self).__init__()

	def forward(self, adj_rec, adj_tgt):
		if adj_tgt.is_sparse:
			shape = adj_tgt.size()
			indices = adj_tgt._indices().T
		else:
			shape = adj_tgt.shape
			indices = adj_tgt.nonzero()
		adj_sum = torch.sparse.sum(adj_tgt)
		bce_weight = (shape[0]*shape[1] - adj_sum) / adj_sum
		norm_w = shape[0]*shape[1] / float((shape[0]*shape[1] - adj_sum) * 2)
		bce_loss = norm_w * F.binary_cross_entropy_with_logits(torch.transpose(adj_rec[:shape[1], :shape[0]], 0, 1), adj_tgt.to_dense(), pos_weight=bce_weight)
		return bce_loss


class FullGraphEncoder(nn.Module):
	def __init__(self, in_feats, n_hidden, n_layers, activation, dropout):
		super(FullGraphEncoder, self).__init__()
		self.in_feats = in_feats
		self.n_hidden = n_hidden
		self.activation = activation

		self.dropout = nn.Dropout(dropout)
		self.layers = nn.ModuleList()
		self.attention = False
		self.out = self.n_hidden
		self.layers.append(SageConvLayer(self.in_feats, self.out))
		for i in range(n_layers - 2):
			self.layers.append(SageConvLayer(self.n_hidden, self.out))
		self.layers.append(SageConvLayer(self.out, self.out))

	def forward(self, adj, inputs):
		h = inputs
		enc_feat_input = h
		for l, conv in enumerate(self.layers):
			h = conv(adj, h)
			if l == len(self.layers) - 2 and len(self.layers) > 1:
				enc_feat_input = h
				# Should I put normalization and dropout here?
				enc_feat_input = F.normalize(enc_feat_input)
				enc_feat_input = self.dropout(enc_feat_input)
			if l != len(self.layers) - 1:
				h = self.activation(h)
				h = F.normalize(h)
				h = self.dropout(h)
		return h, enc_feat_input


class Encoder(nn.Module):
	def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, ext_mode=None):
		super(Encoder, self).__init__()
		self.in_feats = in_feats
		self.n_hidden = n_hidden
		self.activation = activation

		self.dropout = nn.Dropout(dropout)
		self.layers = nn.ModuleList()
		self.attention = False
		if ext_mode == "attention":
			self.attention = True
			self.num_heads = 5
			self.layers.append(
				dglnn.GATConv(self.in_feats, self.n_hidden, num_heads=self.num_heads, allow_zero_in_degree=True,
							  negative_slope=0.2, feat_drop=dropout, attn_drop=dropout, activation=activation))
			for i in range(n_layers - 2):
				self.layers.append(
					dglnn.GATConv(self.n_hidden*self.num_heads, self.n_hidden, num_heads=self.num_heads, allow_zero_in_degree=True,
								  negative_slope=0.2, feat_drop=dropout, attn_drop=dropout, activation=activation))
			self.layers.append(
				dglnn.GATConv(self.n_hidden*self.num_heads, self.n_hidden, num_heads=self.num_heads, allow_zero_in_degree=True,
							  negative_slope=0.2, feat_drop=dropout, attn_drop=dropout, activation=None))
		else:
			if ext_mode == "lstm":
				aggregator_type = "lstm"
			else:
				aggregator_type = "pool"
			self.out = self.n_hidden
			self.layers.append(dglnn.SAGEConv(self.in_feats, self.out, aggregator_type=aggregator_type))
			for i in range(n_layers - 2):
				# self.out = int(self.out / 2)
				self.layers.append(dglnn.SAGEConv(self.n_hidden, self.out, aggregator_type=aggregator_type))
			self.layers.append(dglnn.SAGEConv(self.out, self.out, aggregator_type=aggregator_type))

	def forward(self, blocks, inputs):
		h = inputs
		enc_feat_input = h
		for l, (conv, block) in enumerate(zip(self.layers, blocks)):
			h = conv(block, h)
			if l == len(self.layers) - 2 and len(self.layers) > 1:
				enc_feat_input = h
				# Should I put normalization and dropout here?
				enc_feat_input = F.normalize(enc_feat_input)
				enc_feat_input = self.dropout(enc_feat_input)
			if l != len(self.layers) - 1:
				if self.attention:
					h = h.flatten(1)
				else:
					h = self.activation(h)
				# Should I put normalization and dropout here?
				h = F.normalize(h)
				h = self.dropout(h)
		if self.attention:
			h = h.mean(1)
			enc_feat_input = enc_feat_input.mean(1)
		return h, enc_feat_input

	def decode(self, h):
		h = self.dropout(h)
		adj = torch.matmul(h, h.t())
		return adj


class GraphSMOTE(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.1, ext_mode=None, adj_thresh=0.01):
		super(GraphSMOTE, self).__init__()
		self.n_layers = n_layers
		self.n_classes = n_classes
		self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout, ext_mode)
		# n_hidden = int((n_hidden / (2**(n_layers - 2)))) if n_layers >= 2 else n_hidden
		if n_layers > 1:
			dec_feats = n_hidden
		else:
			dec_feats = in_feats
		self.decoder = SageDecoder(n_hidden, dec_feats, dropout)
		self.linear = nn.Linear(( n_hidden if n_layers > 1 else in_feats), n_hidden)
		self.classifier = SageClassifier(n_hidden, n_hidden, n_classes, n_layers=1, activation=activation, dropout=dropout)
		# self.smote = MBSMOTE(n_classes=n_classes, dims=n_hidden, k=2)
		self.smote = SMOTE(dims=n_hidden, k=3)
		self.decoder_loss = GaugLoss()
		self.adj_thresh = adj_thresh

	def forward(self, adj, input_feats, batch_labels):
		x = input_feats
		x, prev_feats = self.encoder(adj, x)
		x, y = self.smote.fit_generate(x, batch_labels)
		pred_adj = self.decoder(x, prev_feats)
		loss = self.decoder_loss(pred_adj, adj)
		pred_adj = F.hardshrink(pred_adj, lambd=self.adj_thresh)
		x = self.classifier(pred_adj, x, prev_feats)
		return x, y.type(torch.long), loss

	def inference(self, dataloader, node_features, labels, device):
		prediction = torch.zeros(len(labels), self.n_classes).to(device)
		with torch.no_grad():
			for input_nodes, seeds, mfgs in tqdm(dataloader, position=0, leave=True):
				batch_inputs = node_features[input_nodes].to(device)
				mfgs = [mfg.int().to(device) for mfg in mfgs]
				batch_pred, prev_encs = self.encoder(mfgs, batch_inputs)
				pred_adj = F.hardshrink(self.decoder(batch_pred, prev_encs), lambd=self.adj_thresh)
				prediction[seeds] = F.softmax(self.classifier(pred_adj, batch_pred, prev_encs), dim=1)
			return prediction

	def edge_inference(self, g, node_features, labels, device, batch_size, num_workers=0):
		g.ndata['idx'] = torch.tensor(range(g.number_of_nodes()))
		sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
		dataloader = dgl.dataloading.EdgeDataLoader(
			g,
			torch.arange(g.number_of_edges()),
			sampler,
			device=device,
			batch_size=batch_size,
			shuffle=False,
			drop_last=False,
			num_workers=num_workers)
		y = torch.zeros(g.num_nodes(), self.n_classes)

		for input_nodes, sub_g, blocks in tqdm(dataloader, position=0, leave=True):
			blocks = [block.int().to(device) for block in blocks]
			batch_inputs = node_features[input_nodes].to(device)
			batch_labels = labels[sub_g.ndata["idx"]].to(device)

			# feat_inputs = sub_g.ndata["feat"].to(device)

			adj = sub_g.adj(ctx=device).to_dense()
			h = self.encoder(blocks, batch_inputs)
			pred_adj = F.hardshrink(self.decoder(h), lambd=self.adj_thresh)
			h = self.classifier(pred_adj, h)
			# TODO prediction may replace values because Edge dataloder repeats nodes, maybe take average or addition.
			y[sub_g.ndata['idx']] = h.cpu()[:len(batch_labels)]
		return y




class FullGraphSMOTE(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.1, ext_mode=None, adj_thresh=0.01):
		super(FullGraphSMOTE, self).__init__()
		self.n_layers = n_layers
		self.n_classes = n_classes
		self.encoder = FullGraphEncoder(in_feats, n_hidden, n_layers, activation, dropout)
		# n_hidden = int((n_hidden / (2**(n_layers - 2)))) if n_layers >= 2 else n_hidden
		if n_layers > 1:
			dec_feats = n_hidden
		else:
			dec_feats = in_feats
		self.decoder = SageDecoder(n_hidden, dec_feats, dropout)
		self.classifier = SageClassifier(n_hidden, n_hidden, n_classes, n_layers=1, activation=activation, dropout=dropout)
		self.smote = SMOTE(dims=n_hidden, k=5)
		self.decoder_loss = GaugLoss()
		self.adj_thresh = adj_thresh

	def forward(self, adj, input_feats, batch_labels):
		x = input_feats
		x, prev_feats = self.encoder(adj, x)
		x, y = self.smote.fit_generate(x, batch_labels)
		pred_adj = self.decoder(x, prev_feats)
		loss = self.decoder_loss(pred_adj, adj)
		# Thesholding Adjacency with Harshrink since sigmoid output is positive.
		pred_adj = F.hardshrink(pred_adj, lambd=self.adj_thresh)
		x = self.classifier(pred_adj, x, prev_feats)
		return x, y.type(torch.long), loss

	def inference(self, dataloader, node_features, labels, device):
		prediction = list()
		with torch.no_grad():
			for graph in tqdm(dataloader, position=0, leave=True):
				batch_inputs = graph.ndata["feat"].to(device)
				adj = graph.adj()
				batch_pred, prev_encs = self.encoder(adj, batch_inputs)
				pred_adj = F.hardshrink(self.decoder(batch_pred, prev_encs), lambd=self.adj_thresh)
				prediction.append(F.softmax(self.classifier(pred_adj, batch_pred, prev_encs), dim=1))
			return torch.cat(prediction, dim=0)