import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import dgl.nn as dglnn
from random import randint
import random
import numpy as np
import pickle

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


class DQNSolver(nn.Module):
	"""
    Convolutional Neural Net with 3 conv layers and two linear layers
    """
	def __init__(self, input_shape, n_hidden, n_actions):
		super(DQNSolver, self).__init__()
		self.layers = nn.ModuleList()
		self.layers.append(nn.Linear(input_shape, n_hidden))
		self.layers.append(nn.Linear(n_hidden, n_hidden))
		self.layers.append(nn.Linear(n_hidden, n_actions))
		self.reset_parameters()

	def reset_parameters(self):
		for i, l in enumerate(self.layers):
			nn.init.xavier_uniform_(self.layers[i].weight, gain=nn.init.calculate_gain('relu'))


	def forward(self, x):
		h = x
		for i, layer in enumerate(self.layers):
			h = layer(h)
			if i != len(self.layers)-1:
				h = F.relu(h)
		return h


class DQNAgent:
	def __init__(self, state_space, n_hidden, action_space, max_memory_size, gamma, epsilon, deltaK):

		# Define DQN Layers
		self.state_space = state_space
		self.action_space = action_space

		# DQN network
		self.dqn = DQNSolver(state_space, action_space)

		if self.pretrained:
			self.dqn.load_state_dict(torch.load("DQN.pt"))
		self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=0.001)

		# Create memory
		self.max_memory_size = max_memory_size
		if self.pretrained:
			self.STATE_MEM = torch.load("STATE_MEM.pt")
			self.ACTION_MEM = torch.load("ACTION_MEM.pt")
			self.REWARD_MEM = torch.load("REWARD_MEM.pt")
			self.STATE2_MEM = torch.load("STATE2_MEM.pt")
			self.DONE_MEM = torch.load("DONE_MEM.pt")
			with open("ending_position.pkl", 'rb') as f:
				self.ending_position = pickle.load(f)
			with open("num_in_queue.pkl", 'rb') as f:
				self.num_in_queue = pickle.load(f)
		else:
			self.STATE_MEM = torch.zeros(max_memory_size, *self.state_space)
			self.ACTION_MEM = torch.zeros(max_memory_size, 1)
			self.REWARD_MEM = torch.zeros(max_memory_size, 1)
			self.STATE2_MEM = torch.zeros(max_memory_size, *self.state_space)
			self.DONE_MEM = torch.zeros(max_memory_size, 1)
			self.ending_position = 0
			self.num_in_queue = 0

		# Learning parameters
		self.gamma = gamma
		self.l1 = nn.SmoothL1Loss()  # Also known as Huber loss

	def remember(self, state, action, reward, state2, done):
		"""Store the experiences in a buffer to use later"""
		self.STATE_MEM[self.ending_position] = state.float()
		self.ACTION_MEM[self.ending_position] = action.float()
		self.REWARD_MEM[self.ending_position] = reward.float()
		self.STATE2_MEM[self.ending_position] = state2.float()
		self.DONE_MEM[self.ending_position] = done.float()
		self.ending_position = (self.ending_position + 1) % self.max_memory_size  # FIFO tensor
		self.num_in_queue = min(self.num_in_queue + 1, self.max_memory_size)

	def batch_experiences(self):
		"""Randomly sample 'batch size' experiences"""
		idx = random.choices(range(self.num_in_queue), k=self.memory_sample_size)
		STATE = self.STATE_MEM[idx]
		ACTION = self.ACTION_MEM[idx]
		REWARD = self.REWARD_MEM[idx]
		STATE2 = self.STATE2_MEM[idx]
		DONE = self.DONE_MEM[idx]
		return STATE, ACTION, REWARD, STATE2, DONE

	def act(self, state):
		"""Epsilon-greedy action"""
		if random.random() < self.exploration_rate:
			return torch.tensor([[random.randrange(self.action_space)]])
		else:
			return torch.argmax(self.dqn(state.to(self.device))).unsqueeze(0).unsqueeze(0).cpu()

	def experience_replay(self):
		if self.memory_sample_size > self.num_in_queue:
			return

		# Sample a batch of experiences
		STATE, ACTION, REWARD, STATE2, DONE = self.batch_experiences()
		STATE = STATE.to(self.device)
		ACTION = ACTION.to(self.device)
		REWARD = REWARD.to(self.device)
		STATE2 = STATE2.to(self.device)
		DONE = DONE.to(self.device)

		self.optimizer.zero_grad()
		# Q-Learning target is Q*(S, A) <- r + γ max_a Q(S', a)
		target = REWARD + torch.mul((self.gamma * self.dqn(STATE2).max(1).values.unsqueeze(1)), 1 - DONE)
		current = self.dqn(STATE).gather(1, ACTION.long())

		loss = self.l1(current, target)
		loss.backward()  # Compute gradients
		self.optimizer.step()  # Backpropagate error

		self.exploration_rate *= self.exploration_decay

		# Makes sure that exploration rate is always at least 'exploration min'
		self.exploration_rate = max(self.exploration_rate, self.exploration_min)


class GraphMixup(nn.Module):
	def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.1):
		super(GraphMixup, self).__init__()
		self.n_layers = n_layers
		self.n_classes = n_classes
		self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout)
		if n_layers > 1:
			dec_feats = n_hidden
		else:
			dec_feats = in_feats
		self.decoder = SageDecoder(n_hidden, dec_feats, dropout)
		self.linear = nn.Linear(( n_hidden if n_layers > 1 else in_feats), n_hidden)
		self.agent = DQNAgent(n_classes, n_hidden, 2, 20, 1, 0.9, 0.05)
		self.classifier = SageClassifier(n_hidden, n_hidden, n_classes, n_layers=1, activation=activation, dropout=dropout)
		self.smote = SMOTE(dims=n_hidden, k=3)
		self.decoder_loss = EdgeLoss()

	def forward(self, blocks, input_feats, adj, batch_labels):
		x = input_feats
		x, prev_feats = self.encoder(blocks, x)
		x, y = self.smote.fit_generate(x, batch_labels)
		pred_adj = self.decoder(x, prev_feats)
		loss = self.decoder_loss(pred_adj, adj)
		dum =  torch.tensor(0, dtype=pred_adj.dtype).to(pred_adj.get_device()) if pred_adj.get_device() >= 0 else torch.tensor(0, dtype=pred_adj.dtype)
		pred_adj = torch.where(pred_adj >= 0.5, pred_adj, dum)
		x = self.classifier(pred_adj, x, prev_feats)
		return x, y.type(torch.long), loss

	def generate_episode(self, agent, max_steps, render=False):
		"""
		Generates one episode (S0, A0, R1, ..., RT) on a given
		environment following the agent's policy.

		Inputs:
			@env (OpenAI Gym environment): the environment to interact with
			@agent (as defined above): the agent (with the policy it follows)
			@max_steps (integer): maximum number of steps in episode
			@render (boolean): render environment after each action

		Returns:
			@states (float array): sequence of observed states
			@actions (float array): sequence of selected actions
			@rewards (float array): sequence of received rewards
			@total_reward (numerical): total accumulated reward of an episode
		"""

		done = False
		states  = []
		actions = []
		rewards = []
		for t in range(max_steps):
			states.append(state)
			actions.append(agent.choose_action(state))
			state, reward, done, _ = actions[t]
			rewards.append(reward)

			if done:
				#print("Episode finished after {} timesteps".format(t+1))
				break

		return np.array(states), np.array(actions), np.array(rewards), np.array(rewards).sum()

