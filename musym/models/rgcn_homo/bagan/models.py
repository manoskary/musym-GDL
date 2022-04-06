import torch
import torch.nn as nn
import torch.nn.functional as F

class AdjLoss(nn.Module):
	def __init__(self):
		super(AdjLoss, self).__init__()

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


class AdjacencyGenerator(nn.Module):
    def __init__(self, n_hidden, activation=torch.sigmoid, dropout=0.5):
        super(AdjacencyGenerator, self).__init__()
        self.adj_linear = nn.Linear(n_hidden, n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
                if self.linear.bias is not None:
                    nn.init.constant_(self.linear.bias, 0.)

    def forward(self, inputs, h):
        z = torch.cat((inputs, h), dim=0)
        z = self.adj_linear(z)
        z = F.normalize(z)
        z = self.dropout(z)
        out = torch.sigmoid(torch.mm(z, z.T))
        return out


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
		if not isinstance(adj, torch.sparse.Tensor):
			if len(adj.shape) == 3:
				neigh_feature = torch.bmm(adj, features) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
			else:
				neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
		# For Sparse Adjacency Matrices
		else:
			neigh_feature = torch.spmm(adj, features) / (torch.sparse.sum(adj, dim=1).to_dense().reshape(adj.shape[0], -1) + 1)

		# perform conv
		data = torch.cat([features, neigh_feature], dim=-1)
		combined = self.linear(data)
		return combined



# Generator
class Decoder(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation=F.relu, dropout=0.5):
        super(Decoder, self).__init__()

        self.n_hidden = n_hidden
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

        self.proj = nn.Linear(self.n_hidden, out_feats)
        self.adj_gen = AdjacencyGenerator(n_hidden)
        self.tanh = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
                if self.linear.bias is not None:
                    nn.init.constant_(self.linear.bias, 0.)

    def forward(self, adj, inputs):
        h = inputs
        for l, conv in enumerate(self.layers):
            h = conv(adj, h)
            h = self.activation(h)
            h = F.normalize(h)
            h = self.dropout(h)
        adj_out = self.adj_gen(inputs, h)
        h = self.proj(h)
        output = self.tanh(h)

        return adj_out, output # (c_dim, 64, 64)


# Discriminator
class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation=F.relu, dropout=0.5):
        super(Encoder, self).__init__()
        self.n_hidden = n_hidden
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

        self.fc_z1 = nn.Linear(n_hidden, out_feats)
        self.fc_z2 = nn.Linear(n_hidden, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

    def forward(self, adj, inputs):
        h = inputs
        for l, conv in enumerate(self.layers):
            h = conv(adj, h)
            h = self.activation(h)
            h = F.normalize(h)
            h = self.dropout(h)

        mu = self.fc_z1(h.view(-1, self.n_hidden))	# (1, 128*8*4*4)
        sigma = self.fc_z2(h.view(-1, self.n_hidden))
        return mu,sigma # by squeeze, get just float not float Tenosor


class Generator(nn.Module):
    def __init__(self, in_feats, n_hidden, out_samples, n_layers, activation, dropout):
        super(Generator, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.adj_init = nn.Linear(self.in_feats, self.in_feats)
        self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

        self.adj_gen = AdjacencyGenerator(n_hidden)
        self.convTrans = nn.Linear(n_hidden, out_samples)
        # self.convTrans4 = nn.ConvTranspose2d(n_hidden, out_feats, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
                if self.linear.bias is not None:
                    nn.init.constant_(self.linear.bias, 0.)

    def forward(self, inputs):
        h = inputs
        z = self.adj_init(h)
        adj = torch.mm(z, z.T)
        for l, conv in enumerate(self.layers):
            h = conv(adj, h)
            h = self.activation(h)
            h = F.normalize(h)
            h = self.dropout(h)
        new_adj = self.adj_gen(inputs, h)
        h = self.convTrans(h)
        output = self.tanh(h)
        return new_adj, output # (out_samples + in_samples, out_samples + in_samples) , (out_samples + in_samples, n_hidden)




class _ganLogits(nn.Module):
    '''
    Layer of the GAN logits of the discriminator
    The layer gets class logits as inputs and calculates GAN logits to
    differentiate real and fake images in a numerical stable way
    '''
    def __init__(self, num_classes):
        '''
        :param num_classes: Number of real data classes (10 for SVHN)
        '''
        super(_ganLogits, self).__init__()
        self.num_classes = num_classes

    def forward(self, class_logits):
        '''
        :param class_logits: Unscaled log probabilities of house numbers
        '''

        # Set gan_logits such that P(input is real | input) = sigmoid(gan_logits).
        # Keep in mind that class_logits gives you the probability distribution over all the real
        # classes and the fake class. You need to work out how to transform this multiclass softmax
        # distribution into a binary real-vs-fake decision that can be described with a sigmoid.
        # Numerical stability is very important.
        # You'll probably need to use this numerical stability trick:
        # log sum_i exp a_i = m + log sum_i exp(a_i - m).
        # This is numerically stable when m = max_i a_i.
        # (It helps to think about what goes wrong when...
        #   1. One value of a_i is very large
        #   2. All the values of a_i are very negative
        # This trick and this value of m fix both those cases, but the naive implementation and
        # other values of m encounter various problems)
        real_class_logits, fake_class_logits = torch.split(class_logits, self.num_classes, dim=1)
        fake_class_logits = torch.squeeze(fake_class_logits)

        max_val, _ = torch.max(real_class_logits, 1, keepdim=True)
        stable_class_logits = real_class_logits - max_val
        max_val = torch.squeeze(max_val)
        gan_logits = torch.log(torch.sum(torch.exp(stable_class_logits), 1)) + max_val - fake_class_logits

        return gan_logits	# [128]


class Discriminator(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(Discriminator, self).__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(SageConvLayer(self.in_feats, self.n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SageConvLayer(self.n_hidden, self.n_hidden))

        self.fc_aux = nn.Linear(n_hidden, n_classes+1)
        self.softmax = nn.LogSoftmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, adj, inputs):
        h = inputs
        for l, conv in enumerate(self.layers):
            h = conv(adj, h)
            h = self.activation(h)
            h = F.normalize(h)
            h = self.dropout(h)
        output = self.softmax(self.fc_aux(h.view(-1, self.n_hidden)))
        return h, output

