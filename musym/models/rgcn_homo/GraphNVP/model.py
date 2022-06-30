import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl


class GraphFlow(nn.Module):
    """
    Generic class for GraphFlow functions.
    """
    def __init__(self):
        super(GraphFlow, self).__init__()

    def forward(self, z, adj):
        """
        Parameters:
        -----------
        z : torch tensor
            Input variable, first dimension is batch dim.
        adj : torch tensor
            Input variable, first dimension is batch dim.
        """
        raise NotImplementedError('Forward pass has not been implemented.')

    def inverse(self, z, adj):
        raise NotImplementedError('This flow has no algebraic inverse.')


class Split(GraphFlow):
    """
    Split features into two sets
    """
    def __init__(self, mode='channel'):
        """
        Constructor
        :param mode: Splitting mode, can be
            channel: Splits first feature dimension, usually channels, into two halfs
            channel_inv: Same as channel, but with z1 and z2 flipped
            checkerboard: Splits features using a checkerboard pattern (last feature dimension must be even)
            checkerboard_inv: Same as checkerboard, but with inverted coloring
        """
        super().__init__()
        self.mode = mode

    def forward(self, z, adj):
        if self.mode == 'channel':
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == 'channel_inv':
            z2, z1 = z.chunk(2, dim=1)
        elif 'checkerboard' in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(*z_size[:-1], -1)
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(*z_size[:-1], -1)
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z, adj):
        z1, z2 = z
        if self.mode == 'channel':
            z = torch.cat([z1, z2], 1)
        elif self.mode == 'channel_inv':
            z = torch.cat([z2, z1], 1)
        elif 'checkerboard' in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if 'inv' in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError('Mode ' + self.mode + ' is not implemented.')
        log_det = 0
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """
    def __init__(self, mode='channel'):
        super().__init__(mode)

    def forward(self, z, adj):
        return super().inverse(z, adj)

    def inverse(self, z, adj):
        return super().forward(z, adj)


class Squeeze(GraphFlow):
    """
    Squeeze operation of multi-scale architecture, RealNVP or Glow paper
    """
    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def forward(self, z, adj):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z, adj):
        log_det = 0
        s = z.size()
        z = z.view(*s[:2], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det


class AffineCoupling(GraphFlow):
    """
    Affine Coupling layer as introduced RealNVP paper, see arXiv: 1605.08803
    """

    def __init__(self, param_map, scale=True, scale_map='exp'):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow, 'sigmoid_inv' uses multiplicative sigmoid
        scale when sampling from the model
        """
        super().__init__()
        self.add_module('param_map', param_map)
        self.scale = scale
        self.scale_map = scale_map

    def forward(self, z, adj):
        """
        z is a list of z1 and z2; z = [z1, z2]
        z1 is left constant and affine map is applied to z2 with parameters depending
        on z1
        """
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = z2 * torch.exp(scale_) + shift
                log_det = torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 / scale + shift
                log_det = -torch.sum(torch.log(scale),
                                     dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = z2 * scale + shift
                log_det = torch.sum(torch.log(scale),
                                    dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 += param
            log_det = 0
        return [z1, z2], log_det

    def inverse(self, z, adj):
        z1, z2 = z
        param = self.param_map(z1)
        if self.scale:
            shift = param[:, 0::2, ...]
            scale_ = param[:, 1::2, ...]
            if self.scale_map == 'exp':
                z2 = (z2 - shift) * torch.exp(-scale_)
                log_det = -torch.sum(scale_, dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) * scale
                log_det = torch.sum(torch.log(scale),
                                    dim=list(range(1, shift.dim())))
            elif self.scale_map == 'sigmoid_inv':
                scale = torch.sigmoid(scale_ + 2)
                z2 = (z2 - shift) / scale
                log_det = -torch.sum(torch.log(scale),
                                     dim=list(range(1, shift.dim())))
            else:
                raise NotImplementedError('This scale map is not implemented.')
        else:
            z2 -= param
            log_det = 0
        return [z1, z2], log_det


class AffineCouplingBlock(GraphFlow):
    """
    Affine Coupling layer including split and merge operation
    """
    def __init__(self, param_map, scale=True, scale_map='exp', split_mode='channel'):
        """
        Constructor
        :param param_map: Maps features to shift and scale parameter (if applicable)
        :param scale: Flag whether scale shall be applied
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Split layer
        self.flows += [Split(split_mode)]
        # Affine coupling layer
        self.flows += [AffineCoupling(param_map, scale, scale_map)]
        # Merge layer
        self.flows += [Merge(split_mode)]

    def forward(self, z, adj):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z, adj)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z, adj):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, adj)
            log_det_tot += log_det
        return z, log_det_tot


class AffineConstFlow(GraphFlow):
    """
    scales and shifts with learned constants per dimension. In the NICE paper there is a
    scaling layer which is a special case of this where t is None
    """

    def __init__(self, shape, scale=True, shift=True):
        """
        Constructor
        :param shape: Shape of the coupling layer
        :param scale: Flag whether to apply scaling
        :param shift: Flag whether to apply shift
        :param logscale_factor: Optional factor which can be used to control
        the scale of the log scale factor
        """
        super().__init__()
        if scale:
            self.s = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('s', torch.zeros(shape)[None])
        if shift:
            self.t = nn.Parameter(torch.zeros(shape)[None])
        else:
            self.register_buffer('t', torch.zeros(shape)[None])
        self.n_dim = self.s.dim()
        self.batch_dims = torch.nonzero(torch.tensor(self.s.shape) == 1, as_tuple=False)[:, 0].tolist()

    def forward(self, z, adj):
        z_ = z * torch.exp(self.s) + self.t
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = prod_batch_dims * torch.sum(self.s)
        return z_, log_det

    def inverse(self, z, adj):
        z_ = (z - self.t) * torch.exp(-self.s)
        if len(self.batch_dims) > 1:
            prod_batch_dims = np.prod([z.size(i) for i in self.batch_dims[1:]])
        else:
            prod_batch_dims = 1
        log_det = -prod_batch_dims * torch.sum(self.s)
        return z_, log_det


class ActNorm(AffineConstFlow):
    """
    An AffineConstFlow but with a data-dependent initialization,
    where on the very first batch we clever initialize the s,t so that the output
    is unit gaussian. As described in Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done_cpu = torch.tensor(0.)
        self.register_buffer('data_dep_init_done', self.data_dep_init_done_cpu)

    def forward(self, z, adj):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (-z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().forward(z, adj)

    def inverse(self, z, adj):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.)
        return super().inverse(z, adj)


class GraphGlowBlock(GraphFlow):
    """
    Glow: Generative Flow with Invertible 1×1 Convolutions, arXiv: 1807.03039
    One Block of the Glow model, comprised of
    MaskedAffineFlow (affine coupling layer
    Invertible1x1Conv (dropped if there is only one channel)
    ActNorm (first batch used for initialization)
    """
    def __init__(self, channels, hidden_channels, scale=True, scale_map='sigmoid',
                 split_mode='channel', leaky=0.0, init_zeros=True, use_lu=True,
                 net_actnorm=False):
        """
        Constructor
        :param channels: Number of channels of the data
        :param hidden_channels: number of channels in the hidden layer of the ConvNet
        :param scale: Flag, whether to include scale in affine coupling layer
        :param scale_map: Map to be applied to the scale parameter, can be 'exp' as in
        RealNVP or 'sigmoid' as in Glow
        :param split_mode: Splitting mode, for possible values see Split class
        :param leaky: Leaky parameter of LeakyReLUs of ConvNet2d
        :param init_zeros: Flag whether to initialize last conv layer with zeros
        :param use_lu: Flag whether to parametrize weights through the LU decomposition
        in invertible 1x1 convolution layers
        :param logscale_factor: Factor which can be used to control the scale of
        the log scale factor, see https://github.com/openai/glow
        """
        super().__init__()
        self.flows = nn.ModuleList([])
        # Coupling layer
        num_param = 2 if scale else 1
        if 'channel' == split_mode:
            channels_ = (channels // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * ((channels + 1) // 2),)
        elif 'channel_inv' == split_mode:
            channels_ = ((channels + 1) // 2,) + 2 * (hidden_channels,)
            channels_ += (num_param * (channels // 2),)
        elif 'checkerboard' in split_mode:
            channels_ = (channels,) + 2 * (hidden_channels,)
            channels_ += (num_param * channels,)
        else:
            raise NotImplementedError('Mode ' + split_mode + ' is not implemented.')
        # Have to add activation and normalization here.
        param_map = SageConvLayer(channels_, hidden_channels)
        self.flows += [AffineCouplingBlock(param_map, scale, scale_map, split_mode)]
        # Activation normalization
        self.flows += [ActNorm((channels,) + (1, 1))]

    def forward(self, z, adj):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for flow in self.flows:
            z, log_det = flow(z, adj)
            log_det_tot += log_det
        return z, log_det_tot

    def inverse(self, z, adj):
        log_det_tot = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z, adj)
            log_det_tot += log_det
        return z, log_det_tot


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

    def forward(self, features, adj, neigh_feats=None):
        if neigh_feats == None:
            neigh_feats = features
        h = self.neigh_linear(neigh_feats)
        if not isinstance(adj, torch.sparse.FloatTensor):
            if len(adj.shape) == 3:
                h = torch.bmm(adj, h) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                h = torch.mm(adj, h) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            h = torch.mm(adj, h) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)
        z = self.linear(torch.cat([features, h], dim=-1))
        return z


class Dequantization(nn.Module):
    def __init__(self, alpha=1e-5, quants=256):
        """
        Args:
            alpha: small constant that is used to scale the original input.
                    Prevents dealing with values very close to 0 and 1 when inverting the sigmoid
            quants: Number of possible discrete values (usually 256 for 8-bit image)
        """
        super(Dequantization, self).__init__()
        self.alpha = alpha
        self.quants = quants

    def forward(self, z, adj, ldj, reverse=False):
        if not reverse:
            z, ldj = self.dequant(z, adj, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * z.shape[1]
            z = torch.floor(z).clamp(min=0, max=self.quants - 1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z - 2 * F.softplus(-z)).sum(dim=1)
            z = torch.sigmoid(z)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * z.shape[1]
            ldj += (-torch.log(z) - torch.log(1 - z)).sum(dim=1)
            z = torch.log(z) - torch.log(1 - z)
        return z, ldj

    def dequant(self, z, adj, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.quants
        ldj -= np.log(self.quants) * z.shape[1]
        return z, ldj


class VariationalDequantization(Dequantization):
    def __init__(self, var_flows, alpha=1e-5):
        """
        Args:
            var_flows: A list of flow transformations to use for modeling q(u|x)
            alpha: Small constant, see Dequantization for details
        """
        super(VariationalDequantization, self).__init__(alpha=alpha)
        self.flows = nn.ModuleList(var_flows)

    def dequant(self, z, adj, ldj):
        z = z.to(torch.float32)

        # Prior of u is a uniform distribution as before
        # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
        deq_noise = torch.rand_like(z).detach()
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        for flow in self.flows:
            deq_noise, ldj = flow(deq_noise, adj, ldj, reverse=False, orig_img=z)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        # After the flows, apply u as in standard dequantization
        z = (z + deq_noise)
        ldj -= np.log(1) * z.shape[1]
        return z, ldj


def create_checkerboard_mask(h, invert=False):
    x = torch.arange(h, dtype=torch.int32)
    mask = torch.fmod(x, 2)
    if invert:
        mask = 1 - mask
    return mask


class CouplingLayer(nn.Module):
    def __init__(self, in_features, out_features, mask, c_in):
        """Coupling layer inside a normalizing flow.

        Args:
            network: A PyTorch nn.Module constituting the deep neural network for mu and sigma.
                      Output shape should be twice the channel size as the input.
            mask: Binary mask (0 or 1) where 0 denotes that the element should be transformed,
                   while 1 means the latent will be used as input to the NN.
            c_in: Number of input channels
        """
        super().__init__()
        self.network = SageConvLayer(in_features, out_features)
        self.scaling_factor = nn.Parameter(torch.zeros(c_in))
        # Register mask as buffer as it is a tensor which is not a parameter,
        # but should be part of the modules state.
        self.register_buffer("mask", mask)

    def forward(self, z, adj, ldj, reverse=False, orig_img=None):
        """
        Args:
            z: Latent input to the flow
            ldj: The current ldj of the previous flows.
                  The ldj of this layer will be added to this tensor.
            reverse: If True, we apply the inverse of the layer.
            orig_img (optional): Only needed in VarDeq. Allows external
                                  input to condition the flow on (e.g. original image)
        """
        # Apply network to masked input
        z_in = z * self.mask
        if orig_img is None:
            nn_out = self.network(z_in, adj)
        else:
            nn_out = self.network(torch.cat([z_in, orig_img], dim=1), adj)
        s, t = nn_out.chunk(2, dim=1)

        # Stabilize scaling output
        s_fac = self.scaling_factor.exp()
        s = torch.tanh(s / s_fac) * s_fac

        # Mask outputs (only transform the second part)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        # Affine transformation
        if not reverse:
            # Whether we first shift and then scale, or the other way round,
            # is a design choice, and usually does not have a big impact
            z = (z + t) * torch.exp(s)
            ldj += s.sum(dim=1)
        else:
            z = (z * torch.exp(-s)) - t
            ldj -= s.sum(dim=1)

        return z, ldj


class GraphFlow(nn.Module):
    def __init__(self, in_features, n_layers):
        super(GraphFlow, self).__init__()
        self.layers = nn.ModuleList()
        vardeq_layers = CouplingLayer(
                in_features=in_features,
                out_features=in_features,
                mask=create_checkerboard_mask(h=in_features, invert=False),
                c_in=1,
            )
        self.layers.append(VariationalDequantization(var_flows=[vardeq_layers]))
        for i in range(n_layers):
            self.layers.append(CouplingLayer(
                in_features=in_features,
                out_features=in_features,
                mask=create_checkerboard_mask(h=in_features, invert=(i % 2 == 1)),
                c_in=1,
            ))

    def forward(self, x, adj, ldj):
        z = x
        for i, layer in enumerate(self.layers):
            z, ldz = layer(z, adj, ldj)
        return z, ldj


class GraphFlowLight(pl.LightningModule):
    def __init__(self, flows, import_samples=8):
        """
        Args:
            flows: A list of flows (each a nn.Module) that should be applied on the images.
            import_samples: Number of importance samples to use during testing (see explanation below). Can be changed at any time
        """
        super().__init__()
        self.flow = flows
        self.import_samples = import_samples
        # Create prior distribution for final latent space
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, imgs):
        # The forward function is only used for visualizing the graph
        return self._get_likelihood(imgs)

    def encode(self, imgs, adj):
        # Given a batch of images, return the latent representation z and ldj of the transformations
        z, ldj = imgs, torch.zeros(imgs.shape[0], device=self.device)
        z, ldj = self.flow(z, adj, ldj)
        return z, ldj

    def _get_likelihood(self, x, adj, return_ll=False):
        """Given a batch of images, return the likelihood of those.

        If return_ll is True, this function returns the log likelihood of the input. Otherwise, the ouptut metric is
        bits per dimension (scaled negative log likelihood)
        """
        z, ldj = self.encode(x, adj)
        log_pz = self.prior.log_prob(z).sum(dim=1)
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / x.shape[1]
        return bpd.mean() if not return_ll else log_px

    @torch.no_grad()
    def sample(self, img_shape, z_init=None):
        """Sample a batch of images from the flow."""
        # Sample latent representation from prior
        if z_init is None:
            z = self.prior.sample(sample_shape=img_shape).to(self.device)
        else:
            z = z_init.to(self.device)

        # Transform z to x by inverting the flows
        ldj = torch.zeros(img_shape[0], device=self.device)
        for flow in reversed(self.flows):
            z, ldj = flow(z, ldj, reverse=True)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # An scheduler is optional, but can help in flows to get the last bpd improvement
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Normalizing flows are trained by maximum likelihood => return bpd
        x, adj = batch
        loss = self._get_likelihood(x, adj)
        self.log("train_bpd", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, adj = batch
        loss = self._get_likelihood(x, adj)
        self.log("val_bpd", loss)


class AffineNodeCoupling(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(AffineNodeCoupling, self).__init__()
        self.out_features = out_features
        self.out_size = out_features
        self.conv = SageConvLayer(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = activation
        self.lin1 = nn.Linear(self.out_features, self.out_features)
        self.lin2 = nn.Linear(self.out_features, 2*self.out_features)
        self.scale_factor = nn.Parameter(torch.tensor([0.], requires_grad=True))

    def forward(self, x, adj):
        s, t = self._s_t_functions(x, adj)
        h = s + t
        log_det = torch.log(torch.abs(s)).sum(-1)
        return h, log_det

    def reverse(self, y, adj):
        s, t = self._s_t_functions(y, adj)
        x = y + ((y-t)/s)
        return x, None

    def _s_t_functions(self, x, adj):
        y = self.conv(x, adj)
        y = self.norm(F.relu(y))
        y = F.tanh(self.lin1(y))
        y = self.lin2(y) * torch.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.sigmoid(s + 2)
        return s, t


def train_flow(flow, train_set, model_name="MNISTFlow"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=200,
        gradient_clip_val=1.0
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    train_data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, drop_last=False, pin_memory=True, num_workers=4
    )
    print("Start training", model_name)
    trainer.fit(flow, train_data_loader)


class GraphNvpModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_layers, activation=None):
        super(GraphNvpModel, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers
        self.activation = activation
        self.layers = nn.ModuleList()
        self.layers.append(AffineNodeCoupling(in_features=in_features, out_features=hidden_features, activation=activation))
        for i in range(n_layers-1):
            self.layers.append(AffineNodeCoupling(in_features=hidden_features, out_features=hidden_features, activation=activation))
        self.layers.append(AffineNodeCoupling(in_features=hidden_features, out_features=out_features, activation=activation))

    def forward(self, x, adj):
        h = x
        h += torch.rand((x.shape[0], x.shape[1]))
        sum_log_det_jacs_x = torch.zeros((x.shape[0]))
        for i, layer in enumerate(self.layers):
            h, log_det_jacobians = layer(h, adj)
            sum_log_det_jacs_x += log_det_jacobians
        return h, sum_log_det_jacs_x

    def reverse(self, z, adj):
        h = z
        for i, layer in enumerate(self.layers):
            h, log_det_jacobians = layer.reverse(h, adj)
        return h


class FlowLikelihood(nn.Module):
    def __init__(self):
        super(FlowLikelihood, self).__init__()
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def forward(self, z, ldj):
        log_pz = self.prior.log_prob(z).sum(dim=1)
        log_px = ldj + log_pz
        nll = -log_px
        # Calculating bits per dimension
        bpd = nll * np.log2(np.exp(1)) / z.shape[1]
        return bpd.mean()