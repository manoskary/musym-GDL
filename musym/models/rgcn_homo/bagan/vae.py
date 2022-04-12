import time
import torch.distributions.multivariate_normal as mn
from torch.utils.data import DataLoader
import argparse
from graphsaint import SAINTNodeSampler
from utils import *
from models import *
import dgl
import os


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, dest='config', help='the name of yaml file to set parameter', default='./config.yml')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model", action='store_true', default=False)
	parser.add_argument('--anomaly', dest='anomaly', help="switch for anomaly detecting", action='store_true', default=True)
	parser.add_argument('--root_dir', type=str, dest='root_dir', help='the path of current directory')
	parser.add_argument('--graph_dir', type=str, dest='graph_dir', help='the path of train data', default="./data/cad_feature_quartets/cad_feature_quartets_graph.bin")
	parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='the path of chekcpoint dir', default='./checkpoint')
	parser.add_argument('--save_dir', type=str, dest='save_dir', help='the path of generated data dir', default='sample')
	parser.add_argument('--distribution_dir', type=str, dest='distribution_dir', help='the path of class distribution dir', default='./distribution')
	parser.add_argument('--test_dir', type=str, dest='test_dir', help='the path of anomaly test data')
	parser.add_argument('--test_result_dir', type=str, dest='test_result_dir', help='the path of anomaly test result dir')


	args = parser.parse_args()
	config = Config(args.config)

	use_cuda = torch.cuda.is_available() and args.gpu >= 0
	gpu = args.gpu

	if not os.path.exists(args.save_dir):
		os.mkdir(os.path.join(args.root_dir, args.save_dir))

	g = dgl.load_graphs(args.graph_dir)[0][0]
	n_classes = max(g.ndata["label"]) + 1
	n_feats = g.ndata['feat'].shape[1]
	train_mask = g.ndata['train_mask']
	val_mask = g.ndata['val_mask']
	test_mask = g.ndata['test_mask']
	labels = g.ndata['label']

	train_nid = torch.nonzero(train_mask, as_tuple=True)[0]

	in_feats = g.ndata['feat'].shape[1]
	n_nodes = g.num_nodes()
	n_edges = g.num_edges()

	n_train_samples = train_mask.int().sum().item()
	n_val_samples = val_mask.int().sum().item()
	n_test_samples = test_mask.int().sum().item()

	kwargs = {
		'dn': "cad_quartets", 'g': g, 'train_nid': train_nid, 'num_workers_sampler': 2,
	}

	saint_sampler = SAINTNodeSampler(4000, **kwargs)
	train_loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1,
							  shuffle=True, num_workers=2, drop_last=False)

	decoder = Decoder(in_feats=config.gf_dim, n_hidden=config.df_dim, out_feats=in_feats, n_layers=1)
	encoder = Encoder(in_feats=in_feats, n_hidden=config.df_dim, out_feats=config.df_dim, n_layers=1)

	if not args.pretrained:
		if use_cuda:
			decoder = decoder.cuda(gpu)
			encoder = encoder.cuda(gpu)

		# WHY BECLoss() - only need to determine fake/real for Discriminator
		adj_loss = AdjLoss()
		if use_cuda:
			adj_loss = adj_loss.cuda(gpu)

		optimizerE = torch.optim.Adam(encoder.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
		optimizerD = torch.optim.Adam(decoder.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
	
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()

		fixed_noise = torch.FloatTensor(8 * 8, config.z_dim, 1, 1).normal_(0, 1)
		if use_cuda:
			fixed_noise = fixed_noise.cuda(gpu)
		with torch.no_grad():
			fixed_noisev = fixed_noise

		end = time.time()
		
		encoder.train()
		decoder.train()
		loss_list = []

		criterion = nn.L1Loss(size_average=False)
		# criterion = nn.MSELoss(size_average=False)
		for epoch in range(config.epoches):
			for i, subg in enumerate(train_loader):
				#l Update 'D' : max log(D(x)) + log(1-D(G(z)))
				data_time.update(time.time()-end)

				batch_size = len(subg.nodes())
				batch_inputs = subg.ndata['feat']
				batch_labels = subg.ndata['label']
				adj = subg.adj()
				if use_cuda:
					batch_inputs = batch_inputs.cuda(gpu)
					batch_labels = batch_labels.cuda(gpu)
					adj = adj.cuda(gpu)
				
				mu, log_sigmoid = encoder(adj, batch_inputs)
				# reparameterization
				std = torch.exp(log_sigmoid/2)
				eps = torch.randn_like(std)
				z = mu + eps * std
				if use_cuda:
					z = z.cuda(gpu)

				# reconstruct image
				adj_reconstruct, x_reconstruct = decoder(adj, z)

				# reconstruct_loss + KL_divergence + adj_loss
				reg_adj_loss = adj_loss(adj_reconstruct, adj)
				reconstruct_loss = criterion(x_reconstruct, batch_inputs)
				kl_div = -0.5 * torch.sum(1+log_sigmoid-mu.pow(2)-log_sigmoid.exp())
				loss = reconstruct_loss + kl_div + reg_adj_loss
				losses.update(loss.item())	
				optimizerE.zero_grad()
				optimizerD.zero_grad()
				loss.backward()
				optimizerE.step()
				optimizerD.step()

				batch_time.update(time.time()-end)
				end = time.time()
		
				# log every 100th train data of train_loader - display(100)	
				if (i+1) % config.display == 0:
					print_vae_log(epoch+1, config.epoches, i+1, len(train_loader), config.base_lr, config.display, batch_time, data_time, losses)
					# Is it Continous ???
					batch_time.reset()
					data_time.reset()
				# log every 1 epoch (all of train_loader)
				elif (i+1) == len(train_loader):
					print_vae_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr, (i + 1) % config.display, batch_time, data_time, losses)
					batch_time.reset()
					data_time.reset()

			# log every 1 epoch
			loss_list.append(losses.avg)
			losses.reset()
			save_checkpoint({'epoch': epoch, 'state_dict': encoder.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,"vae"), 'encoder_epoch_{}'.format(epoch)))
			save_checkpoint({'epoch': epoch, 'state_dict': decoder.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,"vae"), 'decoder_epoch_{}'.format(epoch)))


	## Class Conditional Generator - Pretrained Model"
	else:
		print("Class Conditional Generator - Use Pretrained Model")
		if use_cuda:
			encoder = encoder.cuda(gpu)
			decoder = decoder.cuda(gpu)
			device = "cuda:{}".format(gpu)
		else:
			device = "cpu"
		encoder.load_state_dict(torch.load(os.path.join(os.path.join(args.checkpoint_dir, "vae"), "encoder_epoch_"+ str(config.epoches-1) + ".pth.tar"), map_location=device)['state_dict'])
		decoder.load_state_dict(torch.load(os.path.join(os.path.join(args.checkpoint_dir, "vae"), "decoder_epoch_"+ str(config.epoches-1) + ".pth.tar"), map_location=device)['state_dict'])
		#Z = np.empty([config.class_num, config.z_dim], dtype=float)
		# Z : [label-1, labe-2, ... ]
		# Z[label-1] : [[z1], [z2], ... ] (#labeld_data, #z_dim)
		encoder.eval()
		decoder.eval()
		Z = []
		with torch.no_grad():
			for i in range(config.class_num):
				Z.append(torch.zeros((1, config.df_dim), dtype=torch.float)) # Z : [class_num, df_dim]

			for i, subg in enumerate(train_loader):
				batch_inputs = subg.ndata['feat']
				batch_labels = subg.ndata['label']
				adj = subg.adj()
				if use_cuda:
					batch_inputs = batch_inputs.cuda(gpu)
					adj = adj.cuda(gpu)
				mu, log_sigmoid = encoder(adj, batch_inputs)
				std = torch.exp(log_sigmoid/2)
				eps = torch.randn_like(std)
				z = mu + eps * std
				Z = batch2one(Z, batch_labels, z, config.class_num)

			N = []
			for i in range(config.class_num):
				label_mean = torch.mean(Z[i][1:], dim=0)
				label_cov = torch.from_numpy(np.cov(Z[i][1:].numpy(), rowvar=False))
				m = mn.MultivariateNormal(label_mean, label_cov)
				N.append(m)

			torch.save({'distribution': N}, os.path.join(args.distribution_dir, 'class_distribution')+'.dt')
