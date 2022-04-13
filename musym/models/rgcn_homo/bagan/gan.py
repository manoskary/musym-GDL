import dgl.data
import time
import math
from torch.utils.data import DataLoader
import argparse
from utils import *
from models import *
from graphsaint import SAINTNodeSampler


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, dest='config', help='the name of yaml file to set parameter', default='./config.yml')
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model", action='store_true', default=False)
	parser.add_argument('--anomaly', dest='anomaly', help="switch for anomaly detecting", action='store_true', default=True)
	parser.add_argument('--root_dir', type=str, dest='root_dir', help='the path of current directory')
	parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='the path of chekcpoint dir', default='checkpoint')
	parser.add_argument('--graph_dir', type=str, dest='graph_dir', help='the path of train data', default="./data/cad_pac_wtc/cad_pac_wtc_graph.bin")
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
		'dn': "cad_pac_wtc", 'g': g, 'train_nid': train_nid, 'num_workers_sampler': 2,
	}

	saint_sampler = SAINTNodeSampler(4000, **kwargs)
	train_loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1,
							  shuffle=True, num_workers=2, drop_last=False)

	distribution = torch.load(os.path.join(args.distribution_dir,'class_distribution.dt'))['distribution']

	G = Generator(in_feats=config.df_dim, n_hidden=config.df_dim, out_samples=in_feats, n_layers=1)
	D = Discriminator(in_feats=in_feats, n_hidden=config.df_dim, n_classes=n_classes, n_layers=1)


	if not args.pretrained:
		if use_cuda:
			G = G.cuda(gpu)
			D = D.cuda(gpu)

		criterion = nn.NLLLoss()

		optimizerD = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
		optimizerG = torch.optim.Adam(G.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))

		batch_time = AverageMeter()
		data_time = AverageMeter()
		D_losses = AverageMeter()
		G_losses = AverageMeter()

		end = time.time()
		
		D.train()
		G.train()
		D_loss_list = []
		G_loss_list = []
	
		real_label = torch.LongTensor(config.batch_size)
		fake_label = torch.LongTensor(config.batch_size)	

		for epoch in range(config.epoches):
			total_real = 0
			total_fake = 0
			correct_real = 0
			correct_fake = 0
			for i, sub_g in enumerate(train_loader):
				# Update 'D' : max log(D(x)) + log(1-D(G(z)))
				data_time.update(time.time()-end)
				batch_size = len(sub_g.nodes())
				fake_num = math.ceil(batch_size/config.class_num)	# For each batch, 1/(n+1) of total images are fake
				conditional_z, z_label = conditional_latent_generator(distribution, config.class_num, batch_size)

				batch_inputs = sub_g.ndata['feat']
				batch_labels = sub_g.ndata['label']
				batch_adj = sub_g.adj()
				# label = label.long().squeeze() # "squeeze" : [batch, 1] --> [batch] ... e.g) [1,2,3,4...]

				if use_cuda:
					batch_labels = batch_labels.cuda(gpu)
					batch_inputs = batch_inputs.cuda(gpu)
					batch_adj = batch_adj.cuda(gpu)
				
				sample_features, D_real = D(batch_adj, batch_inputs)
				real_label.resize_(batch_size).copy_(batch_labels)	# "cpu" : gpu --> cpu // <<.data.cpu vs cpu>> // "resize_as" : get tensor size and resize
				if use_cuda:
					real_label = real_label.cuda(gpu) 
				
				D_loss_real = criterion(D_real, real_label)
				noise = conditional_z[0:fake_num].float()

	
				fake_label.resize_(noise.shape[0]).fill_(config.class_num)	# fake_label = '(num_class)+1'
				if use_cuda:
					noise = noise.cuda(gpu)
					fake_label = fake_label.cuda(gpu)
					z_label = z_label.cuda(gpu)
				
				fake_adj, fake = G(noise)
	
				_, D_fake = D(fake_adj, fake.detach()) # Fake image...
				D_loss_fake = criterion(D_fake, fake_label)	# Hmmmm...... fake_label? or z_label?
		
				D_loss = D_loss_real + D_loss_fake
				D_losses.update(D_loss.item())
				D.zero_grad()
				G.zero_grad()
				D_loss.backward()
				optimizerD.step()

				# Update 'G' : max log(D(G(z)))
				noise = conditional_z.float()
				if use_cuda:
					noise = noise.cuda(gpu)
				fake_adj, fake = G(noise)
				_, D_fake = D(fake_adj, fake)
				G_loss = criterion(D_fake, z_label)
				G_losses.update(G_loss.item())
			
				D.zero_grad()
				G.zero_grad()
				G_loss.backward()
				optimizerG.step()

				batch_time.update(time.time()-end)
				end = time.time()
				
				pred_real = torch.max(D_real.data, 1)[1]
				pred_fake = torch.max(D_fake.data, 1)[1]
				total_real += real_label.size(0)
				total_fake += z_label.size(0)
				correct_real += (pred_real == real_label).sum().item()
				correct_fake += (pred_fake == z_label).sum().item()
	
				# log every 100th train data of train_loader - display(100)	
				if (i+1) % config.display == 0:
					print_gan_log(epoch+1, config.epoches, i+1, len(train_loader), config.base_lr, config.display, batch_time, data_time, D_losses, G_losses)
					# Is it Continous ???
					batch_time.reset()
					data_time.reset()
				# log every 1 epoch (all of train_loader) ... "End of all mini-Batch"
				elif (i+1) == len(train_loader):
					print_gan_log(epoch + 1, config.epoches, i + 1, len(train_loader), config.base_lr,
	                          (i + 1) % config.display, batch_time, data_time, D_losses, G_losses)
					real_acc = 100 * correct_real / total_real
					fake_acc = 100 * correct_fake / total_fake
					print('Real Accuracy : {}'.format(real_acc))
					print('Fake Accuracy : {}'.format(fake_acc))
					batch_time.reset()
					data_time.reset()

			# log every 1 epoch
			D_loss_list.append(D_losses.avg)
			G_loss_list.append(G_losses.avg)
			D_losses.reset()
			G_losses.reset()
			# save the D and G.
			save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,'gan'), 'D_epoch_{}'.format(epoch)))
			save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,'gan'), 'G_epoch_{}'.format(epoch)))

	else:
		print("Class Conditional Generator - Use Pretrained Model")
		if use_cuda:
			D = D.cuda(gpu)
			G = G.cuda(gpu)
			device = "cuda:{}".format(gpu)
		else:
			device = "cpu"
		D.load_state_dict(torch.load(os.path.join(os.path.join(args.checkpoint_dir, "gan"),
														"D_epoch_" + str(config.epoches - 1) + ".pth.tar"),
										   map_location=device)['state_dict'])
		G.load_state_dict(torch.load(os.path.join(os.path.join(args.checkpoint_dir, "gan"),
														"G_epoch_" + str(config.epoches - 1) + ".pth.tar"),
										   map_location=device)['state_dict'])
		# Z = np.empty([config.class_num, config.z_dim], dtype=float)
		# Z : [label-1, labe-2, ... ]
		# Z[label-1] : [[z1], [z2], ... ] (#labeld_data, #z_dim)
		D.eval()
		G.eval()
		predictions = list()
		labels = list()
		with torch.no_grad():
			for i, subg in enumerate(train_loader):
				batch_inputs = subg.ndata['feat']
				batch_labels = subg.ndata['label']
				adj = subg.adj()
				if use_cuda:
					batch_inputs = batch_inputs.cuda(gpu)
					adj = adj.cuda(gpu)
				_, batch_pred = D(adj, batch_inputs)
				predictions.append(torch.argmax(batch_pred, dim=1))
				labels.append(batch_labels)
			preds = torch.cat(predictions)
			labels = torch.cat(labels)
			acc = (preds == labels).float().mean().item()
			from sklearn.metrics import f1_score
			fscore = f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
			print("Accuracy: {:.4f} | Fscore: {:.4f} | Num Classes : {}".format(acc, fscore, preds.max()+1))
