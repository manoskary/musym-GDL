import os

import dgl.data
import scipy.misc
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from musym.models.cad.train.tune_cad import n_classes
from torch import optim
from torch.autograd import Variable
from .utils import *
from .models import *
from .graphsaint import SAINTNodeSampler


if __name__ == "__main__":
	use_cuda = torch.cuda.is_available()
	gpu = 0

	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str, dest='config', help='the name of yaml file to set parameter', default='./config.yaml')
	parser.add_argument('--pretrained', dest='pretrained', help="switch for using pretrained model", action='store_true', default=False)
	parser.add_argument('--anomaly', dest='anomaly', help="switch for anomaly detecting", action='store_true', default=True)
	parser.add_argument('--root_dir', type=str, dest='root_dir', help='the path of current directory')
	parser.add_argument('--graph_dir', type=str, dest='train_dir', help='the path of train data')
	parser.add_argument('--checkpoint_dir', type=str, dest='checkpoint_dir', help='the path of chekcpoint dir', default='checkpoint')
	parser.add_argument('--save_dir', type=str, dest='save_dir', help='the path of generated data dir', default='sample')
	parser.add_argument('--distribution_dir', type=str, dest='distribution_dir', help='the path of class distribution dir', default='distribution')
	parser.add_argument('--test_dir', type=str, dest='test_dir', help='the path of anomaly test data')
	parser.add_argument('--test_result_dir', type=str, dest='test_result_dir', help='the path of anomaly test result dir')

	args = parser.parse_args()
	config = Config(args.config)

	if not os.path.exists(args.save_dir):
		os.mkdir(os.path.join(args.root_dir, args.save_dir))

	g = dgl.load_graphs(args.graph_dir)[0][0]
	n_classes = max(g.ndata["label"])+1
	n_feats = g.nada['feat'].shape[1]
	train_mask = g.ndata['train_mask']
	val_mask = g.ndata['val_mask']
	test_mask = g.ndata['test_mask']
	labels = g.ndata['label']

	train_nid = data.train_nid

	in_feats = g.ndata['feat'].shape[1]
	n_nodes = g.num_nodes()
	n_edges = g.num_edges()

	n_train_samples = train_mask.int().sum().item()
	n_val_samples = val_mask.int().sum().item()
	n_test_samples = test_mask.int().sum().item()

	kwargs = {
		'dn': args.dataset, 'g': g, 'train_nid': train_nid, 'num_workers_sampler': args.num_workers_sampler,
		'num_subg_sampler': args.num_subg_sampler, 'batch_size_sampler': args.batch_size_sampler,
		'online': args.online, 'num_subg': args.num_subg, 'full': args.full
	}

	saint_sampler = SAINTNodeSampler(4000, **kwargs)
	train_loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1,
							shuffle=True, num_workers=args.num_workers, drop_last=False)

	distribution = torch.load(os.path.join(args.distribution_dir,'class_distribution.dt'))['distribution']

	G = Generator(in_feats=n_feats, n_hidden=config.n_hidden, out_feats=config.n_hidden, n_layers=3, activation=F.relu, dropout=0.5)
	D = Discriminator(in_feats=n_feats, n_hidden=config.n_hidden, n_classes=n_classes, n_layers=3, activation=F.relu, dropout=0.5)



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
				# label = label.long().squeeze() # "squeeze" : [batch, 1] --> [batch] ... e.g) [1,2,3,4...]

				if use_cuda:
					batch_labels = batch_labels.cuda(gpu)
					batch_inputs = batch_inputs.cuda(gpu)
					blocks = [b.int().cuda(gpu) for b in blocks]
				
				sample_features, D_real = D(sub_g.adj(), batch_inputs)
				real_label.resize_(batch_size).copy_(batch_labels)	# "cpu" : gpu --> cpu // <<.data.cpu vs cpu>> // "resize_as" : get tensor size and resize
				if use_cuda:
					real_label = real_label.cuda(gpu) 
				
				D_loss_real = criterion(D_real, real_label)
				noise = conditional_z[0:fake_num].view(-1, config.z_dim)

	
				fake_label.resize_(noise.shape[0]).fill_(config.class_num)	# fake_label = '(num_class)+1'
				if use_cuda:
					noise = noise.cuda(gpu)
					fake_label = fake_label.cuda(gpu)
					z_label = z_label.cuda(gpu)
				
				fake_adg, fake = G(noise)
	
				_, D_fake = D(fake_adg, fake.detach()) # Fake image...
				D_loss_fake = criterion(D_fake, fake_label)	# Hmmmm...... fake_label? or z_label?
		
				D_loss = D_loss_real + D_loss_fake
				D_losses.update(D_loss.item())
				D.zero_grad()
				G.zero_grad()
				D_loss.backward()
				optimizerD.step()

				# Update 'G' : max log(D(G(z)))
				noise = conditional_z
				if use_cuda:
					noise = noise.cuda(gpu)
				fake_adg, fake = G(noise)
				_, D_fake = D(fake_adg, fake)
				G_loss = criterion(D_fake, z_label)
				G_losses.update(G_loss.data[0])
			
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
					#accuracy = masked_correct.item()/max(1.0, num_samples.item())
					plot_result2(fake, config.image_size, epoch + 1, args.save_dir, 'gan', is_gray=(config.c_dim == 1))
					real_acc = 100 * correct_real / total_real
					fake_acc = 100 * correct_fake / total_fake
					print('Real Accuracy : {}'.format(real_acc))
					print('Fake Accuracy : {}'.format(fake_acc))
					plot_accuracy(epoch+1, config.epoches, args.save_dir, real_acc, fake_acc)
					batch_time.reset()
					data_time.reset()

			# log every 1 epoch
			D_loss_list.append(D_losses.avg)
			G_loss_list.append(G_losses.avg)
			D_losses.reset()
			G_losses.reset()

			plot_result(G, fixed_noisev, config.image_size, epoch + 1, args.save_dir, 'gan', is_gray=(config.c_dim == 1))
			plot_loss(epoch+1, config.epoches, args.save_dir, d_loss=D_loss_list, g_loss=G_loss_list)
			# save the D and G.
			save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,'gan'), 'D_epoch_{}'.format(epoch)))
			save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(),}, os.path.join(os.path.join(args.checkpoint_dir,'gan'), 'G_epoch_{}'.format(epoch)))
	
		create_gif(config.epoches, args.save_dir, 'gan')
