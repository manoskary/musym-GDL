import torch
torch.manual_seed(0)
import random
random.seed(0)
import numpy as np
np.random.seed(0)

import argparse

import dgl.dataloading

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models import SAGE
from sklearn import metrics
import os, sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import *

def get_sample_weights(labels):
	class_sample_count = torch.tensor([(labels == t).sum() for t in torch.unique(labels, sorted=True)])
	weight = 1. / class_sample_count.float()
	sample_weights = torch.tensor([weight[t] for t in labels])
	return sample_weights


def main(config):
	"""Pass parameters to create experiment"""

	# --------------- Dataset Loading -------------------------
	# dataset = dgl.data.CoraGraphDataset()
	# g = dataset[0]
	# n_classes = dataset.num_classes
	g, n_classes = load_and_save("cad_basis_homo", "./data/")

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
	# ---------------- Sampler Definition ---------------
	# Graph Sampler takes all available neighbors
	graph_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)

	# Balance Label Sampler
	label_weights = get_sample_weights(g.ndata["label"])
	# Torch Sampler
	sampler = torch.utils.data.sampler.WeightedRandomSampler(label_weights, len(label_weights), replacement=False)

	dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		graph_sampler,
		device=dataloader_device,
		batch_size=config["batch_size"],
		drop_last=False,
		num_workers=0,
		sampler=sampler
	)

	# Define model and optimizer
	model = SAGE(in_feats, 16, n_classes, 3, F.relu, 0.5)
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	# Training loop
	for epoch in tqdm(range(config["num_epochs"]), desc='epoch'):
		# Loop over the dataloader to sample the computation dependency graph as a list of blocks.
		acc = 0
		for step, (input_nodes, seeds, blocks) in enumerate(tqdm(dataloader, position=0, leave=True, desc='data')):
			# Load the input features as well as output labels
			batch_inputs = blocks[0].srcdata["feat"]
			batch_labels = blocks[-1].dstdata['label']
			if len(torch.nonzero(batch_labels))>2:
				# print(step, blocks[0].num_nodes(), batch_labels.shape)
				# Predict and loss
				batch_pred = model(blocks, batch_inputs)
				loss = criterion(batch_pred, batch_labels)
				acc = (torch.argmax(batch_pred, dim=1) == batch_labels).float().sum() / len(batch_pred)
				metrics_out = compute_metrics(torch.argmax(batch_pred, dim=1), batch_labels)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		print('Epoch {:04d} | Loss {:.4f} | Accuracy {:.4f} | Precision {:.4f} | Recall {:.4f}| F1 Score {:.4f}'.format(epoch, loss.item(), acc, metrics_out["precision"], metrics_out["recall"], metrics_out["fscore"]))
		print()

if __name__ == '__main__':
	argparser = argparse.ArgumentParser(description='Weighted Sampling SAGE')
	argparser.add_argument('--gpu', type=int, default=-1,
						   help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('-d', '--dataset', type=str, default='toy01')
	argparser.add_argument('--num-epochs', type=int, default=20)
	argparser.add_argument('--batch-size', type=int, default=1024)

	args = argparser.parse_args()
	config = vars(args)

	main(config)
