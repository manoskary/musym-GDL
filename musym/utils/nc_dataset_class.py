import os
import numpy as np
import torch as torch
import pandas as pd
import random
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info
import networkx as nx
import matplotlib.pyplot as plt
from musym.utils.metadata import *
from musym.utils.pos_enc import positional_encoding


def min_max_scaler(X):
	data_min = np.nanmin(X, axis=0)
	data_max = np.nanmax(X, axis=0)
	data_range = data_max - data_min
	X_std = (X - data_min) / (data_range + 1e-8)
	return X_std


class CadHomoGraphDataset(DGLDataset):
	def __init__(self, name, url, add_inverse_edges=False, add_aug=True, select_piece=None, features=None, normalize=False, save_path=None, piece_list=None, pos_enc_dim=0):
		if features:
			self.features = features
		else:
			self.features = ["onset", "duration", "ts"] + BASIS_FN + CAD_FEATURES + ["voice"]
		self.normalize = normalize
		self.add_inverse_edges = add_inverse_edges
		self.add_aug = add_aug
		if piece_list:
			self.piece_list = piece_list
		self.select_piece = select_piece
		self.pos_enc_dim = pos_enc_dim
		super().__init__(name=name, url=url, save_dir=save_path)

	def process(self):
		if not hasattr(self, 'piece_list'):
			self.piece_list = MOZART_PIANO_SONATAS

		piece_list_retracing = ["augmented_piece"] + self.piece_list
		self.piece_encoding = dict(zip(range(len(piece_list_retracing)), piece_list_retracing))
		self.inverse_piece_encoding = dict(zip(piece_list_retracing, range(len(piece_list_retracing))))
		self.test_piece_list = random.sample(self.piece_list, int(0.5*len(self.piece_list)))
		self.val_piece_list = random.sample(list(set(self.piece_list)-set(self.test_piece_list)), int(0.1*len(self.piece_list)))

		self.train_piece_list = list(set(self.piece_list)-(set(self.test_piece_list).union(set(self.val_piece_list))))

		self.FILE_LIST = ["nodes.csv", "edges.csv"]
		if self.select_piece and self.select_piece in self.piece_list:          
			self._process_select_piece()
			n_nodes = self.graph.num_nodes()
			n_train = int(n_nodes * 0.8)
			n_val = n_train + int(n_nodes*0.1)
		elif self.name == "toy_homo_onset":
			self._process_toy()
			n_nodes = self.graph.num_nodes()
			n_train = int(n_nodes * 0.8)
			n_val = n_train + int(n_nodes*0.1)
		else:             
			print("------- Loading Train Pieces -------")
			self._process_loop(self.train_piece_list)
			n_train = self.graph.num_nodes()
			self.add_aug = False
			print("------- Loading Validation Pieces -------")
			self._process_loop(self.val_piece_list)
			n_val = self.graph.num_nodes()
			print("------- Loading Test Pieces -------")
			self._process_loop(self.test_piece_list)

		# If your dataset is a node classification dataset, you will need to assign
		# masks indicating whether a node belongs to training, validation, and test set.
		self.num_classes = int(self.graph.ndata['label'].max().item() + 1)
		self.predict_category = "note"
		n_nodes = self.graph.num_nodes()
		if self.normalize:
			self.graph.ndata["feat"] = min_max_scaler(self.graph.ndata["feat"])
		train_mask = torch.zeros(n_nodes, dtype=torch.bool)
		val_mask = torch.zeros(n_nodes, dtype=torch.bool)
		test_mask = torch.zeros(n_nodes, dtype=torch.bool)
		train_mask[:n_train] = True
		val_mask[n_train:n_val] = True
		test_mask[n_val :] = True
		self.graph.ndata['train_mask'] = train_mask
		self.graph.ndata['val_mask'] = val_mask
		self.graph.ndata['test_mask'] = test_mask

		self.print_dataset_info()


	def _process_loop(self, piece_list):            
		for fn in piece_list:
			print(fn)
			edge_dict = dict()
			for csv in self.FILE_LIST:
				path = self.url + "/" + fn + "/" + csv
				if csv == "nodes.csv":
					notes = pd.read_csv(path)
					a = notes[self.features].to_numpy()
					note_node_features = torch.from_numpy(a)
					note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
				else :
					edges_data = pd.read_csv(path)
					if edges_data.empty:   
						edges_src = torch.tensor([0])
						edges_dst = torch.tensor([0])                       
					else :
						edges_src = torch.from_numpy(edges_data['src'].to_numpy())
						edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
					if self.add_inverse_edges:
						src = torch.cat((edges_src, edges_dst))
						dst = torch.cat((edges_dst, edges_src))
						edges_src = src
						edges_dst = dst
					edges = (edges_src, edges_dst)

			# Have to store onset local and onset global.
			try:
				g = self.graph
				graph = dgl.graph(edges)
				if self.pos_enc_dim > 0:
					pos_enc = positional_encoding(graph, self.pos_enc_dim)
					graph.ndata['feat'] = torch.cat((note_node_features.float(), pos_enc), dim=1)
				else:
					graph.ndata['feat'] = note_node_features.float()
				graph.ndata['label'] = note_node_labels
				graph.ndata['score_name'] = torch.tensor([self.inverse_piece_encoding[fn]]).repeat(len(note_node_labels))
				self.graph = dgl.batch([g, graph])
			except AttributeError:
				self.graph = dgl.graph(edges)
				if self.pos_enc_dim > 0:
					pos_enc = positional_encoding(self.graph, self.pos_enc_dim)
					self.graph.ndata['feat'] = torch.cat((note_node_features.float(), pos_enc), dim=1)
				else:
					pos_enc = torch.tensor([])
					self.graph.ndata['feat'] = note_node_features.float()
				self.graph.ndata['label'] = note_node_labels
				self.graph.ndata['score_name'] = torch.tensor([self.inverse_piece_encoding[fn]]).repeat(len(note_node_labels))


			# Perform Data Augmentation
			if self.add_aug:
				for _ in range(10):
					g = self.graph
					resize_factor = random.choice([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5])
					n = random.choice(range(-6, 6, 1))
					if resize_factor != 1 or n != 0:
						if "pitch" in self.features:
							num_feat = len(self.features)
							dur_resize = torch.tensor([resize_factor for _ in range(3)] + [1 for _ in range(num_feat-3)]).float()
							pitch_aug = torch.tensor([0 for _ in  range(num_feat-1)] + [n]).float()
							graph = dgl.graph(edges)
							graph.ndata['feat'] = torch.cat((note_node_features.float()*dur_resize + pitch_aug, pos_enc), dim=1)
							graph.ndata['label'] = note_node_labels
							graph.ndata['score_name'] = torch.tensor([0]).repeat(len(note_node_labels))
							self.graph = dgl.batch([g, graph])
						else:                           
							num_feat = len(self.features)
							dur_resize = torch.tensor([resize_factor for _ in range(num_feat)]).float()
							graph = dgl.graph(edges)
							graph.ndata['feat'] = note_node_features.float()*dur_resize
							graph.ndata['label'] = note_node_labels
							# Augmented Scores have a 0 score name to filter them out.
							graph.ndata['score_name'] = torch.tensor([0]).repeat(len(note_node_labels))
							self.graph = dgl.batch([g, graph])


	def _process_select_piece(self):
		for csv in self.FILE_LIST:
			path = self.url + "/" + self.select_piece + "/" + csv
			if csv == "nodes.csv":
				notes = pd.read_csv(path)
				note_node_features = torch.from_numpy(notes[self.features].to_numpy())
				note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
			else :
				edges_data = pd.read_csv(path)
				if edges_data.empty:   
					edges_src = torch.tensor([0])
					edges_dst = torch.tensor([0])                       
				else :
					edges_src = torch.from_numpy(edges_data['src'].to_numpy())
					edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
				if self.add_inverse_edges:
					src = torch.cat((edges_src, edges_dst))
					dst = torch.cat((edges_dst, edges_src))
					edges_src = src
					edges_dst = dst
				edges = (edges_src, edges_dst) 
		self.graph = dgl.graph(edges)
		self.graph.ndata['feat'] = note_node_features.float()
		self.graph.ndata['label'] = note_node_labels

	def _process_toy(self):	
		for csv in self.FILE_LIST:
			path = self.url + "/toy/" + csv
			if csv == "nodes.csv":
				notes = pd.read_csv(path)
				a = notes[self.features].to_numpy()
				if self.normalize:
					a = min_max_scaler(a)
				note_node_features = torch.from_numpy(a)
				note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
			else :
				edges_data = pd.read_csv(path)
				if edges_data.empty:   
					edges_src = torch.tensor([0])
					edges_dst = torch.tensor([0])                       
				else :
					edges_src = torch.from_numpy(edges_data['src'].to_numpy())
					edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
				if self.add_inverse_edges:
					src = torch.cat((edges_src, edges_dst))
					dst = torch.cat((edges_dst, edges_src))
					edges_src = src
					edges_dst = dst
				edges = (edges_src, edges_dst)
		self.graph = dgl.graph(edges)
		self.graph.ndata['feat'] = note_node_features.float()
		self.graph.ndata['label'] = note_node_labels
		

	def __getitem__(self, i):
		return self.graph

	def __len__(self):
		return 1

	def save_data(self):
		# save graphs and labels
		graph_path = os.path.join(self.save_path, self.name + '_graph.bin')
		save_graphs(graph_path, self.graph)
		# save other information in python dict
		info_path = os.path.join(self.save_path, self.name + '_info.pkl')
		save_info(info_path, {
			'num_classes': self.num_classes, "predict_category" : self.predict_category, 
			"features" : self.features, "PIECE_LIST" : self.piece_list, 
			"test_piece_list" : self.test_piece_list, "train_piece_list" : self.train_piece_list, "piece_encoding" : self.piece_encoding})

	def load_data(self):
		# load processed data from directory `self.save_path`
		graph_path = os.path.join(self.save_path, self.name + '_graph.bin')
		self.graph = load_graphs(graph_path)
		info_path = os.path.join(self.save_path, self.name + '_info.pkl')
		self.num_classes = load_info(info_path)['num_classes']
		self.features = load_info(info_path)['features']
		self.piece_list = load_info(info_path)['PIECE_LIST']
		self.test_piece_list = load_info(info_path)['test_piece_list']
		self.train_piece_list = load_info(info_path)['train_piece_list']

	def has_cache(self):
		# check whether there are processed data in `self.save_path`
		graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
		info_path = os.path.join(self.save_path, self.name + '_info.pkl')
		return os.path.exists(graph_path) and os.path.exists(info_path)

	def print_dataset_info(self):
		print("NumNodes: ", self.graph.num_nodes())
		print("NumEdges: ", self.graph.num_edges())
		print("NumFeats: ", self.graph.ndata["feat"].shape[1])
		print("NumClasses: ", self.num_classes)
		print("NumTrainingSamples: ", torch.count_nonzero(self.graph.ndata["train_mask"]).item())
		print("NumValidationSamples: ", torch.count_nonzero(self.graph.ndata["val_mask"]).item())
		print("NumTestSamples: ", torch.count_nonzero(self.graph.ndata["test_mask"]).item())

	def visualize_graph(self, save_dir, name=None):
		G = dgl.node_subgraph(self.graph, list(range(10, 50)))
		nx_G = G.to_networkx().to_undirected()
		# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
		pos = nx.kamada_kawai_layout(nx_G)
		nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
		plt.show(block=False)
		if not os.path.exists(save_dir):
			os.path.makedirs(save_dir)
		if name:
			plt.savefig(os.path.join(save_dir, name+".png"), format="PNG")
		else:
			plt.savefig(os.path.join(save_dir, "graph.png"), format="PNG")





class cad_feature_homo(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-homo/"
		super().__init__(
				name='cad_feature_homo', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = MIX, pos_enc_dim=20)

class cad_feature_hsq(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-hsq/"
		super().__init__(
				name='cad_feature_hsq', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = HAYDN_STRING_QUARTETS, pos_enc_dim=20)


class cad_pac_hsq(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-pac-hsq/"
		super().__init__(
				name='cad_pac_hsq', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = HAYDN_STRING_QUARTETS, pos_enc_dim=20)


class cad_feature_wtc(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-wtc/"
		super().__init__(
				name='cad_feature_wtc', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = BACH_FUGUES_CAD, pos_enc_dim=20)


class cad_pac_wtc(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-pac-wtc/"
		super().__init__(
				name='cad_pac_wtc', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = BACH_FUGUES_PAC, pos_enc_dim=20)


class cad_riac_wtc(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-riac-wtc/"
		super().__init__(
				name='cad_riac_wtc', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = BACH_FUGUES, pos_enc_dim=20)


class cad_hc_hsq(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-hc-hsq/"
		super().__init__(
				name='cad_hc_hsq', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = HAYDN_STRING_QUARTETS, pos_enc_dim=20)


class cad_feature_msq(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-msq/"
		super().__init__(
				name='cad_feature_msq', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = MOZART_STRING_QUARTETS, pos_enc_dim=20)

class cad_feature_quartets(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-quartets/"
		super().__init__(
				name='cad_feature_quartets', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list=QUARTETS, pos_enc_dim=20)


class cad_feature_piano(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-piano/"
		super().__init__(
				name='cad_feature_piano', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = PIANO, pos_enc_dim=20)


class cad_feature_mozart(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-mozart/"
		super().__init__(
				name='cad_feature_mozart', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = MOZART, pos_enc_dim=20)


class cad_feature_mix(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=False, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-mix/"
		super().__init__(
				name='cad_feature_mix', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = MIX, pos_enc_dim=20)






