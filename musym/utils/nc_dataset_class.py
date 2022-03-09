import os
import numpy as np
import torch as torch
from torch import tensor
from torch.nn.functional import normalize
import pandas as pd
import random
import dgl
from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import networkx as nx
import matplotlib.pyplot as plt

MOZART_PIANO_SONATAS = [
	'K279-1', 'K279-2', 'K279-3', 'K280-1', 'K280-2', 
	'K280-3', 'K281-1', 'K281-2', 'K281-3', 'K282-1', 
	'K282-2', 'K282-3', 'K283-1', 'K283-2', 'K283-3', 
	'K284-1', 'K284-2', 'K284-3', 'K309-1', 'K309-2', 
	'K309-3', 'K310-1', 'K310-2', 'K310-3', 'K311-1', 
	'K311-2', 'K311-3', 'K330-1', 'K330-2', 'K330-3', 
	'K331-1', 'K331-2', 'K331-3', 'K332-1', 'K332-2', 
	'K332-3', 'K333-1', 'K333-2', 'K333-3', 'K457-1', 
	'K457-2', 'K457-3', 'K533-1', 'K533-2', 'K533-3', 
	'K545-1', 'K545-2', 'K545-3', 'K570-1', 'K570-2', 
	'K570-3', 'K576-1', 'K576-2', 'K576-3'
	]


MOZART_STRING_QUARTETS = [
	"k080-02", "k156-01", "k157-01", "k158-01",
	"k159-02", "k168-02", "k171-01", "k171-04",
	"k172-02", "k173-01", "k428-02", "k499-01",
	"k589-01", "k080-01", "k155-01", "k156-02",
	"k157-02", "k159-01", "k168-01", "k169-01",
	"k171-03", "k172-01", "k172-04", "k428-01",
	"k465-04", "k499-03", "k590-01"]



BACH_FUGUES = [
	'wtc1f01', 'wtc1f07', 'wtc1f15', 'wtc1f13',
	'wtc1f06', 'wtc1f03', 'wtc1f02', 'wtc1f18',
	'wtc1f17', 'wtc1f09', 'wtc1f24', 'wtc1f10',
	'wtc1f22', 'wtc1f16', 'wtc1f12', 'wtc1f23',
	'wtc1f19', 'wtc1f05', 'wtc1f14', 'wtc1f04',
	'wtc1f08', 'wtc1f20', 'wtc1f21',
	]

HAYDN_STRING_QUARTETS = [
	'haydn_op064_no06_mv01_1770', 'haydn_op050_no06_mv01_1756',
	'haydn_op020_no06_mv02_1740', 'haydn_op020_no01_mv04_1733',
	'haydn_op076_no05_mv02_1776', 'haydn_op020_no05_mv01_1739',
	'haydn_op017_no02_mv01_1728', 'haydn_op033_no02_mv01_1743',
	'haydn_op054_no03_mv01_1761', 'haydn_op050_no01_mv01_1748',
	'haydn_op017_no06_mv01_1732', 'haydn_op064_no04_mv01_1768',
	'haydn_op064_no04_mv04_1769', 'haydn_op017_no05_mv01_1731',
	'haydn_op064_no03_mv04_1767', 'haydn_op054_no02_mv01_1760',
	'haydn_op055_no02_mv02_1764', 'haydn_op064_no03_mv01_1766',
	'haydn_op033_no03_mv03_1744', 'haydn_op074_no01_mv01_1772',
	'haydn_op054_no01_mv01_1758', 'haydn_op076_no02_mv01_1774',
	'haydn_op033_no05_mv02_1747', 'haydn_op055_no01_mv02_1763',
	'haydn_op054_no01_mv02_1759', 'haydn_op050_no02_mv01_1750',
	'haydn_op050_no03_mv04_1752', 'haydn_op020_no04_mv04_1738',
	'haydn_op033_no01_mv03_1742', 'haydn_op033_no05_mv01_1746',
	'haydn_op050_no06_mv02_1757', 'haydn_op020_no03_mv04_1736',
	'haydn_op076_no04_mv01_1775', 'haydn_op050_no05_mv04_1755',
	'haydn_op033_no01_mv01_1741', 'haydn_op054_no03_mv04_1762',
	'haydn_op050_no04_mv01_1753', 'haydn_op050_no02_mv04_1751',
	'haydn_op017_no01_mv01_1727', 'haydn_op033_no04_mv01_1745',
	'haydn_op017_no03_mv04_1729', 'haydn_op050_no01_mv04_1749',
	'haydn_op055_no03_mv01_1765', 'haydn_op074_no01_mv02_1773',
	'haydn_op020_no03_mv03_1735'
]


PIANO = BACH_FUGUES + MOZART_PIANO_SONATAS

QUARTETS = MOZART_STRING_QUARTETS + HAYDN_STRING_QUARTETS

MOZART = MOZART_STRING_QUARTETS + MOZART_PIANO_SONATAS

MIX = PIANO + QUARTETS

FILE_LIST = [
	'note-during-note.csv', 'note-follows-note.csv', 
	'note-follows-rest.csv', 'note-onset-note.csv', 
	'note.csv', 'rest-follows-note.csv', 'rest.csv' 
	]

BASIS_FN = [
	'onset_feature.score_position', 'duration_feature.duration', 'fermata_feature.fermata',
	'grace_feature.n_grace', 'grace_feature.grace_pos', 'onset_feature.onset',
	'polynomial_pitch_feature.pitch', 'grace_feature.grace_note',
	'relative_score_position_feature.score_position', 'slur_feature.slur_incr',
	'slur_feature.slur_decr', 'time_signature_feature.time_signature_num_1',
	'time_signature_feature.time_signature_num_2', 'time_signature_feature.time_signature_num_3',
	'time_signature_feature.time_signature_num_4', 'time_signature_feature.time_signature_num_5',
	'time_signature_feature.time_signature_num_6', 'time_signature_feature.time_signature_num_7',
	'time_signature_feature.time_signature_num_8', 'time_signature_feature.time_signature_num_9',
	'time_signature_feature.time_signature_num_10', 'time_signature_feature.time_signature_num_11',
	'time_signature_feature.time_signature_num_12', 'time_signature_feature.time_signature_num_other',
	'time_signature_feature.time_signature_den_1', 'time_signature_feature.time_signature_den_2',
	'time_signature_feature.time_signature_den_4', 'time_signature_feature.time_signature_den_8',
	'time_signature_feature.time_signature_den_16', 'time_signature_feature.time_signature_den_other',
	'vertical_neighbor_feature.n_total', 'vertical_neighbor_feature.n_above', 'vertical_neighbor_feature.n_below',
	'vertical_neighbor_feature.highest_pitch', 'vertical_neighbor_feature.lowest_pitch',
	'vertical_neighbor_feature.pitch_range'
	]

def min_max_scaler(X):
	data_min = np.nanmin(X, axis=0)
	data_max = np.nanmax(X, axis=0)
	data_range = data_max - data_min
	X_std = (X - data_min) / (data_range + 1e-8)
	return X_std


class CadHomoGraphDataset(DGLDataset):
	def __init__(self, name, url, add_inverse_edges=False, add_aug=True, select_piece=None, features=None, normalize=False, save_path=None, piece_list=None):
		if features:
			self.features = features
		elif "feature" in url:
			self.features = ["onset", "duration", "ts"] + BASIS_FN + ["pitch"]
		else :
			self.features = ["onset", "duration", "ts", "pitch"]
		self.normalize = normalize
		self.add_inverse_edges = add_inverse_edges
		self.add_aug = add_aug
		if piece_list:
			self.piece_list = piece_list
		self.select_piece = select_piece
		super().__init__(name=name, url=url, save_dir=save_path)

	def process(self):
		if not hasattr(self, 'piece_list'):
			self.piece_list = MOZART_PIANO_SONATAS

		piece_list_retracing = ["augmented_piece"] + self.piece_list
		self.piece_encoding = dict(zip(range(len(piece_list_retracing)), piece_list_retracing))
		self.inverse_piece_encoding = dict(zip(piece_list_retracing, range(len(piece_list_retracing))))
		self.test_piece_list = random.sample(self.piece_list, int(0.2*len(self.piece_list)))
		self.val_piece_list = random.sample(list(set(self.piece_list)-set(self.test_piece_list)), int(0.2*len(self.piece_list))) 
		self.train_piece_list = list(set(self.piece_list)-(set(self.test_piece_list).union(set(self.val_piece_list))))
		self.FILE_LIST = ["nodes.csv", "edges.csv"]
		if self.select_piece and self.select_piece in self.piece_list:          
			self._process_select_piece()
			n_nodes = self.graph.num_nodes()
			n_train = int(n_nodes * 0.6)
			n_val = n_train + int(n_nodes*0.2)
		elif self.name == "toy_homo_onset":
			self._process_toy()
			n_nodes = self.graph.num_nodes()
			n_train = int(n_nodes * 0.6)
			n_val = n_train + int(n_nodes*0.2)
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
				graph.ndata['feat'] = note_node_features.float()
				graph.ndata['label'] = note_node_labels
				graph.ndata['score_name'] = torch.tensor([self.inverse_piece_encoding[fn]]).repeat(len(note_node_labels))
				self.graph = dgl.batch([g, graph])
			except AttributeError:
				self.graph = dgl.graph(edges)
				self.graph.ndata['feat'] = note_node_features.float()
				self.graph.ndata['label'] = note_node_labels
				self.graph.ndata['score_name'] = torch.tensor([self.inverse_piece_encoding[fn]]).repeat(len(note_node_labels))


			# Perform Data Augmentation
			if self.add_aug:
				for _ in range(5):
					g = self.graph
					resize_factor = random.choice([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5])
					n = random.choice(range(-6, 6, 1))
					if resize_factor != 1 or n != 0:
						if "pitch" in self.features:
							num_feat = len(self.features)
							dur_resize = torch.tensor([resize_factor for _ in range(3)] + [1 for _ in range(num_feat-3)]).float()
							pitch_aug = torch.tensor([0 for _ in  range(num_feat-1)] + [n]).float()
							graph = dgl.graph(edges)
							graph.ndata['feat'] = note_node_features.float()*dur_resize + pitch_aug
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
				piece_list = MIX)

class cad_feature_hsq(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-hsq/"
		super().__init__(
				name='cad_feature_hsq', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = HAYDN_STRING_QUARTETS)

class cad_feature_wtc(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-wtc/"
		super().__init__(
				name='cad_feature_wtc', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = BACH_FUGUES)

class cad_feature_msq(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-msq/"
		super().__init__(
				name='cad_feature_msq', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = MOZART_STRING_QUARTETS)

class cad_feature_quartets(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-quartets/"
		super().__init__(
				name='cad_feature_quartets', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = QUARTETS)

class cad_feature_piano(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-piano/"
		super().__init__(
				name='cad_feature_piano', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = PIANO)

class cad_feature_mozart(CadHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/cad-feature-mozart/"
		super().__init__(
				name='cad_feature_mozart', url=url,
				add_inverse_edges=add_inverse_edges, add_aug=add_aug,
				select_piece=select_piece, normalize=False,
				features=None, save_path=save_path,
				piece_list = MOZART)


if __name__ == "__main__":
	import dgl
	

	dirname = os.path.dirname(__file__)
	

	dirname = os.path.dirname(__file__)
	par = lambda x: os.path.abspath(os.path.join(x, os.pardir))
	par_dir = par(par(dirname))    
	# raw_dir = os.path.join(par_dir, "artifacts", "data", "cadences", "graph_t345.pkl")
	# name = "dummyExample"
	# dataset = MyGraphDataset(name, raw_dir=raw_dir)
	# print(dataset.graphs, dataset.labels)
	# dataset.save()
	raw_dir = os.path.join(par_dir, "artifacts", "data", "cadences", "mozart_piano_sonatas")
	# name = "dummyExample"
	dataset = MozartPianoGraphDataset()
	print(dataset.num_classes, dataset[0])
