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


PIECE_LIST = [
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

FILE_LIST = [
	'note-during-note.csv', 'note-follows-note.csv', 
	'note-follows-rest.csv', 'note-onset-note.csv', 
	'note.csv', 'rest-follows-note.csv', 'rest.csv' 
	]

def min_max_scaler(X):
	data_min = np.nanmin(X, axis=0)
	data_max = np.nanmax(X, axis=0)
	data_range = data_max - data_min
	X_std = (X - data_min) / data_range
	return X_std


class MozartPianoHomoGraphDataset(DGLDataset):
	def __init__(self, name, url, add_inverse_edges=False, add_aug=True, select_piece=None, features=None, normalize=False, save_path=None):
		if features:
			self.features = features
		else :
			self.features = ["onset", "duration", "ts", "pitch"]
		self.normalize = normalize
		self.add_inverse_edges = add_inverse_edges
		self.add_aug = add_aug
		self.select_piece = select_piece
		super().__init__(name=name, url=url, save_dir=save_path)

	def process(self):
		self.PIECE_LIST = PIECE_LIST
		self.test_piece_list = random.sample(self.PIECE_LIST, 16) 
		self.train_piece_list = list(set(self.PIECE_LIST)-set(self.test_piece_list))
		self.FILE_LIST = ["nodes.csv", "edges.csv"]
		if self.select_piece and self.select_piece in self.PIECE_LIST:          
			self._process_select_piece()
			n_nodes = self.graph.num_nodes()
			n_train = int(n_nodes * 0.8)
		elif "toy" in self.name:
			self._process_toy()
			n_nodes = self.graph.num_nodes()
			n_train = int(n_nodes * 0.8)
		else:             
			print("------- Loading Train Pieces -------")
			self._process_loop(self.train_piece_list)
			n_train = self.graph.num_nodes()
			self.add_aug = False
			print("------- Loading Test Pieces -------")
			self._process_loop(self.test_piece_list)

		# If your dataset is a node classification dataset, you will need to assign
		# masks indicating whether a node belongs to training, validation, and test set.
		self.num_classes = int(self.graph.ndata['label'].max().item() + 1)
		self.predict_category = "note"
		n_nodes = self.graph.num_nodes()
		

		train_mask = torch.zeros(n_nodes, dtype=torch.bool)
		test_mask = torch.zeros(n_nodes, dtype=torch.bool)
		train_mask[:n_train] = True
		test_mask[n_train :] = True
		self.graph.ndata['train_mask'] = train_mask
		self.graph.ndata['val_mask'] = train_mask
		self.graph.ndata['test_mask'] = test_mask

	def _process_loop(self, piece_list):            
		for fn in piece_list:
			print(fn)
			edge_dict = dict()
			for csv in self.FILE_LIST:
				path = self.url + "/" + fn + "/" + csv
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

			try:
				g = self.graph
				graph = dgl.graph(edges)
				graph.ndata['feat'] = note_node_features.float()
				graph.ndata['label'] = note_node_labels                    
				self.graph = dgl.batch([g, graph])
			except AttributeError:
				self.graph = dgl.graph(edges)
				self.graph.ndata['feat'] = note_node_features.float()
				self.graph.ndata['label'] = note_node_labels
				


			# Perform Data Augmentation
			if self.add_aug:
				for _ in range(5):
					g = self.graph
					resize_factor = random.choice([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5])
					n = random.choice(range(-6, 6, 1))
					if resize_factor != 1 or n != 0:
						if "pitch" in self.features:
							num_feat = len(self.features)
							dur_resize = torch.tensor([resize_factor for _ in range(num_feat-1) + [1]]).float()
							pitch_aug = torch.tensor([0 for _ in  range(num_feat-1)] + [n]).float()
							graph = dgl.graph(edges)
							graph.ndata['feat'] = note_node_features.float()*dur_resize + pitch_aug
							graph.ndata['label'] = note_node_labels
							self.graph = dgl.batch([g, graph])
						else:                           
							num_feat = len(self.features)
							dur_resize = torch.tensor([resize_factor for _ in range(num_feat)]).float()
							graph = dgl.graph(edges)
							graph.ndata['feat'] = note_node_features.float()*dur_resize
							graph.ndata['label'] = note_node_labels
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
		# for ind in range(200):
		for csv in self.FILE_LIST:
			# path = os.path.join(self.url, "toy_"+str(ind), csv)
			path = os.path.join(self.url, "toy", csv)
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
		try:
			g = self.graph
			graph = dgl.graph(edges)
			graph.ndata['feat'] = note_node_features.float()
			graph.ndata['label'] = note_node_labels                    
			self.graph = dgl.batch([g, graph])
		except AttributeError:
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
			"features" : self.features, "PIECE_LIST" : self.PIECE_LIST, 
			"test_piece_list" : self.test_piece_list, "train_piece_list" : self.train_piece_list})

	def load_data(self):
		# load processed data from directory `self.save_path`
		graph_path = os.path.join(self.save_path, self.name + '_graph.bin')
		self.graph = load_graphs(graph_path)
		info_path = os.path.join(self.save_path, self.name + '_info.pkl')
		self.num_classes = load_info(info_path)['num_classes']
		self.features = load_info(info_path)['features']
		self.PIECE_LIST = load_info(info_path)['PIECE_LIST']
		self.test_piece_list = load_info(info_path)['test_piece_list']
		self.train_piece_list = load_info(info_path)['train_piece_list']

	def has_cache(self):
		# check whether there are processed data in `self.save_path`
		graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
		info_path = os.path.join(self.save_path, self.name + '_info.pkl')
		return os.path.exists(graph_path) and os.path.exists(info_path)



class MozartPianoGraphDataset(DGLDataset):
	def __init__(self, name, url, raw_dir=None, add_inverse_edges=False, add_aug=True, select_piece=None, features=None, normalize=False):
		if features:
			self.note_features = features
			self.rest_features = features.remove("pitch") if "pitch" in features else features 
		else :
			self.note_features = ["onset", "duration", "ts", "pitch"]
			self.rest_features = ["onset", "duration", "ts"]
		self.normalize = normalize
		self.add_inverse_edges = add_inverse_edges
		self.add_aug = add_aug
		self.select_piece = select_piece
		# url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mozart_piano_sonatas/"
		super().__init__(name=name, raw_dir=raw_dir, url=url)

	def process(self):
		self.PIECE_LIST = PIECE_LIST
		self.FILE_LIST = FILE_LIST
		if self.select_piece and self.select_piece in self.PIECE_LIST:
			edge_dict = dict()
			for csv in self.FILE_LIST:
				path = self.url + "/" + self.select_piece + "/" + csv
				if csv == "note.csv":
					notes = pd.read_csv(path)
					note_node_features = torch.from_numpy(notes[self.note_features].to_numpy())
					note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
				elif csv == "rest.csv":      
					rests = pd.read_csv(path)
					a = rests[self.rest_features].to_numpy()
					rest_node_features = torch.from_numpy(np.hstack((a,np.zeros((a.shape[0],1)))))
					rest_node_labels = torch.from_numpy(rests['label'].astype('category').cat.codes.to_numpy()).long()
				else :
					name = tuple(csv.split(".")[0].split("-"))
					edges_data = pd.read_csv(path)
					if edges_data.empty:   
						edges_src = torch.tensor([0])
						edges_dst = torch.tensor([0])                       
					else :
						edges_src = torch.from_numpy(edges_data['src'].to_numpy())
						edges_dst = torch.from_numpy(edges_data['des'].to_numpy())
					edge_dict[name] = (edges_src, edges_dst)
					if self.add_inverse_edges:
						inv_name = (name[2], name[1]+"_inv", name[0])
						edge_dict[inv_name] = (edges_dst, edges_src)

			self.graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" :rests.shape[0]})
			self.graph.nodes['note'].data['feat'] = note_node_features.float()
			self.graph.nodes['note'].data['label'] = note_node_labels
			self.graph.nodes['rest'].data['feat'] = rest_node_features.float()
			self.graph.nodes['rest'].data['label'] = rest_node_labels
		else:             
			for fn in self.PIECE_LIST:
				print(fn)
				edge_dict = dict()
				for csv in self.FILE_LIST:
					path = self.url + "/" + fn + "/" + csv
					if csv == "note.csv":
						notes = pd.read_csv(path)
						a = notes[self.note_features].to_numpy()
						if self.normalize:
							a = min_max_scaler(a)
						note_node_features = torch.from_numpy(a)
						note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
					elif csv == "rest.csv":      
						rests = pd.read_csv(path)
						a = rests[self.rest_features].to_numpy()
						if self.normalize:
							a = min_max_scaler(a)
						if len(self.note_features) > len(self.rest_features):
							rest_node_features = torch.from_numpy(np.hstack((a,np.zeros((a.shape[0],1)))))
						else : 
							rest_node_features = torch.from_numpy(a)
						rest_node_labels = torch.from_numpy(rests['label'].astype('category').cat.codes.to_numpy()).long()
					# elif csv == "ts.csv":
					#   ts = pd.read_csv(path)
					#   rest_node_features = torch.from_numpy(np.zeros((ts.shape[0],3)))
					#   rest_node_labels = torch.from_numpy(rests['label'].astype('category').cat.codes.to_numpy()).long()
					else :
						name = tuple(csv.split(".")[0].split("-"))
						edges_data = pd.read_csv(path)
						if edges_data.empty:   
							edges_src = torch.tensor([0])
							edges_dst = torch.tensor([0])                       
						else :
							edges_src = torch.from_numpy(edges_data['src'].to_numpy())
							edges_dst = torch.from_numpy(edges_data['des'].to_numpy())
						edge_dict[name] = (edges_src, edges_dst)
						if self.add_inverse_edges:
							inv_name = (name[2], name[1]+"_inv", name[0])
							edge_dict[inv_name] = (edges_dst, edges_src)


				try:
					g = self.graph
					
					graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" :rests.shape[0]})
					graph.nodes['note'].data['feat'] = note_node_features.float()
					graph.nodes['note'].data['label'] = note_node_labels
					graph.nodes['rest'].data['feat'] = rest_node_features.float()
					graph.nodes['rest'].data['label'] = rest_node_labels
					self.graph = dgl.batch_hetero([g, graph])
				except AttributeError:
					self.graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" : rests.shape[0]})
					self.graph.nodes['note'].data['feat'] = note_node_features.float()
					self.graph.nodes['note'].data['label'] = note_node_labels
					self.graph.nodes['rest'].data['feat'] = rest_node_features.float()
					self.graph.nodes['rest'].data['label'] = rest_node_labels


				# Perform Data Augmentation
				if self.add_aug:
					for _ in range(5):
						g = self.graph
						resize_factor = random.choice([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5])
						n = random.choice(range(-6, 6, 1))
						if resize_factor != 1 or n != 0:
							dur_resize = torch.tensor([resize_factor, resize_factor, resize_factor, 1]).float()
							pitch_aug = torch.tensor([0, 0, 0, n]).float()
							graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" :rests.shape[0]})
							graph.nodes['note'].data['feat'] = note_node_features.float()*dur_resize + pitch_aug
							graph.nodes['note'].data['label'] = note_node_labels
							graph.nodes['rest'].data['feat'] = rest_node_features.float()
							graph.nodes['rest'].data['label'] = rest_node_labels
							self.graph = dgl.batch_hetero([g, graph])


		# If your dataset is a node classification dataset, you will need to assign
		# masks indicating whether a node belongs to training, validation, and test set.
		self.num_classes = int(self.graph.nodes['note'].data['label'].max().item() + 1)
		self.predict_category = "note"
		n_nodes = self.graph.num_nodes("note")
		n_train = int(n_nodes * 0.8)

		train_mask = torch.zeros(n_nodes, dtype=torch.bool)
		test_mask = torch.zeros(n_nodes, dtype=torch.bool)
		train_mask[:n_train] = True
		test_mask[n_train :] = True
		self.graph.nodes['note'].data['train_mask'] = train_mask
		self.graph.nodes['note'].data['test_mask'] = test_mask

		
		r_nodes = self.graph.num_nodes("rest")
		train_mask = torch.zeros(r_nodes, dtype=torch.bool)
		test_mask = torch.zeros(r_nodes, dtype=torch.bool)
		self.graph.nodes['rest'].data['train_mask'] = train_mask
		self.graph.nodes['rest'].data['test_mask'] = test_mask

	def __getitem__(self, i):
		return self.graph

	def __len__(self):
		return 1

	def load(self):
		pass

	def save(self):
		# save graphs and labels
		graph_path = os.path.join(self.save_dir, 'mozart_piano_sonatas.bin')
		save_graphs(graph_path, self.graph)
		# save other information in python dict
		info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
		save_info(info_path, {'num_classes': self.num_classes, "predict_category" : self.predict_category, "features" : self.note_features})



class MPGD_cad(MozartPianoGraphDataset):
	def __init__(self, raw_dir=None, add_inverse_edges=False, add_aug=True, select_piece=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mps_ts_att_cadlab"
		super().__init__(name='mpgd_cad', url=url)


class MPGD_onset(MozartPianoGraphDataset): 
	def __init__(self, raw_dir=None, add_inverse_edges=False, add_aug=True, select_piece=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mps_ts_att_onlab/"
		super().__init__(name='mpgd_onset', url=url)

class MPGD_onset_test(MozartPianoGraphDataset):
	def __init__(self, raw_dir=None, add_inverse_edges=False, add_aug=False, select_piece=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mps_ts_att_onlab/"
		super().__init__(name='mpgd_onset', url=url, raw_dir=raw_dir, 
				add_inverse_edges=add_inverse_edges, add_aug=add_aug, select_piece=select_piece, features = ["onset", "ts"], normalize=True)

class MPGD_homo_onset(MozartPianoHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mps_homo_onlab/"
		super().__init__(name='mpgd_homo_onset', url=url, 
				add_inverse_edges=add_inverse_edges, add_aug=add_aug, select_piece=select_piece, normalize=False, features=["onset", "ts"], save_path=save_path)

class toy_homo_onset(MozartPianoHomoGraphDataset):
	def __init__(self, add_inverse_edges=False, add_aug=True, select_piece=None, save_path=None):
		# url = os.path.dirname("C:\\Users\\melki\\Desktop\\JKU\\codes\\tonnetzcad\\node_classification\\toy_homo_onlab\\")
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/toy_homo_onlab/"
		super().__init__(name='toy_homo_onset', url=url, 
				add_inverse_edges=add_inverse_edges, add_aug=add_aug, select_piece=select_piece, normalize=False, features=["onset", "ts"], save_path=save_path)


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
