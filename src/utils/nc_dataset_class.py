import os
import numpy as np
import torch as torch
from torch import tensor
import pandas as pd
import random
import dgl
from dgl.data import DGLDataset


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

class MozartPianoGraphDataset(DGLDataset):
	def __init__(self, name, url, raw_dir=None, add_inverse_edges=False, add_aug=True, select_piece=None):
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
					note_node_features = torch.from_numpy(notes[["onset", "duration", "pitch"]].to_numpy())
					note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
				elif csv == "rest.csv":      
					rests = pd.read_csv(path)
					a = rests[["onset", "duration"]].to_numpy()
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
			self.graph.nodes['note'].data['feature'] = note_node_features.float()
			self.graph.nodes['note'].data['labels'] = note_node_labels
			self.graph.nodes['rest'].data['feature'] = rest_node_features.float()
			self.graph.nodes['rest'].data['labels'] = rest_node_labels
		else:
			if 'mozart' in self.raw_dir : 
				print(self.raw_dir)    
				if all([os.path.isdir(os.path.join(self.raw_dir, fn)) for fn in os.listdir(self.raw_dir)]):
					for fn in os.listdir(self.raw_dir):
						print(fn)
						edge_dict = dict()
						for csv in os.listdir(os.path.join(self.raw_dir, fn)):
							if csv == "note.csv":
								notes = pd.read_csv(os.path.join(self.raw_dir, fn, "note.csv"))
								note_node_features = torch.from_numpy(notes[["onset", "duration", "pitch"]].to_numpy())
								note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()

							elif csv == "rest.csv":      
								rests = pd.read_csv(os.path.join(self.raw_dir, fn, "rest.csv"))
								a = rests[["onset", "duration"]].to_numpy()
								rest_node_features = torch.from_numpy(np.hstack((a,np.zeros((a.shape[0],1)))))
								rest_node_labels = torch.from_numpy(rests['label'].astype('category').cat.codes.to_numpy()).long()
							else :
								name = tuple(csv.split(".")[0].split("-"))
								edges_data = pd.read_csv(os.path.join(self.raw_dir, fn, csv)) 
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
							graph.nodes['note'].data['feature'] = note_node_features.float()
							graph.nodes['note'].data['labels'] = note_node_labels
							graph.nodes['rest'].data['feature'] = rest_node_features.float()
							graph.nodes['rest'].data['labels'] = rest_node_labels
							self.graph = dgl.batch_hetero([g, graph])
						except AttributeError:
							self.graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" : rests.shape[0]})
							self.graph.nodes['note'].data['feature'] = note_node_features.float()
							self.graph.nodes['note'].data['labels'] = note_node_labels
							self.graph.nodes['rest'].data['feature'] = rest_node_features.float()
							self.graph.nodes['rest'].data['labels'] = rest_node_labels
			else:    

				for fn in self.PIECE_LIST:
					print(fn)
					edge_dict = dict()
					for csv in self.FILE_LIST:
						path = self.url + "/" + fn + "/" + csv
						if csv == "note.csv":
							notes = pd.read_csv(path)
							note_node_features = torch.from_numpy(notes[["onset", "duration", "ts", "pitch"]].to_numpy())
							note_node_labels = torch.from_numpy(notes['label'].astype('category').cat.codes.to_numpy()).long()
						elif csv == "rest.csv":      
							rests = pd.read_csv(path)
							a = rests[["onset", "duration", "ts"]].to_numpy()
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


					try:
						g = self.graph
						
						graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" :rests.shape[0]})
						graph.nodes['note'].data['feature'] = note_node_features.float()
						graph.nodes['note'].data['labels'] = note_node_labels
						graph.nodes['rest'].data['feature'] = rest_node_features.float()
						graph.nodes['rest'].data['labels'] = rest_node_labels
						self.graph = dgl.batch_hetero([g, graph])
					except AttributeError:
						self.graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" : rests.shape[0]})
						self.graph.nodes['note'].data['feature'] = note_node_features.float()
						self.graph.nodes['note'].data['labels'] = note_node_labels
						self.graph.nodes['rest'].data['feature'] = rest_node_features.float()
						self.graph.nodes['rest'].data['labels'] = rest_node_labels


					# Perform Data Augmentation
					if self.add_aug:
						for _ in range(5):
							g = self.graph
							resize_factor = random.choice([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5])
							n = random.choice(range(-6, 6, 1))
							if resize_factor != 1 or n != 0:
								dur_resize = torch.tensor([resize_factor, resize_factor, resize_factor, 1]).float()
								pitch_aug = torch.tensor([0, 0 , 0, n]).float()
								graph = dgl.heterograph(edge_dict, num_nodes_dict={"note" : notes.shape[0], "rest" :rests.shape[0]})
								graph.nodes['note'].data['feature'] = note_node_features.float()*dur_resize + pitch_aug
								graph.nodes['note'].data['labels'] = note_node_labels
								graph.nodes['rest'].data['feature'] = rest_node_features.float()
								graph.nodes['rest'].data['labels'] = rest_node_labels
								self.graph = dgl.batch_hetero([g, graph])


		# If your dataset is a node classification dataset, you will need to assign
		# masks indicating whether a node belongs to training, validation, and test set.
		self.num_classes = int(self.graph.nodes['note'].data['labels'].max().item() + 1)
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


class MPGD_cad(MozartPianoGraphDataset):
	def __init__(self, raw_dir=None, add_inverse_edges=False, add_aug=True, select_piece=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mps_ts_att_cadlab"
		super().__init__(name='mpgd_cad', url=url)


class MPGD_onset(MozartPianoGraphDataset):
	def __init__(self, raw_dir=None, add_inverse_edges=False, add_aug=True, select_piece=None):
		url = "https://raw.githubusercontent.com/melkisedeath/tonnetzcad/main/node_classification/mps_ts_att_onlab/"
		super().__init__(name='mpgd_onset', url=url)




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
