import numpy.lib.recfunctions as rfn
import scipy.sparse as sp
import numpy as np
import os
import pickle

class ScoreGraph(object):
    def __init__(self, note_array, edges, name=None, edge_weights=None, labels=None):
        self.node_features = note_array.dtype.names if note_array.dtype.names else []
        self.note_array = note_array
        # Filter out string fields of structured array.
        if self.node_features:
            self.node_features = [feat for feat in self.node_features if note_array.dtype.fields[feat][0] != np.dtype('U256')]
            self.note_array = self.note_array[self.node_features]
        self.x = rfn.structured_to_unstructured(self.note_array) if self.node_features else self.note_array
        self.edge_index = edges
        self.edge_weights = edge_weights
        self.name = name
        self.y = labels

    def adj(self):
        ones = np.ones(len(self.edge_index[0]), np.uint32)
        matrix = sp.coo_matrix((ones, (self.edge_index[0], self.edge_index[1])))
        return matrix

    def save(self, save_dir):
        save_name = self.name if self.name else ''.join(random.choice(string.ascii_letters) for i in range(10))
        (os.makedirs(os.path.join(save_dir, save_name)) if not os.path.exists(os.path.join(save_dir, save_name)) else None)
        with open(os.path.join(save_dir, save_name, "x.npy"), "wb") as f:
            np.save(f, self.x)
        with open(os.path.join(save_dir, save_name, "edge_index.npy"), "wb") as f:
            np.save(f, self.edge_index)
        if isinstance(self.y, np.ndarray):
            with open(os.path.join(save_dir, save_name, "y.npy"), "wb") as f:
                np.save(f, self.y)
        if isinstance(self.edge_weights, np.ndarray):
            np.save(open(os.path.join(save_dir, save_name, "edge_weights.npy"), "wb"), self.edge_weights)
        with open(os.path.join(save_dir, save_name, 'graph_info.pkl'), 'wb') as handle:
            pickle.dump({"node_features": self.node_features}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_score_graph(load_dir, name=None):
    path = os.path.join(load_dir, name) if os.path.basename(load_dir) != name else load_dir
    if not os.path.exists(path) or not os.path.isdir(path):
        raise ValueError("The directory is not recognized.")
    x = np.load(open(os.path.join(path, "x.npy"), "rb"))
    edge_index = np.load(open(os.path.join(path, "edge_index.npy"), "rb"))
    graph_info = pickle.load(open(os.path.join(path, "graph_info.pkl"), "rb"))
    y = np.load(open(os.path.join(path, "y.npy"), "rb")) if os.path.exists(os.path.join(path, "y.npy")) else None
    edge_weights = np.load(open(os.path.join(path, "edge_weights.npy"), "rb")) if os.path.exists(os.path.join(path, "edge_weights.npy")) else None
    name = name if name else os.path.basename(path)
    note_array = rfn.unstructured_to_structured(x, dtype=np.dtype(list(map(lambda x: (x, "<f4"), graph_info["node_features"]))))
    return ScoreGraph(note_array=note_array, edges=edge_index, name=name, labels=y, edge_weights=edge_weights)