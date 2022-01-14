import os.path
import random
import torch
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import dgl

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def refine_label_order(labels):
    IMBALANCE_THRESH = 101
    max_label = labels.max()
    j = 0

    for i in range(labels.max(),0,-1):
        if sum(labels==i) >= IMBALANCE_THRESH and i>j:
            while sum(labels==j) >= IMBALANCE_THRESH and i>j:
                j = j+1
            if i > j:
                head_ind = labels == j
                tail_ind = labels == i
                labels[head_ind] = i
                labels[tail_ind] = j
                j = j+1
            else:
                break
        elif i <= j:
            break

    return labels


def load_cora_local(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    #input: idx_features_labels, adj
    #idx,labels are not required to be processed in advance
    #adj: save in the form of edges. idx1 idx2
    #output: adj, features, labels are all torch.tensor, in the dense form
    #-------------------------------------------------------

    path = os.path.join(os.path.dirname(__file__), path)
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = idx_features_labels[:, -1]
    set_labels = set(labels)
    classes_dict = {c: np.arange(len(set_labels))[i] for i, c in enumerate(set_labels)}
    classes_dict = {'Neural_Networks': 0, 'Reinforcement_Learning': 1, 'Probabilistic_Methods': 2, 'Case_Based': 3, 'Theory': 4, 'Rule_Learning': 5, 'Genetic_Algorithms': 6}

    #ipdb.set_trace()
    labels = np.array(list(map(classes_dict.get, labels)))

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    g = dgl.from_scipy(adj, eweight_name="w")
    g.ndata["label"] = labels
    g.ndata["feat"] = features
    return g


def load_data_Blog():#
    #--------------------
    #
    #--------------------
    pathjoin = lambda x : os.path.join(os.path.dirname(__file__), os.path.normpath(x))
    mat = loadmat(pathjoin('data/BlogCatalog/blogcatalog.mat'))
    adj = mat['network']
    label = mat['group']

    embed = np.loadtxt(pathjoin('data/BlogCatalog/blogcatalog.embeddings_64'))
    feature = np.zeros((embed.shape[0],embed.shape[1]-1))
    feature[embed[:,0].astype(int),:] = embed[:,1:]

    features = normalize(feature)
    labels = np.array(label.todense().argmax(axis=1)).squeeze()

    labels[labels>16] = labels[labels>16]-1

    print("change labels order, imbalanced classes to the end.")
    labels = refine_label_order(labels)

    g = dgl.from_scipy(adj, eweight_name="w")
    g.ndata["label"] = torch.LongTensor(labels)
    g.ndata["feat"] = torch.FloatTensor(features)
    return g


def load_data_reddit():
    dataset = dgl.data.RedditDataset()
    g = dataset[0]
    print(
        "The Reddit Dataset : \n num classes = {} \n feature size = {} \n Num nodes = {} ".format(
            dataset.num_classes,
            g.ndata["feat"].shape[1],
            g.num_nodes()
        )
    )
    print()
    return g


def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+25]
        test_idx = test_idx + c_idx[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat


def split_genuine(labels):
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)

        train_idx = train_idx + c_idx[:c_num_mat[i,0]]
        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]
    random.shuffle(train_idx)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return train_idx, val_idx, test_idx, c_num_mat


def load_imbalanced_local(name):
    """
    Loads imbalanced local dataset similar to the GraphSMOTE paper.

    Parameters
    ----------
    name : string in {'cora', 'BlogCatalog'}
        The name of the dataset to load.
    Returns
    -------
    g : dgl.graph
    n_classes : int
    """
    if name == 'cora':
        g = load_cora_local()
        class_sample_num = 20
        im_class_num = 3
    elif name == "BlogCatalog":
        g = load_data_Blog()
        im_class_num = 14  # set it to be the number less than 100
        class_sample_num = 20
    elif name == "reddit":
        g = load_data_reddit()
        im_class_num = 5
        class_sample_num = 20  # not used
    else:
        raise ValueError("Unknown dataset name")

    labels = g.ndata["label"]
    # for artificial imbalanced setting: only the last im_class_num classes are imbalanced
    c_train_num = []
    imbalance = True
    im_ratio = 0.6
    n_classes = labels.max().item() + 1
    for i in range(n_classes):
        if imbalance and i > labels.max().item() - im_class_num:  # only imbalance the last classes
            c_train_num.append(int(class_sample_num * im_ratio))

        else:
            c_train_num.append(class_sample_num)
    idx_train, idx_val, idx_test, class_num_mat = split_arti(labels, c_train_num) if name == "cora" else split_genuine(labels)
    train_mask = torch.zeros(g.num_nodes())
    val_mask = torch.zeros(g.num_nodes())
    test_mask = torch.zeros(g.num_nodes())
    train_mask[idx_train] = 1
    val_mask[idx_val] = 1
    test_mask[idx_test] = 1
    g.ndata["train_mask"] = train_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["test_mask"] = test_mask
    return g, n_classes


if __name__ == '__main__':
    # g, n_classes = load_imbalanced_local("reddit")
    g, n_classes = load_imbalanced_local("cora")
    # g, n_classes = load_imbalanced_local("BlogCatalog")

