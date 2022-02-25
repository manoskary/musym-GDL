import os

import hmmlearn.hmm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import tqdm
from sklearn.metrics import f1_score
from GraphSMOTE.models import GraphSMOTE
from musym.utils import load_and_save
# import pyro
# from pyro.distributions import Normal
# from pyro.distributions import Categorical
# from pyro.optim import Adam
# from pyro.infer import SVI
# from pyro.infer import Trace_ELBO
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from dgl.sampling import node2vec_random_walk
from torchmetrics.functional import f1


class Node2vec(nn.Module):
    """Node2vec model from paper node2vec: Scalable Feature Learning for Networks <https://arxiv.org/abs/1607.00653>
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.  Same notation as in the paper.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
        Same notation as in the paper.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, use PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly.
    """

    def __init__(self, g, embedding_dim, walk_length, p, q, num_walks=10, window_size=5, num_negatives=5,
                 use_sparse=True, weight_name=None):
        super(Node2vec, self).__init__()

        assert walk_length >= window_size

        self.g = g
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.num_walks = num_walks
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.N = self.g.num_nodes()
        if weight_name is not None:
            self.prob = weight_name
        else:
            self.prob = None

        self.embedding = nn.Embedding(self.N, embedding_dim, sparse=use_sparse)

    def reset_parameters(self):
        self.embedding.reset_parameters()

    def sample(self, batch):
        """
        Generate positive and negative samples.
        Positive samples are generated from random walk
        Negative samples are generated from random sampling
        """
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)

        batch = batch.repeat(self.num_walks)
        # positive
        pos_traces = node2vec_random_walk(self.g, batch, self.p, self.q, self.walk_length, self.prob)
        pos_traces = pos_traces.unfold(1, self.window_size, 1)  # rolling window
        pos_traces = pos_traces.contiguous().view(-1, self.window_size)

        # negative
        neg_batch = batch.repeat(self.num_negatives)
        neg_traces = torch.randint(self.N, (neg_batch.size(0), self.walk_length))
        neg_traces = torch.cat([neg_batch.view(-1, 1), neg_traces], dim=-1)
        neg_traces = neg_traces.unfold(1, self.window_size, 1)  # rolling window
        neg_traces = neg_traces.contiguous().view(-1, self.window_size)

        return pos_traces, neg_traces

    def forward(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding
        """
        emb = self.embedding.weight
        if nodes is None:
            return emb
        else:
            return emb[nodes]

    def loss(self, pos_trace, neg_trace):
        """
        Computes the loss given positive and negative random walks.
        Parameters
        ----------
        pos_trace: Tensor
            positive random walk trace
        neg_trace: Tensor
            negative random walk trace
        """
        e = 1e-15

        # Positive
        pos_start, pos_rest = pos_trace[:, 0], pos_trace[:, 1:].contiguous()  # start node and following trace
        w_start = self.embedding(pos_start).unsqueeze(dim=1)
        w_rest = self.embedding(pos_rest)
        pos_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # Negative
        neg_start, neg_rest = neg_trace[:, 0], neg_trace[:, 1:].contiguous()

        w_start = self.embedding(neg_start).unsqueeze(dim=1)
        w_rest = self.embedding(neg_rest)
        neg_out = (w_start * w_rest).sum(dim=-1).view(-1)

        # compute loss
        pos_loss = -torch.log(torch.sigmoid(pos_out) + e).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_out) + e).mean()

        return pos_loss + neg_loss

    def loader(self, batch_size):
        """
        Parameters
        ----------
        batch_size: int
            batch size
        Returns
        -------
        DataLoader
            Node2vec training data loader
        """
        return DataLoader(torch.arange(self.N), batch_size=batch_size, shuffle=True, collate_fn=self.sample)

    @torch.no_grad()
    def evaluate(self, x_train, y_train, x_val, y_val):
        """
        Evaluate the quality of embedding vector via a downstream classification task with logistic regression.
        """
        x_train = self.forward(x_train)
        x_val = self.forward(x_val)

        x_train, y_train = x_train.cpu().numpy(), y_train.cpu().numpy()
        x_val, y_val = x_val.cpu().numpy(), y_val.cpu().numpy()
        lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150).fit(x_train, y_train)

        return lr.score(x_val, y_val)


class Node2vecModel(object):
    """
    Wrapper of the ``Node2Vec`` class with a ``train`` method.
    Attributes
    ----------
    g: DGLGraph
        The graph.
    embedding_dim: int
        Dimension of node embedding.
    walk_length: int
        Length of each trace.
    p: float
        Likelihood of immediately revisiting a node in the walk.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
    num_walks: int
        Number of random walks for each node. Default: 10.
    window_size: int
        Maximum distance between the center node and predicted node. Default: 5.
    num_negatives: int
        The number of negative samples for each positive sample.  Default: 5.
    use_sparse: bool
        If set to True, uses PyTorch's sparse embedding and optimizer. Default: ``True``.
    weight_name : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.
        If omitted, DGL assumes that the neighbors are picked uniformly. Default: ``None``.
    eval_set: list of tuples (Tensor, Tensor)
        [(nodes_train,y_train),(nodes_val,y_val)]
        If omitted, model will not be evaluated. Default: ``None``.
    eval_steps: int
        Interval steps of evaluation.
        if set <= 0, model will not be evaluated. Default: ``None``.
    device: str
        device, default 'cpu'.
    """

    def __init__(self, g, embedding_dim, walk_length, p=1.0, q=1.0, num_walks=1, window_size=5,
                 num_negatives=5, use_sparse=True, weight_name=None, eval_set=None, eval_steps=-1, device='cpu'):

        self.model = Node2vec(g, embedding_dim, walk_length, p, q, num_walks,
                              window_size, num_negatives, use_sparse, weight_name)
        self.g = g
        self.use_sparse = use_sparse
        self.eval_steps = eval_steps
        self.eval_set = eval_set

        if device == 'cpu':
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _train_step(self, epoch, model, loader, optimizer, device):
        model.train()
        total_loss = 0
        for pos_traces, neg_traces in tqdm.tqdm(loader, position=0, leave=True, desc="Pretraining Epoch %i" % epoch):
            pos_traces, neg_traces = pos_traces.to(device), neg_traces.to(device)
            optimizer.zero_grad()
            loss = model.loss(pos_traces, neg_traces)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def _evaluate_step(self):
        nodes_train, y_train = self.eval_set[0]
        nodes_val, y_val = self.eval_set[1]

        acc = self.model.evaluate(nodes_train, y_train, nodes_val, y_val)
        return acc

    def train(self, epochs, batch_size, learning_rate=0.01):
        """
        Parameters
        ----------
        epochs: int
            num of train epoch
        batch_size: int
            batch size
        learning_rate: float
            learning rate. Default 0.01.
        """

        self.model = self.model.to(self.device)
        loader = self.model.loader(batch_size)
        if self.use_sparse:
            optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            loss = self._train_step(epoch, self.model, loader, optimizer, self.device)
            if self.eval_steps > 0:
                if epochs % self.eval_steps == 0:
                    acc = self._evaluate_step()
                    print("Epoch: {}, Train Loss: {:.4f}, Val Acc: {:.4f}".format(epoch, loss, acc))

    def embedding(self, nodes=None):
        """
        Returns the embeddings of the input nodes
        Parameters
        ----------
        nodes: Tensor, optional
            Input nodes, if set `None`, will return all the node embedding.
        Returns
        -------
        Tensor
            Node embedding.
        """

        return self.model(nodes)


def to_sequences(labels, preds, idx, score_duration, piece_idx):
    seqs = list()
    trues = list()
    # Make args same dimensions as preds
    piece_idx = piece_idx[idx]
    labels = labels[idx]
    score_duration = score_duration[idx]
    # Start Building Sequence per piece name.
    for name in torch.unique(piece_idx):
        # Gather on non-augmented Pieces
        if name != 0:
            durs = score_duration[piece_idx == name]
            X = preds[piece_idx == name]
            y = labels[piece_idx == name]
            sorted_durs, resorted_idx = torch.sort(durs)
            X = X[resorted_idx]
            y = y[resorted_idx]
            new_X = []
            new_y = []
            # group by onset
            for udur in torch.unique(sorted_durs):
                x = X[sorted_durs == udur]
                z = y[sorted_durs == udur]
                if len(x.shape) > 1:
                    # Can be Max or Mean aggregation of likelihoods.
                    new_X.append(x.mean(dim=0))
                    new_y.append(z.max().unsqueeze(0))
                else:
                    new_X.append(x)
                    new_y.append(z)
            seqs.append(torch.vstack(new_X).numpy())
            trues.append(torch.cat(new_y).numpy())
    return seqs, trues


def main(args):
    """
    Main Call for Node Classification with Node2Vec + GraphSMOTE + DBNN.
    """
    #--------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    config["num_layers"] = len(config["fan_out"])
    config["shuffle"] = bool(config["shuffle"])

    # --------------- Dataset Loading -------------------------
    g, n_classes = load_and_save("cad_basis_homo", config["data_dir"])
    g = dgl.add_self_loop(dgl.add_reverse_edges(g))
    # training defs
    labels = g.ndata.pop('label')
    train_nids = torch.nonzero(g.ndata.pop('train_mask'), as_tuple=True)[0]
    node_features = g.ndata.pop('feat')
    piece_idx = g.ndata.pop("score_name")
    score_duration = node_features[:, 3]
    node_features = F.normalize(node_features)

    # Validation and Testing
    val_nids = torch.nonzero(g.ndata.pop('val_mask'), as_tuple=True)[0]
    test_nids = torch.nonzero(g.ndata.pop('test_mask'), as_tuple=True)[0]
    # check cuda
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % torch.cuda.current_device() if use_cuda else 'cpu')
    dataloader_device = "cpu"


    # ------------ Pre-Processing Node2Vec ----------------------
    emb_path = os.path.join(config["data_dir"], "cad_basis_homo", "node_emb.pt")
    nodes = g.nodes()
    if config["preprocess"]:
        nodes_train, y_train = nodes[train_nids], labels[train_nids]
        nodes_val, y_val = nodes[val_nids], labels[val_nids]
        eval_set = [(nodes_train, y_train), (nodes_val, y_val)]
        pp_model = Node2vecModel(g=g, embedding_dim=256, walk_length=50, p=0.25, q=4.0, num_walks=10, device=device, eval_set=eval_set, eval_steps=1)
        pp_model.train(epochs=5, batch_size=128)
        node_emb = pp_model.embedding().detach().cpu()
        node_features = torch.cat((node_features, node_emb), dim=1)
        torch.save(node_features, emb_path)

    try:
        node_features = torch.load(emb_path)
    except:
        print("Node embedding was not found continuing with standard node features.")

    # create model
    in_feats =  node_features.shape[1]


    if isinstance(config["fan_out"], str):
        fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    else :
        fanouts = config["fan_out"]

    if use_cuda and config["num_workers"]==0:
        train_nids = train_nids.to(device)
        test_nids = test_nids.to(device)
        g = g.formats(['csc'])
        g = g.to(device)
        dataloader_device = device

    # dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    # The edge dataloader returns a subgraph but iterates on the number of edges.
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nids,
        sampler,
        device=dataloader_device,
        shuffle=config["shuffle"],
        batch_size = config["batch_size"],
        drop_last=False,
        num_workers=config["num_workers"],
    )
    model_path = os.path.join(config["data_dir"], "cad_basis_homo", "model_sd.pt")
    model = GraphSMOTE(in_feats, n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=config["num_layers"],
                       ext_mode=config["ext_mode"])
    print("Model Trainable Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()


    if config["load_model"]:
        model.load_state_dict(torch.load(model_path))
    else:
        # training loop
        print("start training...")

        for epoch in range(config["num_epochs"]):
            model.train()
            acc = list()
            f1_score = 0
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True, desc="Training Epoch %i" % epoch)):
                batch_inputs = node_features[input_nodes].to(device)
                batch_labels = labels[output_nodes].to(device)
                if dataloader_device == "cpu":
                    mfgs = [mfg.int().to(device) for mfg in mfgs]
                adj = mfgs[-1].adj().to_dense()[:len(batch_labels), :len(batch_labels)].to(device)
                pred, upsampl_lab, embed_loss = model(mfgs, batch_inputs, adj, batch_labels)
                loss = criterion(pred, upsampl_lab) + embed_loss * config["gamma"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc.append((torch.argmax(pred[:len(batch_labels)], dim=1) == batch_labels))
                f1_score += f1(torch.argmax(pred[:len(batch_labels)], dim=1), batch_labels, average="macro", num_classes=n_classes).item()

            f1_score = f1_score/ (step+1)
            acc_score = torch.cat(acc).float().sum() / len(labels)
            print("Epoch {:04d} | Loss {:.04f} | Acc {:.04f} | F score {:.04f} |".format(epoch, loss, acc_score, f1_score))
        torch.save(model.state_dict(), model_path)

    model.eval()
    train_prediction = model.inference(train_dataloader, node_features, labels[train_nids], device)
    # pred_path = os.path.join(config["data_dir"], "cad_basis_homo", "preds.pt")
    # posttrain_label_path = os.path.join(config["data_dir"], "cad_basis_homo", "post_train_labels.pt")
    # torch.save(train_prediction.detach().cpu(), pred_path)
    # torch.save(labels[train_nids].detach().cpu(), posttrain_label_path)
    # TODO needs to address post-processing using Dynamic Bayesian Model.
    X_train = train_prediction.detach().cpu()
    y_train = labels[train_nids].detach().cpu()
    if config["eval"]:
        val_dataloader = dgl.dataloading.NodeDataLoader(
            g,
            torch.cat((val_nids, test_nids)),
            sampler,
            device=dataloader_device,
            shuffle=config["shuffle"],
            batch_size = config["batch_size"],
            drop_last=False,
            num_workers=config["num_workers"],
        )
        idx = torch.cat((val_nids, test_nids))
        val_prediction = model.inference(val_dataloader, node_features, labels, device)
        acc = torch.eq(labels[idx], torch.argmax(val_prediction[idx].cpu(), dim=1)).float().sum() / len(labels[idx])
        fscore = f1(val_prediction[idx].cpu(), labels[idx], average="macro", num_classes=2)
        Χ_val = val_prediction[idx].detach().cpu()
        y_val = labels[idx].detach().cpu()
        print("Validation Score : Accuracy {:.4f} | F score {:.4f} |".format(acc, fscore))
        return X_train, y_train, Χ_val, y_val
    else:
        print(X_train.shape, y_train.shape)
        return to_sequences(labels, X_train, train_nids, score_duration, piece_idx)


if __name__ == '__main__':
    import argparse
    import pickle
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-hidden', type=int, default=128)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=0.001123)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--gamma", type=float, default=0.001248,
                           help="weight of decoder regularization loss.")
    argparser.add_argument("--ext-mode", type=str, default=None, choices=["lstm", "attention"])
    argparser.add_argument("--fan-out", default=[5, 10])
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=2048)
    argparser.add_argument("--num-workers", type=int, default=10)
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("./data/"))
    argparser.add_argument("--preprocess", action="store_true", help="Train and store graph embedding")
    argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
    argparser.add_argument("--load-model", action="store_true", help="Load pretrained model.")
    argparser.add_argument("--eval", action="store_true", help="Preview Results on Validation set.")
    args = argparser.parse_args()



    print(args)
    if not args.postprocess:
        X_train, y_train = main(args)
        pred_path = os.path.join(args.data_dir, "cad_basis_homo", "preds.pkl")
        posttrain_label_path = os.path.join(args.data_dir, "cad_basis_homo", "post_train_labels.pkl")
        with open(pred_path, "wb") as f:
            pickle.dump(X_train, f)
        with open(posttrain_label_path, "wb") as f:
            pickle.dump(y_train, f)
    else :
        pred_path = os.path.join(args.data_dir, "cad_basis_homo", "preds.pkl")
        posttrain_label_path = os.path.join(args.data_dir, "cad_basis_homo", "post_train_labels.pkl")
        # X_train, y_train = torch.load(pred_path).numpy(), torch.load(posttrain_label_path).numpy()
        with open(pred_path, "rb") as f:
            X_train = pickle.load(f)
        with open(posttrain_label_path, "rb") as f:
            y_train = pickle.load(f)

        print("Start Post-processing...")



        # -------------- POMEGRANATE HMM --------------------

        # import pomegranate as pg
        # from sklearn.metrics import f1_score
        #
        # trans_mat = np.array([[0.98, 1.],
        #                       [0.02, 0.]])
        # starts = np.array([1.0, 0.0])
        # ends = np.array([1.0, 0.0])
        # model = pg.HiddenMarkovModel.from_matrix(trans_mat, dists, starts, ends)
        # y_pred = model.predict(X_train, algorithm='viterbi')


        # ------------- HMM Learn -------------------------
        from hmmlearn import hmm

        pm = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params="cm", params="cmt")
        pm.startprob_ = np.array([1.0, 0.0])
        pm.transmat_ = np.array([[0.95, 0.05],
                                  [1.00, 0.00]])
        pm.fit(np.concatenate(X_train), [len(x) for x in X_train])
        over_acc = 0
        over_f1 = 0
        for i, seq in enumerate(X_train):
            y_pred = pm.predict(seq)
            acc = np.equal(y_train[i], y_pred).astype(float).sum() / len(y_pred)
            over_acc += acc
            fscore = f1_score(y_train[i], y_pred, average="macro")
            over_f1 += fscore
            print("Post-Process Model: Accuracy {:.4f} | F score {:.4f} |".format(acc, fscore))
        over_acc = over_acc / (i+1)
        over_f1 = over_f1 / (i+1)
        print("Mean Post-Process Model: Accuracy {:.4f} | F score {:.4f} |".format(over_acc, over_f1))

        over_acc = 0
        over_f1 = 0
        for i, seq in enumerate(X_train):
            y_pred = X_train[i].argmax(axis=1)
            acc = np.equal(y_train[i], y_pred).astype(float).sum() / len(y_pred)
            over_acc += acc
            fscore = f1_score(y_train[i], y_pred, average="macro")
            over_f1 += fscore
        over_acc = over_acc / (i+1)
        over_f1 = over_f1 / (i+1)
        print("Mean Post-Process Thresholding: Accuracy {:.4f} | F score {:.4f} |".format(over_acc, over_f1))



