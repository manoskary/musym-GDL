"""Entity Classification for Music with Relational Graph Convolutional Networks

Author : Emmanouil Karystinaios

Reference repo : https://github.com/melkisedeath/musym-GDL
"""
import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import dgl
import dgl.nn as dglnn


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import MPGD_cad, MPGD_onset 

# Define a Heterograph Conv model
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_hidden_layers=2, net_type=None):
        super().__init__()
        self.activation = True
        self.layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
        self.net_type = net_type
        
        if self.net_type == "SAGE":
            # i2h
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.SAGEConv(in_feats, hid_feats, "lstm")
                for rel in rel_names}, aggregate='sum'))
            # h2h
            for i in range(self.num_hidden_layers):
                self.layers.append(dglnn.HeteroGraphConv({
                    rel: dglnn.SAGEConv(hid_feats, hid_feats, "lstm")
                    for rel in rel_names}, aggregate='sum'))
            # h2o
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.SAGEConv(hid_feats, out_feats, "lstm")
                for rel in rel_names}, aggregate='sum'))

        elif self.net_type == "GAT":
            # i2h
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GATConv(in_feats, hid_feats, num_heads=5)
                for rel in rel_names}, aggregate='sum'))
            # h2h
            for i in range(self.num_hidden_layers):
                self.layers.append(dglnn.HeteroGraphConv({
                    rel: dglnn.GATConv(hid_feats, hid_feats, num_heads=5)
                    for rel in rel_names}, aggregate='sum'))
            # h2o
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GATConv(hid_feats, out_feats, num_heads=5)
                for rel in rel_names}, aggregate='sum'))
        else:
            # i2h
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum'))
            # h2h
            for i in range(self.num_hidden_layers):
                self.layers.append(dglnn.HeteroGraphConv({
                    rel: dglnn.GraphConv(hid_feats, hid_feats)
                    for rel in rel_names}, aggregate='sum'))
            # h2o
            self.layers.append(dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum'))


    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = {k : F.normalize(v) for k, v in inputs.items()}  
        for i, conv_l in enumerate(self.layers):
            h = conv_l(graph, h)
            if i == len(self.layers)-1:
                self.activation = False
            if self.activation:
                h = {k: F.relu(v) for k, v in h.items()}
        return h


def standarization(x):
    means = x.mean(dim=1, keepdim=True)
    stds = x.std(dim=1, keepdim=True)
    return (x - means) / stds

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.05, gamma=4):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = th.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets)
        targets = targets.type(th.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = th.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def my_loss(inputs, targets):
    if inputs.shape[1] == 2:
        loss = WeightedFocalLoss()
        return loss(inputs, targets)
    else:
        return F.cross_entropy(inputs, targets)

def sigmoid_focal_loss(logits, target, gamma=2, alpha=0.05):
    num_classes = logits.shape[1]
    dtype = target.dtype
    device = target.device
    class_range = th.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

    t = target.unsqueeze(1)
    p = th.sigmoid(logits)
    term1 = (1 - p) ** gamma * th.log(p)
    term2 = p ** gamma * th.log(1 - p)
    return th.mean(-(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha))

def softmax_focal_loss(x, target, gamma=4, alpha=0.95):
    n = x.shape[0]
    device = target.device
    range_n = th.arange(0, n, dtype=th.int64, device=device)

    pos_num =  float(x.shape[1])
    p = th.softmax(x, dim=1)
    p = p[range_n, target]
    loss = -(1-p)**gamma*alpha*th.log(p)
    return th.sum(loss) / pos_num



def main(args):
    # load graph data
    if args.dataset == 'mps_cad':
        print("Loading Mozart Sonatas For Cadence Detection")
        dataset = MPGD_cad() # select_piece = "K533-1"
    elif args.dataset == "mps_onset":
        print("Loading Mozart Sonatas For Bar Onset Detection")
        dataset = MPGD_onset()
    else:
        raise ValueError()

    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop('train_mask')
    test_mask = g.nodes[category].data.pop('test_mask')
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.nodes[category].data.pop('labels')
    label_balance = {u : th.count_nonzero(labels == u)/labels.shape[0] for u in th.unique(labels)}
    print("The label balance is :", label_balance)
    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to('cuda:%d' % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        test_idx = test_idx.cuda()

    # create model

    model = RGCN(3, args.n_hidden, num_classes, g.etypes, 
        num_hidden_layers=args.n_layers - 2, net_type= None)
    node_features = {nt: g.nodes[nt].data['feature'] for nt in g.ntypes}

    # model = EntityClassify(g,
    #                        args.n_hidden,
    #                        num_classes,
    #                        num_bases=args.n_bases,
    #                        num_hidden_layers=args.n_layers - 2,
    #                        dropout=args.dropout,
    #                        use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model(g, node_features)[category]
        loss = softmax_focal_loss(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = softmax_focal_loss(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward(g, node_features)[category]
    test_loss = softmax_focal_loss(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}| Test precision : {:.4f} | Test Recall : {:.4f} | Test F1 : {:.4f} " .format(test_acc, test_loss.item(), precision , recall, f1_score))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)