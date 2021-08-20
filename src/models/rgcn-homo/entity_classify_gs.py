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
from dgl import load_graphs
from dgl.data.utils import load_info

# Grid Search and Optimization
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# Importing Models
from models import SAGE, SGC


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import MPGD_homo_onset


def normalize(x, eps=1e-8):
    normed = (x - x.mean(axis=0) ) / (x.std(axis=0) + eps)
    return normed


def train(config, checkpoint_dir=None, data_dir=None):
    """
    Train Call for Node Classification with RGCN on Mozart Data.

    """

    # load graph data (from submodule repo)
    print("Loading Mozart Sonatas For Cadence Detection")
    if os.path.exists(data_dir):
        name = "mpgd_homo_onset"
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(data_dir, name, name + '_graph.bin')
        # Load the Homogeneous Graph
        g = load_graphs(graph_path)[0][0]
        info_path = os.path.join(data_dir, name, name + '_info.pkl')
        num_classes = load_info(info_path)['num_classes']
    else:
        dataset =  MPGD_homo_onset(save_path = data_dir) # select_piece = "K533-1"
        # Load the Homogeneous Graph
        g= dataset[0]
        num_classes = dataset.num_classes


    # category = dataset.predict_category
    # num_classes = dataset.num_classes
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
    labels = g.ndata['labels']
    # Informative label balance to choose loss (if balance is relevantly balanced change to cross_entropy loss)
    label_balance = {u : th.count_nonzero(labels == u)/labels.shape[0] for u in th.unique(labels)}
    print("The label balance is :", label_balance)

    # # split dataset into train, validate, test
    # if args.validation:
    #     val_idx = train_idx[:len(train_idx) // 5]
    #     train_idx = train_idx[len(train_idx) // 5:]
    # else:
    #     val_idx = train_idx

    # # check cuda
    # use_cuda = args.gpu >= 0 and th.cuda.is_available()
    # if use_cuda:
    #     th.cuda.set_device(args.gpu)
    #     g = g.to('cuda:%d' % args.gpu)
    #     labels = labels.cuda()
    #     train_idx = train_idx.cuda()
    #     test_idx = test_idx.cuda()

    
    # Load the node features as a Dictionary to feed to the forward layer.
    node_features = g.ndata['feature']

    # if args.init_eweights:
    #     w = th.empty(g.num_edges())
    #     nn.init.uniform_(w)
    #     g.edata["w"] = w
    
    # create model
    in_feats = node_features.shape[1]
    model = SAGE(in_feats, config["n_hidden"], num_classes,
        num_hidden_layers=config["n_layers"] - 2, aggr_type="mean")
    if use_cuda:
        model.cuda()
    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=config["lr"])

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(config["n_epochs"]):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        logits = model(g, node_features)
        # loss = softmax_focal_loss(logits[train_idx], labels[train_idx]) 
        loss = F.cross_entropy(logits[train_idx], labels[train_idx]) 
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        # val_loss = softmax_focal_loss(logits[val_idx], labels[val_idx])
        valid_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        val_loss += loss.cpu().numpy()
        val_steps += 1
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, valid_loss.item(), np.average(dur)))
    print()
    # if args.model_path is not None:
    #     th.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward(g, node_features)
    # test_loss = softmax_focal_loss(logits[test_idx], labels[test_idx])
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)

    print("Test Acc: {:.4f} | Test loss: {:.4f}| " .format(test_acc, test_loss.item()))
    print()


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=0):
    data_dir = os.path.abspath("./data")
    config = {
        "n-hidden": tune.sample_from(lambda _: 2**np.random.randint(4, 7)),
        "n_epochs": tune.choice([50, 100, 150]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "n-layers": tune.choice([2, 3, 4, 5])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))



if __name__ == '__main__':
    
    
    main()