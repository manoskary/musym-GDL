import os
import time
import torch
import torch.nn as nn
import dgl
import numpy as np
import tqdm
from models import ContrastiveGraphSMOTE, SageClassifier
from musym.models.rgcn_homo.GraphSMOTE.data_utils import load_imbalanced_local
from pytorch_metric_learning.losses import SupConLoss

def main(args):
    """
    Main Call for Node Classification with Gaug.
    """
    #--------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    config["num_layers"] = len(config["fan_out"])
    config["shuffle"] = bool(config["shuffle"])
    config["log"] = False if config["unlog"] else True

    # --------------- Dataset Loading -------------------------
    # g, n_classes = load_and_save("cad_basis_homo", config["data_dir"])
    g, n_classes = load_imbalanced_local("BlogCatalog")

    if "train_mask" in g.ndata.keys() and "val_mask" in g.ndata.keys() and "test_mask" in g.ndata.keys():
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        # --------------- Manually build train masks ------------------------
        print("Using manual Masks ...")
        train_mask = val_mask = test_mask = torch.zeros(g.num_nodes())

        rand_inds = torch.tensor(range(g.num_nodes()))
        np.random.shuffle(rand_inds)
        train_inds = torch.tensor(rand_inds[:int(0.7*g.num_nodes())])
        val_inds = torch.tensor(rand_inds[int(0.7 * g.num_nodes()) : int(0.8 * g.num_nodes())])
        test_inds = torch.tensor(rand_inds[int(0.8*g.num_nodes()):])

        train_mask[train_inds] = 1
        val_mask[val_inds] = 1
        test_mask[test_inds] = 1


    # Hack to track node idx in dataloader's subgraphs.
    labels = g.ndata['label']
    train_nids = torch.nonzero(g.ndata.pop('train_mask'), as_tuple=True)[0]
    node_features = g.ndata['feat']


    # Validation and Testing
    val_nids = torch.nonzero(g.ndata.pop('val_mask'), as_tuple=True)[0]
    test_nids = torch.nonzero(g.ndata.pop('test_mask'), as_tuple=True)[0]
    # check cuda
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % torch.cuda.current_device() if use_cuda else 'cpu')
    dataloader_device = "cpu"
    # create model
    in_feats = node_features.shape[1]


    if isinstance(config["fan_out"], str):
        fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    else :
        fanouts = config["fan_out"]

    if use_cuda:
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

    model = ContrastiveGraphSMOTE(in_feats, n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=config["num_layers"])
    classifier = SageClassifier(in_feats=config["num_hidden"], n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=1)
    if use_cuda:
        model = model.cuda()
        classifier = classifier.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = SupConLoss()
    cl_criterion = nn.CrossEntropyLoss()
    cl_optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # training loop
    print("start training...")

    for epoch in range(config["num_epochs"]):
        model.train()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True, desc="Pretraining Epoch %i" % epoch)):
            batch_inputs = mfgs[0].srcdata['feat']
            batch_labels = mfgs[-1].dstdata['label']
            adj = mfgs[-1].adj().to_dense()[:len(batch_labels), :len(batch_labels)].to(device)
            # Prediction of the Conctrastive model
            pred, upsampl_lab, embed_loss = model(mfgs, batch_inputs, adj, batch_labels)
            loss = criterion(pred, upsampl_lab) + embed_loss * 0.000001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for epoch in range(config["num_epochs"]):
        model.eval()
        classifier.train()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True, desc="Classifier Epoch %i" % epoch)):
            batch_inputs = mfgs[0].srcdata['feat']
            batch_labels = mfgs[-1].dstdata['label']
            # Encoding of the Conctrastive model
            batch_pred, prev_encs = model.encoder(mfgs, batch_inputs)
            pred_adj = model.decoder(batch_pred, prev_encs)
            if pred_adj.get_device() >= 0:
                pred_adj = torch.where(pred_adj >= 0.5, pred_adj,
                                       torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
            else:
                pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
            # Classifier Prediction
            batch_pred = classifier(pred_adj, batch_pred, prev_encs)
            acc = (torch.argmax(batch_pred, dim=1) == batch_labels).float().sum() / len(pred)
            cl_loss = cl_criterion(batch_pred, batch_labels)
            cl_optimizer.zero_grad()
            cl_loss.backward()
            cl_optimizer.step()
        print("Step {:04d} | Accuracy : {:.4f} | Loss : {:.4f} |".format(step, acc, loss))

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_nids,
        sampler,
        device=device,
        shuffle=config["shuffle"],
        batch_size=config["batch_size"],
        drop_last=False,
        num_workers=config["num_workers"],
    )

    acc = 0
    with torch.no_grad():
        for step, (input_nodes, output_nodes, mfgs) in enumerate(
                tqdm.tqdm(test_dataloader, position=0, leave=True, desc="Epoch %i" % epoch)):
            batch_inputs = mfgs[0].srcdata['feat']
            batch_labels = mfgs[-1].dstdata['label']
            # Encoding of the Conctrastive model
            batch_pred, prev_encs = model.encoder(mfgs, batch_inputs)
            pred_adj = model.decoder(batch_pred, prev_encs)
            if pred_adj.get_device() >= 0:
                pred_adj = torch.where(pred_adj >= 0.5, pred_adj,
                                       torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
            else:
                pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
            # Classifier Prediction
            batch_pred = classifier(pred_adj, batch_pred, prev_encs)
            acc += (torch.argmax(batch_pred, dim=1) == batch_labels).float().sum() / len(pred)
    acc = acc/(step+1)
    print("Test accuracy: {:04f}".format(acc))



if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", "--dataset", type=str, default="toy_01_homo")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--fan-out", default=[5, 10])
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=100)
    argparser.add_argument("--num-workers", type=int, default=0)
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--init-eweights", type=int, default=0,
                           help="Initialize learnable graph weights. Use 1 for True and 0 for false")
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
    argparser.add_argument("--unlog", action="store_true", help="Unbinds wandb.")
    args = argparser.parse_args()



    print(args)
    main(args)