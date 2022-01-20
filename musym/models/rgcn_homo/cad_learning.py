import os
import torch
import torch.nn as nn
import dgl
import numpy as np
import tqdm
from sklearn.metrics import f1_score
from GraphSMOTE.models import GraphSMOTE
from musym.utils import load_and_save


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
    g, n_classes = load_and_save("cad_basis_homo", config["data_dir"])


    # training defs
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

    model = GraphSMOTE(in_feats, n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=config["num_layers"])
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()

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

    train_prediction = model.inference(train_dataloader, labels, device)
    # TODO needs to address post-processing using Dynamic Bayesian Model.

    acc = (np.argmax(y_pred, axis=1) == labels).sum() / len(labels)
    f1 = f1_score(labels, y_pred, average='macro')
    print("Train post accuracy: {:04f} | Train post Fscore: {:04f}".format(acc, f1))





if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", "--dataset", type=str, default="toy_01_homo")
    argparser.add_argument('--num-epochs', type=int, default=10)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--fan-out", default=[5, 10])
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=256)
    argparser.add_argument("--num-workers", type=int, default=0)
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--init-eweights", type=int, default=0,
                           help="Initialize learnable graph weights. Use 1 for True and 0 for false")
    argparser.add_argument("--temperature", type=float, default=0.2)
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("./data/"))
    argparser.add_argument("--unlog", action="store_true", help="Unbinds wandb.")
    args = argparser.parse_args()



    print(args)
    main(args)