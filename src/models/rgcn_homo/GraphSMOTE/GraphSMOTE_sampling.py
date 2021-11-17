import os
import sys
import time
import torch
import dgl
import numpy as np
from models import *
import tqdm
import torch.optim as optim
from imblearn.over_sampling import ADASYN, SMOTE

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))


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
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    n_classes = dataset.num_classes

    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']
    val_mask = g.ndata['val_mask']

    if config["init_eweights"]:
        w = torch.empty(g.num_edges())
        nn.init.uniform_(w)
        g.edata["w"] = w

    # Hack to track node idx in dataloader's subgraphs.
    train_g = g.subgraph(torch.nonzero(train_mask)[:, 0])
    train_g.ndata['idx'] = torch.tensor(range(train_g.number_of_nodes()))
    labels = train_g.ndata['label']
    eids = torch.arange(train_g.number_of_edges())
    node_features = train_g.ndata['feat']

    # Validation and Testing
    val_g = g.subgraph(torch.nonzero(val_mask)[:, 0])
    val_labels = val_g.ndata['label']
    test_g = g.subgraph(torch.nonzero(test_mask)[:, 0])
    test_labels = test_g.ndata['label']




    # check cuda
    use_cuda = config["gpu"] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % torch.cuda.current_device() if use_cuda else 'cpu')

    # create model
    in_feats = node_features.shape[1]


    if isinstance(config["fan_out"], str):
        fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    else :
        fanouts = config["fan_out"]

    # dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    # The edge dataloader returns a subgraph but iterates on the number of edges.
    train_dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        eids,
        sampler,
        shuffle=config["shuffle"],
        batch_size = config["batch_size"],
        drop_last=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        persistent_workers=config["num_workers"]>0,
    )

    encoder = Encoder(in_feats, config["num_hidden"], config["num_layers"], F.relu, 0.5)
    decoder = SageDecoder(config["num_hidden"])
    classifier = SageClassifier(config["num_hidden"], config["num_hidden"], n_classes, config["num_layers"] - 1)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        classifier = classifier.cuda()

    optim_enc = torch.optim.Adam(encoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optim_dec = torch.optim.Adam(decoder.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optim_clf = torch.optim.Adam(classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    dloss = EdgeLoss()

    # training loop
    print("start training...")
    dur = []
    for epoch in range(config["num_epochs"]):
        encoder.train()
        decoder.train()
        classifier.train()
        t0 = time.time()
        for step, (input_nodes, sub_g, blocks) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True)):
            batch_inputs = node_features[input_nodes].to(device)
            batch_edge_weights = dgl.nn.EdgeWeightNorm(sub_g.edata["w"]).to(device)
            # Hack to track the node idx for NodePred layer (SAGE) not the same as block or input nodes
            batch_labels = labels[sub_g.ndata['idx']].to(device)
            blocks = [block.int().to(device) for block in blocks]
            # The features for the loaded subgraph
            feat_inputs = sub_g.ndata["feat"].to(device)
            # The adjacency matrix of the subgraph
            if config["init_eweights"]:
                subgraph_shape = (sub_g.num_nodes(), sub_g.num_nodes())
                subgraph_indices = torch.vstack(sub_g.edges())
                adj = torch.sparse.FloatTensor(subgraph_indices, sub_g.edata["w"], subgraph_shape).to_dense().to(device)
            else:
                adj = sub_g.adj(ctx=device).to_dense()

            # Prediction of the GraphSMOTE model
            embed = encoder(adj, feat_inputs)
            upsampl_embed, upsampl_lab = map(lambda x: torch.tensor(x), SMOTE().fit_resample(embed.detach(), batch_labels))
            pred_adj = decoder(upsampl_embed).type(torch.double)
            embed_loss = dloss(pred_adj, adj)
            pred_adj = torch.where(pred_adj >= 0.5, pred_adj, 0.).type(torch.float32)
            pred = classifier(pred_adj, upsampl_embed)
            loss = criterion(pred, upsampl_lab) + embed_loss * 0.000001
            acc = (torch.argmax(pred, dim=1) == upsampl_lab).float().sum() / len(pred)
            optim_enc.zero_grad()
            optim_clf.zero_grad()
            optim_dec.zero_grad()
            loss.backward()
            optim_enc.step()
            optim_clf.step()
            optim_dec.step()
            t1 = time.time()
        if epoch > 5:
            dur.append(t1 - t0)
        with torch.no_grad():
            train_acc = torch.sum(torch.argmax(pred, dim=1) == batch_labels).item() / batch_labels.shape[0]
            pred = model.inference(val_g, device=device, batch_size=config["batch_size"], num_workers=config["num_workers"])
            val_loss = F.cross_entropy(pred, val_labels)
            val_acc = (torch.argmax(pred, dim=1) == val_labels.long()).float().sum() / len(pred)
        print(
            "Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Val Acc : {:.4f} | Val CE Loss: {:.4f}| Time: {:.4f}".
            format(epoch, train_acc, loss.item(), val_acc, val_loss, np.average(dur)))
        if epoch%5==0:
            model.eval()
            with torch.no_grad():
                pred = model.inference(test_g, device=device, batch_size=config["batch_size"],
                                       num_workers=config["num_workers"])
                test_loss = F.cross_entropy(pred, test_labels)
                test_acc = (torch.argmax(pred, dim=1) == test_labels.long()).float().sum() / len(pred)
            print("Test Acc: {:.4f} | Test loss: {:.4f}| ".format(test_acc, test_loss.item()))
    print()


    print()

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", "--dataset", type=str, default="toy_01_homo")
    argparser.add_argument('--num-epochs', type=int, default=100)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--fan-out", default=[5, 10])
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=512)
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