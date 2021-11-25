import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import tqdm
from models import GraphSMOTE
from data_utils import load_imbalanced_local
from sklearn.metrics import f1_score, precision_recall_fscore_support
from musym.utils import load_and_save
import wandb

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
    # dataset = dgl.data.CoraGraphDataset()
    # g = dataset[0]
    # n_classes = dataset.num_classes

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


    # --------------------- Init WANDB ---------------------------------
    wandb.init(project="SMOTE", group="GraphSMOTE-sampling", job_type="Cadence-Detection", config=config)


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

    model = GraphSMOTE(in_feats, n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=config["num_layers"])
    if use_cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = nn.CrossEntropyLoss()

    wandb.watch(model, log_freq=1000)
    # training loop
    print("start training...")
    dur = []
    prev_fscore = 0
    for epoch in range(config["num_epochs"]):
        model.train()
        t0 = time.time()

        train_acc = 0
        train_fscore = 0
        for step, (input_nodes, sub_g, blocks) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True)):

            # batch_edge_weights = dgl.nn.EdgeWeightNorm(sub_g.edata["w"]).to(device)
            # Hack to track the node idx for NodePred layer (SAGE) not the same as block or input nodes
            # batch_labels = labels[sub_g.ndata['idx']].to(device)
            blocks = [block.int().to(device) for block in blocks]
            batch_labels = blocks[-1].dstdata["label"]
            batch_inputs = blocks[0].srcdata["feat"]
            # The features for the loaded subgraph
            # feat_inputs = sub_g.ndata["feat"].to(device)
            # The adjacency matrix of the subgraph
            if config["init_eweights"] or "w" in train_g.edata.keys():
                subgraph_shape = (sub_g.num_nodes(), sub_g.num_nodes())
                subgraph_indices = torch.vstack(sub_g.edges())
                adj = torch.sparse.FloatTensor(subgraph_indices, sub_g.edata["w"], subgraph_shape).to_dense().to(device)
            else:
                adj = sub_g.adj(ctx=device).to_dense()

            # Prediction of the GraphSMOTE model
            pred, upsampl_lab, embed_loss = model(blocks, batch_inputs, adj, batch_labels)
            loss = criterion(pred, upsampl_lab) + embed_loss * 0.000001
            acc = (torch.argmax(pred, dim=1) == upsampl_lab).float().sum() / len(pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            train_acc += acc
            train_fscore += f1_score(batch_labels.detach().cpu().numpy(), torch.argmax(pred, dim=1)[:len(batch_labels)].detach().cpu().numpy(), average='weighted')
        if epoch > 5:
            dur.append(t1 - t0)
        with torch.no_grad():
            pred = model.inference(val_g, device=device, batch_size=config["batch_size"], num_workers=config["num_workers"])
            val_fscore = f1_score(val_labels.cpu().numpy(), torch.argmax(pred, dim=1).cpu().numpy(), average='weighted')
            val_loss = F.cross_entropy(pred, val_labels)
            val_acc = (torch.argmax(pred, dim=1) == val_labels.long()).float().sum() / len(pred)
            # if val_fscore > prev_fscore:
            #     torch.save(model.state_dict(), "./data/saved_models/GraphSMOTE.pth")
            scheduler.step(val_acc)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Train f1 score {:.4f} | Val Acc : {:.4f} | Val CE Loss: {:.4f}| Val f1_score: {:4f} | Time: {:.4f}".
            format(epoch, train_acc/(step+1), loss.item(), train_fscore/(step+1), val_acc, val_loss, val_fscore, np.average(dur)))

        wandb.log(dict(zip(["train_acc", "train_loss", "train_fscore", "val_acc", "val_loss", "val_fscore"], [train_acc/(step+1), loss.item(), train_fscore/(step+1), val_acc, val_loss, val_fscore])))
        if epoch%5==0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                pred = model.inference(test_g, device=device, batch_size=config["batch_size"],
                                       num_workers=config["num_workers"])
                test_loss = F.cross_entropy(pred, test_labels)
                test_acc = (torch.argmax(pred, dim=1) == test_labels.long()).float().sum() / len(pred)
                precision, recall, test_fscore, _ = precision_recall_fscore_support(test_labels.detach().cpu().numpy(), torch.argmax(pred, dim=1).detach().cpu().numpy(), average='weighted')
            print("Test Acc: {:.4f} | Test loss: {:.4f}| Test Precision: {:.4f}| Test Recall : {:.4f}| Test Weighted f1 score: {:.4f}|".format(test_acc, test_loss.item(), precision, recall, test_fscore))
            wandb.log(dict(zip(["test_acc", "test_loss", "test_precision", "test_recall", "test_fscore"],
                               [test_acc, test_loss.item(), precision, recall, test_fscore])))
    print()


    print()




if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("-d", "--dataset", type=str, default="toy_01_homo")
    argparser.add_argument('--num-epochs', type=int, default=200)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=5e-4,
                           help="Weight for L2 loss")
    argparser.add_argument("--fan-out", default=[5, 10])
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=1024)
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

    wandb.login()

    print(args)
    main(args)