import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import tqdm
from sklearn.metrics import f1_score
from musym.models.cad.models import GraphSMOTE, Node2vecModel
from musym.utils import load_and_save
from torchmetrics.functional import f1
from ray.tune import report
from hmmlearn import hmm
import socket
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def extract_ip():
    st = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        st.connect(('10.255.255.255', 1))
        IP = st.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        st.close()
    return IP

def post_evaluate(pm, X, target):
    over_acc = 0
    over_f1 = 0
    for i, seq in enumerate(X):
        y_pred = pm.predict(seq)
        acc = np.equal(target[i], y_pred).astype(float).sum() / len(y_pred)
        over_acc += acc
        fscore = f1_score(target[i], y_pred, average="macro")
        over_f1 += fscore
        print("Post-Process Model: Accuracy {:.4f} | F score {:.4f} |".format(acc, fscore))
    over_acc = over_acc / (i+1)
    over_f1 = over_f1 / (i+1)
    print("Mean Post-Process Model: Accuracy {:.4f} | F score {:.4f} |".format(over_acc, over_f1))


def postprocess(X_train, y_train, X_val=None, y_val=None):
    # ------------- HMM Learn -------------------------
    pm = hmm.GaussianHMM(n_components=2, covariance_type="diag", init_params="cm", params="cmts", n_iter=100)
    pm.startprob_ = np.array([1.0, 0.0])
    pm.transmat_ = np.array([[0.95, 0.05],
                              [1.00, 0.00]])

    pm.fit(np.concatenate(X_train), [len(x) for x in X_train])
    print("Post-Training Evaluation :")
    post_evaluate(pm, X_train, y_train)
    if X_val is not None:
        print("Post-Validation Evaluation :")
        post_evaluate(pm, X_val, y_val)
    return pm


    # Validation Post-Process Loop
    over_acc = 0
    over_f1 = 0
    for i, seq in enumerate(X_val):
        y_pred = pm.predict(seq)
        acc = np.equal(y_val[i], y_pred).astype(float).sum() / len(y_pred)
        over_acc += acc
        fscore = f1_score(y_val[i], y_pred, average="macro")
        over_f1 += fscore
        print("Post-Process Validation: Accuracy {:.4f} | F score {:.4f} |".format(acc, fscore))
    over_acc = over_acc / (i + 1)
    over_f1 = over_f1 / (i + 1)
    print("Mean Post-Process Validation: Accuracy {:.4f} | F score {:.4f} |".format(over_acc, over_f1))


def to_sequences(labels, preds, idx, score_duration, piece_idx, onsets):
    seqs = list()
    trues = list()
    # Make args same dimensions as preds
    piece_idx = piece_idx[idx]
    labels = labels[idx]
    score_duration = score_duration[idx]
    onsets = onsets[idx]
    o = torch.zeros(len(onsets))
    o[torch.nonzero(onsets == 0, as_tuple=True)[0]] = 1.00
    # Start Building Sequence per piece name.
    for name in torch.unique(piece_idx):
        # Gather on non-augmented Pieces
        if name != 0:
            durs = score_duration[piece_idx == name]
            is_onset = onsets[piece_idx == name]
            X = preds[piece_idx == name]
            y = labels[piece_idx == name]
            sorted_durs, resorted_idx = torch.sort(durs)
            X = X[resorted_idx]
            y = y[resorted_idx]
            is_onset = is_onset[resorted_idx]
            new_X = []
            new_y = []
            # group by onset
            for udur in torch.unique(sorted_durs):
                x = torch.cat((X[sorted_durs == udur], is_onset[sorted_durs == udur].unsqueeze(1)), dim=1)
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


def train_step(epoch, model, train_dataloader, node_features, labels, device, optimizer, criterion, dataloader_device, config, n_classes):
    model.train()
    acc = list()
    f1_score = 0
    for step, (input_nodes, output_nodes, mfgs) in enumerate(
            tqdm.tqdm(train_dataloader, position=0, leave=True, desc="Training Epoch %i" % epoch)):
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
        f1_score += f1(torch.argmax(pred[:len(batch_labels)], dim=1), batch_labels, average="macro",
                       num_classes=n_classes).item()
    f1_score = f1_score / (step + 1)
    acc_score = torch.cat(acc).float().sum() / len(labels)
    return loss, acc_score, f1_score


def evaluate(model, dataloader, nids, node_features, labels, device):
    model.eval()
    prediction = model.inference(dataloader, node_features, labels, device)
    loss = F.cross_entropy(prediction[nids].cpu(), labels[nids])
    acc = torch.eq(labels[nids], torch.argmax(prediction[nids].cpu(), dim=1)).float().sum() / len(labels[nids])
    fscore = f1(prediction[nids].cpu(), labels[nids], average="macro", num_classes=2)
    return prediction, loss, acc, fscore


def return_dataloaders(g, train_nids, val_nids, test_nids, sampler, device, shuffle, batch_size, drop_last, num_workers):
    train_dataloader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler, device=device, shuffle=shuffle,
                                                      batch_size=batch_size, drop_last=False, num_workers=num_workers)
    val_dataloader = dgl.dataloading.NodeDataLoader(g, val_nids, sampler, device=device, shuffle=shuffle,
                                                    batch_size=batch_size, drop_last=drop_last, num_workers=num_workers)

    test_dataloader = dgl.dataloading.NodeDataLoader(g, test_nids, sampler, device=device, shuffle=shuffle,
                                                    batch_size=batch_size, drop_last=drop_last, num_workers=num_workers)
    return train_dataloader, val_dataloader, test_dataloader


def train(rank, n_gpus, config, data):

    g, model, train_dataloader, sampler, \
    val_dataloader, node_features, labels, device, \
    optimizer, scheduler, criterion, dataloader_device, config, \
    n_classes, train_nids, val_nids = data

    if n_gpus > 1:
        model = model.to(rank)
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
        device = dataloader_device = torch.device('cuda:' + str(rank))

        train_dataloader = dgl.dataloading.NodeDataLoader(g, train_nids, sampler, device=device, shuffle=True,
                                                          batch_size=config["batch_size"], drop_last=False,
                                                          num_workers=0, use_ddp=True)
    else:
        model.to(device)

    print("start training...")

    for epoch in range(config["num_epochs"]):
        loss, acc_score, f1_score = train_step(epoch, model, train_dataloader, node_features, labels, device,
                                               optimizer, criterion, dataloader_device, config, n_classes)
        if config["tune"]:
            report(epoch, loss.item())
        print("Epoch {:04d} | Loss {:.04f} | Acc {:.04f} | F score {:.04f} |".format(epoch, loss, acc_score,
                                                                                     f1_score))
        if n_gpus > 1:
            torch.distributed.barrier()
            if rank == 0 :
                eval_model = model.module
                eval_model.to(rank)
                prediction, loss, acc_score, f1_score = evaluate(eval_model, val_dataloader, val_nids, node_features, labels, rank)
        else:
            eval_model = model
            prediction, loss, acc_score, f1_score = evaluate(eval_model, val_dataloader, val_nids, node_features, labels, device)

        scheduler.step(f1_score)
        print("Validation Score : Loss {:.04f} | Accuracy {:.4f} | F score {:.4f} |".format(loss.item(), acc_score,
                                                                                            f1_score))
    return eval_model

def init_process(rank, world_size, fn, config, data, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = str(extract_ip())
    os.environ['MASTER_PORT'] = str(1234)
    torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
    model = fn(rank, world_size, config, data)
    return model


def main(args):
    """
    Main Call for Node Classification with Node2Vec + GraphSMOTE + DBNN.
    """
    #--------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)


    fanouts = [int(fanout) for fanout in config["fan_out"].split(',')]
    config["num_layers"] = len(fanouts)
    config["shuffle"] = bool(config["shuffle"])

    # --------------- Dataset Loading -------------------------
    g, n_classes = load_and_save(config["dataset"], config["data_dir"])
    g = dgl.add_self_loop(dgl.add_reverse_edges(g))
    # training defs
    labels = g.ndata.pop('label')
    train_nids = torch.nonzero(g.ndata.pop('train_mask'), as_tuple=True)[0]
    node_features = g.ndata.pop('feat')
    piece_idx = g.ndata.pop("score_name")
    onsets = node_features[:, 0]
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
    emb_path = os.path.join(config["data_dir"], config["dataset"], "node_emb.pt")
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


    if use_cuda and config["num_workers"]==0:
        train_nids = train_nids.to(device)
        test_nids = test_nids.to(device)
        g = g.formats(['csc'])
        g = g.to(device)
        dataloader_device = device




    # dataloader
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts)
    # The edge dataloader returns a subgraph but iterates on the number of edges.

    train_dataloader, val_dataloader, test_dataloader = return_dataloaders(
        g=g, train_nids=train_nids, val_nids=val_nids, test_nids=test_nids, sampler=sampler,
        device=dataloader_device, shuffle=config["shuffle"], batch_size=config["batch_size"],
        drop_last=False, num_workers=config["num_workers"]
    )

    model_path = os.path.join(config["data_dir"], config["dataset"], "model_sd.pt")
    model = GraphSMOTE(in_feats, n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=config["num_layers"],
                       ext_mode=config["ext_mode"])
    print("Model Trainable Parameters : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = nn.CrossEntropyLoss()


    if config["load_model"]:
        model.load_state_dict(torch.load(model_path))
    else:
        # training loop
        # --------------------- Init WANDB ---------------------------------
        # wandb.init(
        #     project="Cadence Detection",
        #     group=config["dataset"],
        #     job_type="GraphSMOTE-bs{}-l{}x{}".format(config["batch_size"], config["num_layers"], config["num_hidden"]),
        #     config=config,
        #     reinit=True,
        #     settings=wandb.Settings(start_method='fork')
        # )
        # wandb.watch(model, log_freq=1000)
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            g.create_formats_()
        processes = []
        batch_data = g, model, train_dataloader, sampler, val_dataloader, node_features, labels, \
               device, optimizer, scheduler, criterion, dataloader_device, config, n_classes, train_nids, val_nids

        mp.set_start_method("spawn", force=True)
        for rank in range(n_gpus):
            p = mp.Process(target=init_process, args=(rank, n_gpus, train, config, batch_data))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        torch.save(model.state_dict(), model_path)


    model.eval()
    train_prediction = model.inference(train_dataloader, node_features, labels, device)
    # pred_path = os.path.join(config["data_dir"], "cad_basis_homo", "preds.pt")
    # posttrain_label_path = os.path.join(config["data_dir"], "cad_basis_homo", "post_train_labels.pt")
    # torch.save(train_prediction.detach().cpu(), pred_path)
    # torch.save(labels[train_nids].detach().cpu(), posttrain_label_path)
    # TODO needs to address post-processing using Dynamic Bayesian Model.
    X_train = train_prediction[train_nids].detach().cpu()
    if config["eval"]:
        test_prediction, loss, acc_score, f1_score = evaluate(model, test_dataloader, test_nids, node_features, labels, device)
        print("Test Score : Loss {:.04f} | Accuracy {:.4f} | F score {:.4f} |".format(loss.item(), acc_score, f1_score))
        Χ_test = test_prediction[test_nids].detach().cpu()
        train_set = to_sequences(labels, X_train, train_nids, score_duration, piece_idx, onsets)
        test_set = to_sequences(labels, Χ_test, test_nids, score_duration, piece_idx, onsets)
        return train_set, test_set
    else:
        print(X_train.shape, y_train.shape)
        train_set = to_sequences(labels, X_train, train_nids, score_duration, piece_idx, onsets)
        return train_set




if __name__ == '__main__':
    import argparse
    import pickle
    argparser = argparse.ArgumentParser(description='Cadence Learning GraphSMOTE')
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument("--dataset", type=str, default="cad_basis_homo")
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
    argparser.add_argument("--fan-out", type=str, default='5, 10')
    argparser.add_argument('--shuffle', type=int, default=True)
    argparser.add_argument("--batch-size", type=int, default=2048)
    argparser.add_argument("--num-workers", type=int, default=10)
    argparser.add_argument("--tune", type=bool, default=False)
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../../rgcn_homo/data/"))
    argparser.add_argument("--preprocess", action="store_true", help="Train and store graph embedding")
    argparser.add_argument("--postprocess", action="store_true", help="Train and DBNN")
    argparser.add_argument("--load-model", action="store_true", help="Load pretrained model.")
    argparser.add_argument("--eval", action="store_true", help="Preview Results on Validation set.")
    args = argparser.parse_args()



    print(args)
    if not args.postprocess:
        (X_train, y_train), (X_val, y_val) = main(args)
        pred_path = os.path.join(args.data_dir, args.dataset, "preds.pkl")
        posttrain_label_path = os.path.join(args.data_dir, args.dataset, "post_train_labels.pkl")
        with open(pred_path, "wb") as f:
            pickle.dump(X_train, f)
        with open(posttrain_label_path, "wb") as f:
            pickle.dump(y_train, f)
    else :
        pred_path = os.path.join(args.data_dir, args.dataset, "preds.pkl")
        posttrain_label_path = os.path.join(args.data_dir, config["dataset"], "post_train_labels.pkl")
        # X_train, y_train = torch.load(pred_path).numpy(), torch.load(posttrain_label_path).numpy()
        with open(pred_path, "rb") as f:
            X_train = pickle.load(f)
        with open(posttrain_label_path, "rb") as f:
            y_train = pickle.load(f)

        print("Start Post-processing...")

    postmodel = postprocess(X_train, y_train, X_val, y_val)





