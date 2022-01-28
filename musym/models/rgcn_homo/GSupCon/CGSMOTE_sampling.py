import os
import torch
import torch.nn as nn
import dgl
import tqdm
from models import ContrastiveGraphSMOTE, SageClassifier
from musym.utils import load_and_save
from pytorch_metric_learning.losses import SupConLoss
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.functional import fbeta

def run(proc_id, n_gpus, args, devices, data):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip="140.78.124.137", # RK5
            master_port='12345')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    torch.cuda.set_device(dev_id)

    #--------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    config["num_layers"] = len(config["fan_out"])
    config["shuffle"] = bool(config["shuffle"])
    config["log"] = False if config["unlog"] else True

    # --------------- Dataset Loading -------------------------
    g, n_classes = data


    # Hack to track node idx in dataloader's subgraphs.
    labels = g.ndata.pop('label')
    train_nids = torch.nonzero(g.ndata.pop('train_mask'), as_tuple=True)[0]
    node_features = g.ndata.pop('feat')


    # Validation and Testing
    val_nids = torch.nonzero(g.ndata.pop('val_mask'), as_tuple=True)[0]
    test_nids = torch.nonzero(g.ndata.pop('test_mask'), as_tuple=True)[0]
    # check cuda
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:%d' % dev_id if use_cuda else 'cpu')
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
        use_ddp=True,
        drop_last=False,
        num_workers=config["num_workers"],
    )

    model = ContrastiveGraphSMOTE(in_feats, n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=config["num_layers"])
    classifier = SageClassifier(in_feats=config["num_hidden"], n_hidden=config["num_hidden"], n_classes=n_classes, n_layers=1)
    if use_cuda:
        model = model.to(dev_id)
        classifier = classifier.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
        classifier = DistributedDataParallel(classifier, device_ids=[dev_id], output_device=dev_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = SupConLoss()
    cl_criterion = nn.CrossEntropyLoss()
    cl_optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # training loop
    print("start training...")

    for epoch in range(config["num_epochs"]):
        model.train()
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True, desc="Pretraining Epoch %i" % epoch)):
            batch_inputs = node_features[input_nodes].to(device)
            batch_labels = labels[output_nodes].to(device)
            adj = mfgs[-1].adj().to_dense()[:len(batch_labels), :len(batch_labels)].to(device)
            # Prediction of the Conctrastive model
            pred, upsampl_lab, embed_loss = model(mfgs, batch_inputs, adj, batch_labels)
            loss = criterion(pred, upsampl_lab) + embed_loss * 0.000001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if n_gpus > 1:
            torch.distributed.barrier()

    model.eval()
    classifier.train()
    for epoch in range(config["num_epochs"]):
        for step, (input_nodes, output_nodes, mfgs) in enumerate(tqdm.tqdm(train_dataloader, position=0, leave=True, desc="Classifier Epoch %i" % epoch)):
            with torch.no_grad():
                batch_inputs = node_features[input_nodes].to(device)
                batch_labels = labels[output_nodes].to(device)
                # Encoding of the Contrastive model
                batch_pred, prev_encs = model.module.encoder(mfgs, batch_inputs)
                pred_adj = model.module.decoder(batch_pred, prev_encs)
            # Classifier Prediction
            batch_pred = classifier(pred_adj, batch_pred, prev_encs)
            acc = (torch.argmax(batch_pred, dim=1) == batch_labels).float().sum() / len(pred)
            f1 = fbeta(torch.argmax(batch_pred, dim=1), batch_labels, average="macro", num_classes=n_classes)
            cl_loss = cl_criterion(batch_pred, batch_labels)
            cl_optimizer.zero_grad()
            cl_loss.backward()
            cl_optimizer.step()
        if n_gpus > 1:
            torch.distributed.barrier()
        if dev_id == 0:
            print("Step {:04d} | Accuracy : {:.4f} | F1 Macro {:.4f} | Loss : {:.4f} |".format(step, acc, f1, cl_loss))

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g,
        test_nids,
        sampler,
        device=device,
        shuffle=config["shuffle"],
        batch_size=config["batch_size"],
        drop_last=False,
        use_ddp=True,
        num_workers=config["num_workers"],
    )

    acc = 0
    classifier.eval()
    with torch.no_grad():
        for step, (input_nodes, output_nodes, mfgs) in enumerate(
                tqdm.tqdm(test_dataloader, position=0, leave=True, desc="Epoch %i" % epoch)):
            batch_inputs = node_features[input_nodes].to(device)
            batch_labels = labels[output_nodes].to(device)
            # Encoding of the Conctrastive model
            batch_pred, prev_encs = model.module.encoder(mfgs, batch_inputs)
            pred_adj = model.module.decoder(batch_pred, prev_encs)
            if pred_adj.get_device() >= 0:
                pred_adj = torch.where(pred_adj >= 0.5, pred_adj,
                                       torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
            else:
                pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
            # Classifier Prediction
            batch_pred = classifier(pred_adj, batch_pred, prev_encs)
            acc += (torch.argmax(batch_pred, dim=1) == batch_labels).float().sum() / len(pred)
        if n_gpus > 1:
            torch.distributed.barrier()
    acc = acc/(step+1)
    print("Test accuracy: {:04f}".format(acc))



if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
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
    argparser.add_argument("--data-dir", type=str, default=os.path.abspath("../data/"))
    argparser.add_argument("--unlog", action="store_true", help="Unbinds wandb.")
    args = argparser.parse_args()
    print(args)
    # --------------- Dataset Loading -------------------------
    g, n_classes = load_and_save("cad_basis_homo", args.data_dir)
    # g, n_classes = load_imbalanced_local("BlogCatalog")
    g.create_formats_()
    data = g, n_classes

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
