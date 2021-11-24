import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import math, os
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from musym.benchmark.utils import thread_wrapped_func
from models import GraphSMOTE

from sklearn.metrics import f1_score
from musym.utils import load_and_save



def run(result_queue, proc_id, n_gpus, config, devices, data):
    dev_id = devices[proc_id]
    timing_records = []
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=proc_id)
    torch.cuda.set_device(dev_id)

    n_classes, train_g, val_g, test_g = data

    train_nfeat = train_g.ndata.pop('feat')
    train_labels = train_g.ndata.pop('label')

    train_nfeat = train_nfeat.to(dev_id)
    train_labels = train_labels.to(dev_id)

    in_feats = train_nfeat.shape[1]
    train_eids = torch.arange(train_g.number_of_edges())

    # Split train_nid
    train_eids = torch.split(train_eids, math.ceil(
        len(train_eids) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    graph_sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        train_eids,
        graph_sampler,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config["num_workers"],
    )

    # Define model and optimizer
    model = GraphSMOTE(in_feats, config["num_hidden"], n_classes,
                 config["num_layers"], F.relu, config["dropout"])
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    dur = []
    for epoch in range(config["num_epochs"]):
        model.train()
        t0 = time.time()
        for step, (input_nodes, sub_g, blocks) in enumerate(dataloader):
            if proc_id == 0:
                tic_step = time.time()

            # Load the input features as well as output labels
            blocks = [block.int().to(dev_id) for block in blocks]
            batch_inputs = blocks[0].srcdata['feat']
            batch_labels = blocks[-1].dstdata['label']
            adj = sub_g.adj(ctx=dev_id).to_dense()

            # Compute loss and prediction
            pred, upsampl_lab, embed_loss = model(blocks, batch_inputs, adj, batch_labels)
            loss = loss_fcn(pred, upsampl_lab) + embed_loss * 0.000001
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if proc_id == 0:
                timing_records.append(time.time() - tic_step)

            if step >= 50:
                break
        with torch.no_grad():
            val_labels = val_g.ndata["label"]
            pred = model.inference(val_g, device=dev_id, batch_size=config["batch_size"], num_workers=config["num_workers"])
            val_fscore = f1_score(val_labels.detach().numpy(), torch.argmax(pred, dim=1).detach().numpy(),
                                  average='weighted')
            val_loss = F.cross_entropy(pred, val_labels)
            val_acc = (torch.argmax(pred, dim=1) == val_labels.long()).float().sum() / len(pred)
        print("Epoch {:05d} | Val Acc : {:.4f} | Val CE Loss: {:.4f}| Val f1_score: {:4f}".format(epoch, val_acc, val_loss, val_fscore))

    if n_gpus > 1:
        torch.distributed.barrier()
    if proc_id == 0:
        result_queue.put(np.array(timing_records))



def main(args):
    devices = [0, 1, 2, 3]
    n_gpus = len(devices)

    # --------------- Standarize Configuration ---------------------
    config = args if isinstance(args, dict) else vars(args)

    config["num_layers"] = len(config["fan_out"])
    config["shuffle"] = bool(config["shuffle"])
    config["log"] = False if config["unlog"] else True

    # --------------- Dataset Loading -------------------------
    g, n_classes = load_and_save("cad_basis_homo", config["data_dir"])



    if "train_mask" in g.ndata.keys() and "val_mask" in g.ndata.keys() and "test_mask" in g.ndata.keys():
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
    else:
        # --------------- Manually build train masks ------------------------
        train_mask = val_mask = test_mask = torch.zeros(g.num_nodes())

        rand_inds = torch.tensor(range(g.num_nodes()))
        np.random.shuffle(rand_inds)
        train_inds = torch.tensor(rand_inds[:int(0.7 * g.num_nodes())])
        val_inds = torch.tensor(rand_inds[int(0.7 * g.num_nodes()): int(0.8 * g.num_nodes())])
        test_inds = torch.tensor(rand_inds[int(0.8 * g.num_nodes()):])

        train_mask[train_inds] = 1
        val_mask[val_inds] = 1
        test_mask[test_inds] = 1

    # Hack to track node idx in dataloader's subgraphs.
    train_g = g.subgraph(torch.nonzero(train_mask)[:, 0])
    train_g.ndata['idx'] = torch.tensor(range(train_g.number_of_nodes()))

    # Validation and Testing
    val_g = g.subgraph(torch.nonzero(val_mask)[:, 0])
    test_g = g.subgraph(torch.nonzero(test_mask)[:, 0])


    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    # Pack data
    data = n_classes, train_g, val_g, test_g

    result_queue = mp.Queue()
    procs = []
    for proc_id in range(n_gpus):
        p = mp.Process(target=thread_wrapped_func(run),
                       args=(result_queue, proc_id, n_gpus, config, devices, data))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    time_records = result_queue.get(block=False)
    num_exclude = 10 # exclude first 10 iterations
    if len(time_records) < 15:
        # exclude less if less records
        num_exclude = int(len(time_records)*0.3)
    return np.mean(time_records[num_exclude:])



if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='GraphSAGE')
    argparser.add_argument('--gpu', type=int, default=-1,
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
    argparser.add_argument("--batch-size", type=int, default=2048)
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