import dgl
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import math
from torch.nn.parallel import DistributedDataParallel

from musym.benchmark import utils
from musym.models.rgcn_homo.GraphSMOTE.models import GraphSMOTE


def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(dev_id)
    batch_labels = labels[seeds].to(dev_id)
    return batch_inputs, batch_labels


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, labels, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with torch.no_grad():
        pred = model.inference(g, device, batch_size)
    model.train()
    return compute_acc(pred, labels)

def apply_minority_sampling(labels, n_classes, imbalance_ratio=0.3):
    N = int(n_classes/2)
    label_weights = torch.full((len(labels), 1), 1 - imbalance_ratio).squeeze().type(torch.double)
    for i in range(N):
        label_weights = torch.where(labels == i, imbalance_ratio, label_weights)
    return label_weights




def run(result_queue, proc_id, n_gpus, args, devices, data):
    dev_id = devices[proc_id]
    timing_records = []
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    th.cuda.set_device(dev_id)

    n_classes, train_g, _, _ = data

    train_nfeat = train_g.ndata.pop('feat')
    train_labels = train_g.ndata.pop('label')

    train_nfeat = train_nfeat.to(dev_id)
    train_labels = train_labels.to(dev_id)

    in_feats = train_nfeat.shape[1]

    train_mask = train_g.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()

    # Split train_nid
    train_nid = th.split(train_nid, math.ceil(
        len(train_nid) / n_gpus))[proc_id]

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        train_g,
        eids,
        graph_sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        # sampler=sampler
    )

    # Define model and optimizer
    model = GraphSMOTE(in_feats, args.num_hidden, n_classes,
                 args.num_layers, F.relu, args.dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
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

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        result_queue.put(np.array(timing_records))


@utils.benchmark('time', timeout=600)
@utils.skip_if_not_4gpu()
@utils.parametrize('data', ['reddit', 'ogbn-products'])
def track_time(data):
    args = SimpleNamespace(
        num_hidden=16,
        fan_out = "10,25",
        batch_size = 1000,
        lr = 0.003,
        dropout = 0.5,
        num_layers = 2,
        num_workers = 4,
    )

    devices = [0, 1, 2, 3]
    n_gpus = len(devices)
    data = utils.process_data(data)
    g = data[0]
    n_classes = data.num_classes
    train_g = val_g = test_g = g

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
        p = mp.Process(target=utils.thread_wrapped_func(run),
                       args=(result_queue, proc_id, n_gpus, args, devices, data))
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
    print(track_time("cora"))