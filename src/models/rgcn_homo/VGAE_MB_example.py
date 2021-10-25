from models import VGAE
import argparse
import numpy as np
import time
import os, sys
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(os.path.join(SCRIPT_DIR, PACKAGE_PARENT), PACKAGE_PARENT)))

from utils import *

def main(args):

        use_cuda = args.gpu >= 0 and th.cuda.is_available()
        if use_cuda:
                device = th.device('cuda:%d' % args.gpu)
        else:
                device = th.device('cpu')

        train_nfeat = g.ndata.pop('feat')
        in_feats = train_nfeat.shape[1]
        train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = th.nonzero(g.ndata['test_mask'] , as_tuple=True)[0]


        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts=[4, 4])
        train_dataloader = dgl.dataloading.EdgeDataLoader(
                g,
                torch.arange(g.number_of_edges()),
                sampler,
                batch_size = args.batch_size,
                shuffle=True,
                drop_last=False,
                pin_memory=True,
                num_workers=args.num_workers)
        model = GAE(in_feats=in_feats, n_hidden=args.n_hidden, n_layers=args.n_layers, activation=F.relu, dropout=args.dropout)
        model = model.to(device)
        loss_fn =  nn.BCELoss()
        optimizer = th.optim.Adam(model.parameters(), args.lr)

        for epoch in range(args.n_epochs):
                tic = time.time()
                for step, (input_nodes, sub_g, blocks) in enumerate(train_dataloader):
                        batch_inputs = train_nfeat[input_nodes].to(device)
                        blocks = [block.int().to(device) for block in blocks]
                        adj = sub_g.adjacency_matrix().to_dense().to(device)
                        batch_pred = model(blocks, batch_inputs)
                        loss = loss_fn(batch_pred.view(-1), adj.view(-1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print('Epoch {:05d} | Step {:05d} | Loss {:.4f}'.format(
                                epoch, step, loss.item()))
                toc = time.time()
                print('Epoch Time(s): {:.4f}'.format(toc - tic))

if __name__ == '__main__':
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
        argparser.add_argument('--n-epochs', type=int, default=20)
        argparser.add_argument('--n-hidden', type=int, default=8)
        argparser.add_argument('--n-layers', type=int, default=2)
        argparser.add_argument('--fan-out', type=str, default='5,10')
        argparser.add_argument('--batch-size', type=int, default=512)
        argparser.add_argument('--lr', type=float, default=0.001)
        argparser.add_argument('--dropout', type=float, default=0.5)
        argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
        argparser.add_argument('--sample-gpu', action='store_true',
                           help="Perform the sampling process on the GPU. Must have 0 workers.")
        argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
        args = argparser.parse_args()
        main(args)