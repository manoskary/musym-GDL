import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import glob
import os

from musym.models.rgcn_homo.models import SAGE
from musym.models.rgcn_homo.GraphSMOTE.data_utils import load_imbalanced_local

from torchmetrics import Accuracy, Precision, Recall
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from musym.utils import load_and_save


class SAGELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr):
        super().__init__()
        self.save_hyperparameters()
        self.module = SAGE(in_feats, n_hidden, n_classes, n_layers, activation, dropout)
        self.lr = lr

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_precision = Precision()
        self.val_recall = Recall()

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        batch_pred = self.module(mfgs, batch_inputs)
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.train_acc(th.softmax(batch_pred, 1), batch_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        batch_pred = self.module(mfgs, batch_inputs)
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_acc(th.softmax(batch_pred, 1), batch_labels)
        self.val_precision(torch.softmax(batch_pred, 1), batch_labels)
        self.val_recall(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_precision", self.val_precision, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_recall", self.val_recall, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_cpu=False, fan_out=[10, 25],
                 device=th.device('cpu'), batch_size=1000, num_workers=4):
        super().__init__()
        if dataset_name == 'cora':
            g, n_classes = load_imbalanced_local("cora")
        elif dataset_name == 'BlogCatalog':
            g, n_classes = load_imbalanced_local("BlogCatalog")
        elif dataset_name == "cad":
            g, n_classes = load_and_save("cad_basis_homo", os.path.abspath("../data/"))
        else:
            raise ValueError('unknown dataset')

        train_nid = th.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = th.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = th.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in fan_out])

        dataloader_device = th.device('cpu')
        if not data_cpu:
            train_nid = train_nid.to(device)
            val_nid = val_nid.to(device)
            test_nid = test_nid.to(device)
            g = g.formats(['csc'])
            g = g.to(device)
            dataloader_device = device

        self.g = g
        self.train_nid, self.val_nid, self.test_nid = train_nid, val_nid, test_nid
        self.sampler = sampler
        self.device = dataloader_device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_feats = g.ndata['feat'].shape[1]
        self.n_classes = n_classes

    def train_dataloader(self):
        return dgl.dataloading.NodeDataLoader(
            self.g,
            self.train_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return dgl.dataloading.NodeDataLoader(
            self.g,
            self.val_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers)


def evaluate(model, g, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    nfeat = g.ndata['feat']
    labels = g.ndata['label']
    with th.no_grad():
        pred = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    test_acc = Accuracy()
    return test_acc(th.softmax(pred[val_nid], -1), labels[val_nid].to(pred.device))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
                           help="By default the script puts the graph, node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

    datamodule = DataModule(
        args.dataset, args.data_cpu, [int(_) for _ in args.fan_out.split(',')],
        device, args.batch_size, args.num_workers)
    model = SAGELightning(
        datamodule.in_feats, args.num_hidden, datamodule.n_classes, args.num_layers,
        F.relu, args.dropout, args.lr)

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    trainer = Trainer(gpus=[args.gpu] if args.gpu != -1 else None,
                      max_epochs=args.num_epochs,
                      logger=WandbLogger(project="SMOTE", group="SAGE-Lightning", job_type="Cadence-Detection"),
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)

    # Test
    dirs = glob.glob('./lightning_logs/*')
    version = max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])
    logdir = './lightning_logs/version_%d' % version
    print('Evaluating model in', logdir)
    ckpt = glob.glob(os.path.join(logdir, 'checkpoints', '*'))[0]

    model = SAGELightning.load_from_checkpoint(
        checkpoint_path=ckpt, hparams_file=os.path.join(logdir, 'hparams.yaml')).to(device)
    test_acc = evaluate(model, datamodule.g, datamodule.test_nid, device)
    print('Test accuracy:', test_acc)