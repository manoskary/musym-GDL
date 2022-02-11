import dgl
import torch
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import glob
import os

from musym.models.rgcn_homo.GraphSMOTE.models import GraphSMOTE
from musym.models.rgcn_homo.GraphSMOTE.data_utils import load_imbalanced_local

from torchmetrics import Accuracy, F1, AUROC
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger
from musym.utils import load_and_save



class MyLoader(dgl.dataloading.NodeDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch):
        if self.use_scalar_batcher:
            self.scalar_batcher.set_epoch(epoch)
        else:
            self.dist_sampler.set_epoch(epoch)


class GraphSMOTELightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr,
                 loss_weight = 0.0001,
                 ext_mode=None
        ):
        super(GraphSMOTELightning, self).__init__()
        self.save_hyperparameters()
        self.loss_weight = loss_weight
        self.module = GraphSMOTE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, ext_mode=ext_mode)
        self.lr = lr
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.val_fscore = F1(n_classes, average="macro")
        self.val_auroc = AUROC(num_classes=n_classes, average="macro")
        self.train_loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        adj = mfgs[-1].adj().to_dense()[:len(batch_labels), :len(batch_labels)].to(self.device)
        batch_pred, upsampl_lab, embed_loss = self.module(mfgs, batch_inputs, adj, batch_labels)
        loss = self.train_loss(batch_pred, upsampl_lab) + embed_loss * self.loss_weight
        self.train_acc(torch.softmax(batch_pred, 1), upsampl_lab)
        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        batch_pred, prev_encs = self.module.encoder(mfgs, batch_inputs)
        pred_adj = self.module.decoder(batch_pred, prev_encs)
        if pred_adj.get_device() >= 0 :
            pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
        else:
            pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
        batch_pred = self.module.classifier(pred_adj, batch_pred, prev_encs)
        # loss = self.cross_entropy_loss(batch_pred, batch_labels)
        self.val_acc(torch.softmax(batch_pred, 1), batch_labels)
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_fscore(torch.softmax(batch_pred, 1), batch_labels)
        self.val_auroc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_fscore", self.val_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = mfgs[0].srcdata['feat']
        batch_labels = mfgs[-1].dstdata['label']
        batch_pred, prev_encs = self.module.encoder(mfgs, batch_inputs)
        pred_adj = self.module.decoder(batch_pred, prev_encs)
        if pred_adj.get_device() >= 0 :
            pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
        else:
            pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
        batch_pred = self.module.classifier(pred_adj, batch_pred, prev_encs)
        # loss = self.cross_entropy_loss(batch_pred, batch_labels)
        self.test_acc(torch.softmax(batch_pred, 1), batch_labels)
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.test_fscore(torch.softmax(batch_pred, 1), batch_labels)
        self.test_auroc(torch.softmax(batch_pred, 1), batch_labels)
        output = {
            'test_loss': loss,
            'test_acc': self.test_acc,
            "test_fscore": self.test_fscore,
            "test_auroc": self.test_auroc
        }
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_fscore"
            }
        }


class DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_cpu=False, fan_out=[10, 25],
                 device=torch.device('cpu'), batch_size=1000, num_workers=4, init_weights=False):
        super().__init__()
        if dataset_name == 'cora':
            g, n_classes = load_imbalanced_local("cora")
        elif dataset_name == 'BlogCatalog':
            g, n_classes = load_imbalanced_local("BlogCatalog")
        elif dataset_name == "reddit":
            g, n_classes = load_imbalanced_local("reddit")
        elif dataset_name == "cad":
            g, n_classes = load_and_save("cad_basis_homo", os.path.abspath("../data/"))
        else:
            raise ValueError('unknown dataset')

        train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
        val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
        test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

        if init_weights:
            w = torch.empty(g.num_edges())
            torch.nn.init.uniform_(w)
            g.edata["w"] = w

        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in fan_out])

        dataloader_device = torch.device('cpu')
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
        return MyLoader(
            self.g,
            self.train_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            # use_ddp=True,
            num_workers=0)

    def val_dataloader(self):
        return MyLoader(
            self.g,
            self.val_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0)

    def test_dataloader(self):
        return MyLoader(
            self.g,
            self.test_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0)

def evaluate(model, g, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    nfeat = g.ndata['feat']
    labels = g.ndata['label']
    with torch.no_grad():
        pred = model.module.inference(g, nfeat, labels, device, args.batch_size, args.num_workers)
    model.train()
    test_acc = Accuracy()
    return test_acc(torch.softmax(pred, 1), labels.to(pred.device))


# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'
#
#     # initialize the process group
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)
#
# def cleanup():
#     dist.destroy_process_group()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpu', type=int, default=0,
                           help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='cad')
    argparser.add_argument('--num-epochs', type=int, default=50)
    argparser.add_argument('--num-hidden', type=int, default=32)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--loss-weight', type=float, default=0.0001,
                           help="regularization loss weight.")
    argparser.add_argument('--num-workers', type=int, default=4,
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
        device = torch.device('cuda:%d' % args.gpu)
    else:
        device = torch.device('cpu')

    datamodule = DataModule(
        args.dataset, args.data_cpu, [int(_) for _ in args.fan_out.split(',')],
        device, args.batch_size, args.num_workers)
    model = GraphSMOTELightning(
        datamodule.in_feats, args.num_hidden, datamodule.n_classes, args.num_layers,
        F.relu, args.dropout, args.lr, args.loss_weight)

    # Train
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1)
    trainer = Trainer(gpus=[args.gpu] if args.gpu != -1 else None,
                      max_epochs=args.num_epochs,
                      logger=WandbLogger(project="SMOTE", group="GraphSMOTE-Lightning", job_type="Cadence-Detection"),
                      callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=datamodule)

    # Test
    dirs = glob.glob('./lightning_logs/*')
    version = max([int(os.path.split(x)[-1].split('_')[-1]) for x in dirs])
    logdir = './lightning_logs/version_%d' % version
    print('Evaluating model in', logdir)
    ckpt = glob.glob(os.path.join(logdir, 'checkpoints', '*'))[0]

    model = GraphSMOTELightning.load_from_checkpoint(
        checkpoint_path=ckpt, hparams_file=os.path.join(logdir, 'hparams.yaml')).to(device)
    test_acc = evaluate(model, datamodule.test_g, device)
    print('Test accuracy:', test_acc)