import dgl
import torch
import torch.nn.functional as F
from musym.models.cad.models import GraphSMOTE
from torchmetrics import Accuracy, F1, AUROC
from pytorch_lightning import LightningDataModule, LightningModule


class MyLoader(dgl.dataloading.NodeDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch):
        try:
            if self.use_scalar_batcher:
                self.scalar_batcher.set_epoch(epoch)
            else:
                self.dist_sampler.set_epoch(epoch)
        except AttributeError:
            pass


class CadModelLightning(LightningModule):
    def __init__(self,
                 node_features,
                 labels,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr,
                 loss_weight = 0.0001,
                 ext_mode=None,
                 weight_decay=5e-4,
        ):
        super(CadModelLightning, self).__init__()
        self.save_hyperparameters()
        self.loss_weight = loss_weight
        self.module = GraphSMOTE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, ext_mode=ext_mode)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_fscore = F1(n_classes, average="macro")
        self.train_auroc = AUROC(num_classes=n_classes, average="macro")
        self.val_fscore = F1(n_classes, average="macro")
        self.val_auroc = AUROC(num_classes=n_classes, average="macro")
        self.test_fscore = F1(n_classes, average="macro")
        self.test_auroc = AUROC(num_classes=n_classes, average="macro")
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.node_features = node_features
        self.labels = labels

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = self.node_features[input_nodes].to(self.device)
        batch_labels = self.labels[output_nodes].to(self.device)
        # adj = mfgs[-1].adj().to_dense()[:len(batch_labels), :len(batch_labels)].to(self.device)
        adj = mfgs[-1].adj().to(self.device)
        batch_pred, upsampl_lab, embed_loss = self.module(mfgs, batch_inputs, adj, batch_labels)
        loss = self.train_loss(batch_pred, upsampl_lab) + embed_loss * self.loss_weight
        self.train_acc(torch.softmax(batch_pred, 1), upsampl_lab)
        self.train_fscore(torch.softmax(batch_pred[:len(batch_labels)], 1), batch_labels)
        self.train_auroc(torch.softmax(batch_pred[:len(batch_labels)], 1), batch_labels)
        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_fscore", self.train_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_batch_size", len(output_nodes), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = self.node_features[input_nodes].to(self.device)
        batch_labels = self.labels[output_nodes].to(self.device)
        batch_pred, prev_encs = self.module.encoder(mfgs, batch_inputs)
        pred_adj = self.module.decoder(batch_pred, prev_encs)
        # if pred_adj.get_device() >= 0 :
        #     pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
        # else:
        #     pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
        batch_pred = self.module.classifier(pred_adj, batch_pred, prev_encs)
        self.val_acc(torch.softmax(batch_pred, 1), batch_labels)
        loss = F.cross_entropy(batch_pred, batch_labels)
        self.val_fscore(torch.softmax(batch_pred, 1), batch_labels)
        self.val_auroc(torch.softmax(batch_pred, 1), batch_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_fscore", self.val_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = self.node_features[input_nodes].to(self.device)
        batch_labels = self.labels[output_nodes].to(self.device)
        batch_pred, prev_encs = self.module.encoder(mfgs, batch_inputs)
        pred_adj = self.module.decoder(batch_pred, prev_encs)
        # if pred_adj.get_device() >= 0 :
        #     pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype).to(batch_pred.get_device()))
        # else:
        #     pred_adj = torch.where(pred_adj >= 0.5, pred_adj, torch.tensor(0, dtype=pred_adj.dtype))
        batch_pred = self.module.classifier(pred_adj, batch_pred, prev_encs)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_fscore"
            }
        }


class CadDataModule(LightningDataModule):
    def __init__(self, g, n_classes,  in_feats, train_nid, val_nid=[], test_nid=[], data_cpu=False, fan_out=[10, 25],
                 device=torch.device('cpu'), batch_size=1000, num_workers=4, init_weights=False, use_ddp=True):
        super().__init__()

        if init_weights:
            w = torch.empty(g.num_edges())
            torch.nn.init.uniform_(w)
            g.edata["w"] = w

        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in fan_out])

        dataloader_device = torch.device('cpu')
        if not data_cpu and num_workers==0:
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
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.use_ddp = use_ddp
        self.use_scalar_batcher = False if self.use_ddp else True

    def train_dataloader(self):
        return MyLoader(
            self.g,
            self.train_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_ddp=self.use_ddp,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return MyLoader(
            self.g,
            self.val_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_ddp=self.use_ddp,
            num_workers=self.num_workers)

    def predict_dataloader(self):
        return MyLoader(
            self.g,
            self.test_nid,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_ddp=self.use_ddp,
            num_workers=self.num_workers)
