import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from musym.models.cad.models import FullGraphSMOTE
from torchmetrics import Accuracy, AUROC, F1
from pytorch_lightning import LightningDataModule, LightningModule


class CadGraphDataset(Dataset):
    """Cedence List of Graphs dataset."""

    def __init__(self, g, nids, piece_idx, node_features, labels):
        idx = [torch.nonzero(piece_idx == scidx, as_tuple=True)[0] for scidx in torch.unique(piece_idx[nids])]
        self.graphs = [g.subgraph(ids).adj() for ids in idx]
        self.node_features = [node_features[ids] for ids in idx]
        self.labels = [labels[ids].squeeze() for ids in idx]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx[0]
        sample = (self.graphs[idx].to_dense(), self.node_features[idx], self.labels[idx])
        return sample


class FullGraphCadLightning(LightningModule):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 lr,
                 loss_weight=0.0001,
                 weight_decay=5e-4,
                 adj_thresh=0.01
        ):
        super(FullGraphCadLightning, self).__init__()
        self.save_hyperparameters()
        self.loss_weight = loss_weight
        self.module = FullGraphSMOTE(in_feats, n_hidden, n_classes, n_layers, activation, dropout, adj_thresh=adj_thresh)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_fscore = F1(num_classes=n_classes, average="macro")
        self.train_auroc = AUROC(num_classes=n_classes, average="macro")
        self.val_fscore = F1(num_classes=n_classes, average="macro")
        self.val_auroc = AUROC(num_classes=n_classes, average="macro")
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.n_classes = n_classes

    def training_step(self, batch, batch_idx):
        batch_adj, batch_inputs, batch_labels = batch
        batch_adj = batch_adj.squeeze(0).to(self.device)
        batch_inputs = batch_inputs.squeeze(0).to(self.device)
        batch_labels = batch_labels.squeeze(0).to(self.device)
        batch_pred, upsampl_lab, embed_loss = self.module(batch_adj, batch_inputs, batch_labels)
        loss = self.train_loss(batch_pred, upsampl_lab) + embed_loss * self.loss_weight
        batch_pred = F.softmax(batch_pred, dim=1)
        self.train_acc(batch_pred, upsampl_lab)
        self.train_fscore(batch_pred[:len(batch_labels)], batch_labels)
        self.train_auroc(batch_pred[:len(batch_labels)], batch_labels)
        self.log('train_loss', loss.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_fscore", self.train_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_adj, batch_inputs, batch_labels = batch
        batch_adj = batch_adj.squeeze(0).to(self.device)
        batch_inputs = batch_inputs.squeeze(0).to(self.device)
        batch_labels = batch_labels.squeeze(0).to(self.device)
        batch_pred, prev_encs = self.module.encoder(batch_adj, batch_inputs)
        pred_adj = F.hardshrink(self.module.decoder(batch_pred, batch_pred), lambd=self.module.adj_thresh)
        batch_pred = self.module.classifier(pred_adj, batch_pred, batch_pred)
        self.val_acc(batch_pred, batch_labels)
        loss = F.cross_entropy(batch_pred, batch_labels)
        batch_pred = F.softmax(batch_pred, dim=1)
        self.val_fscore(batch_pred, batch_labels)
        self.val_auroc(batch_pred, batch_labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_fscore", self.val_fscore, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val_auroc", self.val_auroc, on_step=True, on_epoch=True, sync_dist=True)

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


class FullGraphDataModule(LightningDataModule):
    def __init__(self, g, node_features, labels, piece_idx, in_feats, train_nid, val_nid, data_cpu=False, device=torch.device('cpu'), num_workers=4, use_ddp=True):
        super().__init__()
        self.train_graphs = CadGraphDataset(g, train_nid, piece_idx, node_features, labels)
        self.val_graphs = CadGraphDataset(g, val_nid, piece_idx, node_features, labels)

        dataloader_device = torch.device('cpu')
        if not data_cpu and num_workers==0:
            dataloader_device = device

        self.device = dataloader_device
        self.num_workers = num_workers
        self.in_feats = in_feats
        self.use_ddp = use_ddp

    def train_dataloader(self):
        return DataLoader(
            self.train_graphs,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(
            self.val_graphs,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers)




