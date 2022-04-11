import os.path as osp
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
import dgl
import torch
from musym.utils import load_and_save
from torch.nn import functional as F
from torchmetrics.classification.accuracy import Accuracy
from pytorch_lightning import LightningDataModule
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from musym.models.cad.models.cad_lightning import MyLoader

#############################################################################################
#                           KFold Loop / Cross Validation Example                           #
# This example demonstrates how to leverage Lightning Loop Customization introduced in v1.5 #
# Learn more about the loop structure from the documentation:                               #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html                  #
#############################################################################################


#############################################################################################
#                           Step 1 / 5: Define KFold DataModule API                         #
# Our KFold DataModule requires to implement the `setup_folds` and `setup_fold_index`       #
# methods.                                                                                  #
#############################################################################################


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self, num_folds: int) -> None:
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        pass


#############################################################################################
#                           Step 2 / 5: Implement the KFoldDataModule                       #
# The `KFoldDataModule` will take a train and test dataset.                                 #
# On `setup_folds`, folds will be created depending on the provided argument `num_folds`    #
# Our `setup_fold_index`, the provided train dataset will be split accordingly to        #
# the current fold split.                                                                   #
#############################################################################################


@dataclass
class CadLoocvDataModule(BaseKFoldDataModule):

    def __init__(self, dataset, data_dir, data_cpu=False, fan_out=[10, 25], device=torch.device('cpu'),
                 batch_size=1000, num_workers=4, init_weights=False, use_ddp=True):
        super(CadLoocvDataModule, self).__init__()
        self.dataset_name = dataset
        self.data_dir = data_dir
        self.data_cpu = data_cpu
        self.fan_out = fan_out
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.init_weights = init_weights
        self.use_ddp = use_ddp
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler([int(_) for _ in fan_out])
        self.use_scalar_batcher = False if self.use_ddp else True
        # self.prepare_data()


    def prepare_data(self) -> None:
        # download the data.
        self.g, self.n_classes = load_and_save(self.dataset_name, self.data_dir)
        self.in_feats = self.g.ndata["feat"].shape[1]

    def setup(self, stage: Optional[str] = None) -> None:
        # load the data
        self.g = dgl.add_self_loop(dgl.add_reverse_edges(self.g))
        # training defs
        self.train_nid = torch.nonzero(self.g.ndata.pop('train_mask'), as_tuple=True)[0]
        self.dataloader_device = torch.device('cpu')
        if not self.data_cpu and self.num_workers == 0:
            self.train_nid = self.train_nid.to(self.device)
            self.g = self.g.formats(['csc'])
            self.g = self.g.to(self.device)
            self.dataloader_device = self.device
        self.labels = self.g.ndata.pop('label')
        self.node_features = self.g.ndata.pop('feat')
        self.piece_idx = self.g.ndata.pop("score_name")
        self.onsets = self.node_features[:, 0]
        self.score_duration = self.node_features[:, 3]

    def setup_folds(self) -> None:
        train_scores = self.piece_idx[self.train_nid]
        unique_scores = torch.unique(train_scores)
        self.splits = [(torch.nonzero(train_scores != scidx, as_tuple=True)[0], torch.nonzero(train_scores == scidx, as_tuple=True)[0]) for scidx in unique_scores]

    def setup_fold_index(self, fold_index: int) -> None:
        self.train_fold, self.val_fold = self.splits[fold_index]

    def train_dataloader(self):
        return MyLoader(
            self.g,
            self.train_fold,
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
            self.val_fold,
            self.sampler,
            device=self.device,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            use_ddp=self.use_ddp,
            num_workers=self.num_workers)

    def __post_init__(cls):
        super().__init__()


#############################################################################################
#                           Step 3 / 5: Implement the EnsembleVotingModel module            #
# The `EnsembleVotingModel` will take our custom LightningModule and                        #
# several checkpoint_paths.                                                                 #
#                                                                                           #
#############################################################################################


class EnsembleVotingModel(LightningModule):
    def __init__(self, model_cls: Type[LightningModule], checkpoint_paths: List[str]) -> None:
        super().__init__()
        # Create `num_folds` models with their associated fold weights
        self.models = torch.nn.ModuleList([model_cls.load_from_checkpoint(p) for p in checkpoint_paths])
        self.test_acc = Accuracy()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        # Compute the averaged predictions over the `num_folds` models.
        input_nodes, output_nodes, mfgs = batch
        mfgs = [mfg.int().to(self.device) for mfg in mfgs]
        batch_inputs = self.node_features[input_nodes].to(self.device)
        batch_labels = self.labels[output_nodes].to(self.device)
        preds = list()
        for m in self.models:
            batch_pred, prev_encs = m.encoder(mfgs, batch_inputs)
            pred_adj = F.hardshrink(m.decoder(batch_pred, prev_encs), lambd=m.adj_thresh)
            preds.append(m.classifier(pred_adj, batch_pred, prev_encs))
        logits = torch.stack(preds).mean(0)
        loss = F.cross_entropy(logits, batch_labels)
        self.test_acc(logits, batch[1])
        self.log("test_acc", self.test_acc)
        self.log("test_loss", loss)


#############################################################################################
#                           Step 4 / 5: Implement the  KFoldLoop                            #
# From Lightning v1.5, it is possible to implement your own loop. There is several steps    #
# to do so which are described in detail within the documentation                           #
# https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html.                 #
# Here, we will implement an outer fit_loop. It means we will implement subclass the        #
# base Loop and wrap the current trainer `fit_loop`.                                        #
#############################################################################################


#############################################################################################
#                     Here is the `Pseudo Code` for the base Loop.                          #
# class Loop:                                                                               #
#                                                                                           #
#   def run(self, ...):                                                                     #
#       self.reset(...)                                                                     #
#       self.on_run_start(...)                                                              #
#                                                                                           #
#        while not self.done:                                                               #
#            self.on_advance_start(...)                                                     #
#            self.advance(...)                                                              #
#            self.on_advance_end(...)                                                       #
#                                                                                           #
#        return self.on_run_end(...)                                                        #
#############################################################################################


class LOOLoop(Loop):
    def __init__(self, num_folds: int, export_path: str) -> None:
        super().__init__()
        self.num_folds = num_folds
        self.current_fold: int = 0
        self.export_path = export_path

    @property
    def done(self) -> bool:
        return self.current_fold >= self.num_folds

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_folds` from the `BaseKFoldDataModule` instance and store the original weights of the
        model."""
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_folds()
        self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_fold_index` from the `BaseKFoldDataModule` instance."""
        print(f"STARTING FOLD {self.current_fold}")
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)
        self.trainer.datamodule.setup_fold_index(self.current_fold)

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()

        self._reset_testing()  # requires to reset the tracking stage.
        self.trainer.test_loop.run()
        self.current_fold += 1  # increment fold tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current fold and reset the LightningModule and its optimizers."""
        self.trainer.save_checkpoint(osp.join(self.export_path, f"model.{self.current_fold}.pt"))
        # restore the original weights + optimizers and schedulers.
        self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.trainer.strategy.setup_optimizers(self.trainer)
        self.replace(fit_loop=FitLoop)

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        checkpoint_paths = [osp.join(self.export_path, f"model.{f_idx + 1}.pt") for f_idx in range(self.num_folds)]
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer
        # This requires to connect the new model and move it the right device.
        self.trainer.strategy.connect(voting_model)
        self.trainer.strategy.model_to_device()
        self.trainer.test_loop.run()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_fold": self.current_fold}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_fold = state_dict["current_fold"]

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if not key in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]