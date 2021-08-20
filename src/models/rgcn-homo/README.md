# Relational-GCN 

The preprocessing is slightly different from the original RGCN code. We directly load and preprocess
Mozart Piano Sonatas Score Graph Data. 

### Dependencies
* PyTorch 1.0+
* DGL

Example code was tested with pytorch 1.8.1 and pandas 0.23.4

### Entity Classification

Augmentation is performed inside Dataset Class. Applying Normalization with 0 mean and unit variance per node type. Best accuracy reported.

MPGD_cad: 
```sh
python entity_classify.py -d mps_cad --testing --gpu 0
```

MPGD_onset

```sh
python entity_classify.py -d mps_onset --testing --gpu 0
```

Toy Dataset

```sh
python entity_classify.py -d toy --testing --gpu 0
```


### Entity Classification with Mini-Batch Sampling

Run with `--inductive` argument creates an inductive split in the dataset otherwise the entire dataset graph is used for training, validation and testing. Use `--gpu >= 0` for GPU usage otherwise use `--gpu -1` for a CPU run.


MPGD_onset

```sh
python entity_classify_mp.py --dataset mps_onset --num-of-epochs 30 --gpu -1
```

Toy Dataset

```sh
python entity_classify_mp.py --dataset toy --num-of-epochs 30 --gpu -1
```

Cora Dataset

```sh
python entity_classify_mp.py --dataset cora --num-of-epochs 30 --gpu -1
```

Reddit Dataset

```sh
python entity_classify_mp.py --dataset reddit --num-of-epochs 30 --gpu -1
```


The Mini-Batch Uses a Neighhbor Sampling with a Stratified Sampler on top for unbalanced classes.