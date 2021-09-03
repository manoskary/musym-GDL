# Relational-GCN 

The preprocessing is slightly different from the original RGCN code. We directly load and preprocess
Mozart Piano Sonatas Score Graph Data. 

### Dependencies
* PyTorch 1.0+
* DGL

Example code was tested with pytorch 1.8.1 and pandas 0.23.4

### Entity Classification

Augmentation is performed inside Dataset Class. Applying MinMaxScaler per node Feature. Best accuracy reported. Run with `--inductive` argument creates an inductive split in the dataset otherwise the same subset of the training part of the dataset graph is used for training, validation and testing. Use `--gpu >= 0` for GPU usage otherwise use `--gpu -1` for a CPU run.

MPGD_cad: 
```sh
python entity_classify.py --dataset mps_cad --gpu 0
```

MPGD_onset

```sh
python entity_classify.py --dataset mps_onset --gpu 0
```

Toy Dataset

```sh
python entity_classify.py --dataset toy --gpu 0
```


### Entity Classification with Mini-Batch Sampling

Run with `--inductive` argument creates an inductive split in the dataset otherwise the entire dataset graph is used for training, validation and testing. Use `--gpu >= 0` for GPU usage otherwise use `--gpu -1` for a CPU run.


MPGD_onset

```sh
python entity_classify_mp.py --dataset mps_onset --num-epochs 30 --gpu -1
```

Toy Dataset

```sh
python entity_classify_mp.py --dataset toy --num-epochs 30 --gpu -1
```

Cora Dataset

```sh
python entity_classify_mp.py --dataset cora --num-epochs 30 --gpu -1
```

Reddit Dataset

```sh
python entity_classify_mp.py --dataset reddit --num-epochs 30 --gpu -1
```


The Mini-Batch Uses a Neighhbor Sampling with a Stratified Sampler on top for unbalanced classes. If you want to try a different number of layers, i.e. `--num-layers`, for the network you will have to adapt the `--fan-out` accordingly.


### Entity Classification with Grid Searching

We use ray for hyperparameter optimization and wandb for experiment logging.
Please use `pip install ray -U` to install ray  and `pip install wandb` to Install Weights and Biases.


Run with `--inductive` argument creates an inductive split in the dataset otherwise the entire dataset graph is used for training, validation and testing. Use `--gpu >= 0` for GPU usage otherwise use `--gpu -1` for a CPU run.

MPGD_onset

```sh
python sweep-tune-hyperopt.py --dataset mps_onset --inductive --gpu 0
```

Toy Dataset

```sh
python sweep-tune-hyperopt.py --dataset toy --inductive --gpu 0
```

Cora Dataset

```sh
python sweep-tune-hyperopt.py --dataset cora --inductive --gpu 0
```

Reddit Dataset

```sh
python sweep-tune-hyperopt.py --dataset reddit --inductive --gpu 0
```