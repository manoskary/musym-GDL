# Relational-GCN

The preprocessing is slightly different from the original RGCN code. We directly load and preprocess
Mozart Piano Sonatas Score Graph Data. 

### Dependencies
* PyTorch 1.0+
* DGL

Example code was tested with pytorch 1.8.1 and pandas 0.23.4

### Entity Classification

All standarization and augmentation is performed inside Dataset Class. Best accuracy reported.

MPGD_cad: 
```sh
python entity_classify.py -d mps_cad --testing --gpu 0
```

MPGD_onset

```shell
python entity_classify.py -d mps_onset --testing --gpu 0
```


