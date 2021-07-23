# Relational-GCN

The preprocessing is slightly different from the original RGCN code. We directly load and preprocess
Mozart Piano Sonatas Score Graph Data. 

### Dependencies
* PyTorch 1.0+
* DGL
* requests


```
pip install requests torch pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification

All experiments use one-hot encoding as featureless input. Best accuracy reported.


MPGD_cad: 
```sh
python entity_classify.py -d mps_cad --testing --gpu 0
```

MPGD_onset

```shell
python entity_classify.py -d mps_onset --testing --gpu 0
```



### Entity Classification w/ minibatch training

Accuracy numbers are reported by 5 runs.

MPS:
```
python entity_classify_mb.py -d mps --testing --gpu 0 --fanout=8
```


### Offline Inferencing
Trained Model can be exported by providing '--model\_path <PATH>' parameter to entity\_classify.py. And then test\_classify.py can load the saved model and do the testing offline.

MPS:
```
python entity_classify.py -d mps --testing --gpu 0 --model_path "mps.pt"
python test_classify.py -d mps --gpu 0 --model_path "mps.pt"
```
