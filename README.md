# musym-GDL
Geometric Deep Learning Applied on Symbolic Music Scores

Overview
--------

Cadence and Voice Leading Detection in Symbolic Classical Music is a challenging task. This Repository provides method for Geometrical Deep Learning models and score graph modelling paradigms for applying Node Classification on Musical Score graphs for Cadence Detection.



#### A typical Heterogenous graph modelling of the musical score

<img src="static/graph_representation.png" alt="score2graph_representation" style="zoom:50%;" />

<img src="static\node_attributes.png" alt="node_attributes" style="zoom:50%;" />



## Dependencies

- pytorch  1+

- dgl v0.6

- pandas

  

### Quickstart

Install requirements with pip: 

```bash
conda create -n musym python=3.7
conda activate musym
conda install -c dglteam dglconda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

```
or 
```bash
conda env create -f environment.yml
conda activate dar
``` 
or 
```shell
pip install -r requirements.txt
```
or 

Run a experiment :

```shell
cd src/models/rgcn-hetero
python entity_classify.py -d mps_onset --testing --gpu 0
```

