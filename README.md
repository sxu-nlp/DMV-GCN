
## DMV_GCN-pytorch

This is the Pytorch implementation for our JMLC 2022 paper:

>Liu, F., Liao, J., Zheng, J. et al. GCN recommendation model based on the fusion of dynamic multiple-view latent interest topics. Int. J. Mach. Learn. & Cyber. (2022). 
[Paper link](https://doi.org/10.1007/s13042-022-01743-z).
[code link](https://github.com/sxu-nlp/DMV-GCN)

## Introduction

We designed the DMV_GCN model,it contains three parts:  
1.We firstly construct multiple views.  
2.We use the View-specific LightGCN(VS-LightGCN) to learn the rating matrix between the user and item in every view.  
3.We use DMV_GCN to integrate multiple view of the rating matrix.




## Enviroment Requirement

`pip install -r requirements.txt`



## Dataset

We provide three processed datasets: LastFM, Movielens-100k , Movielens-1M.

see more in `dataloader.py`

## An example to create multiple graphs
create two graphs on  **lastfm** dataset:
* change base directory

Change `ROOT_PATH` in `code/world.py`

* command  
` cd code && python create_graph.py --dataset="lastfm" --t="[64,52]" --beta="[1e-6,1e-5]"`

**note: if you want to run our model on your dataset , please create the multiple graphs firstly!**

## An example to run a 3-layer VS-LightGCN on graph-1

run VS-LightGCN on **lastfm** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command

` cd code && python main.py --decay=1e-3 --lr=0.001 --layer=3 --dataset="lastfm" --alpha=0.4 --graphID="1" --save=0`

## An example to run DMV_GCN

run DMV_GCN on **ml-100k** dataset:

* change base directory

Change `ROOT_PATH` in `code/world.py`

* command
` cd code && python run.py --dataset="ml-100k" `

## the experimental parameters on three data sets
* lastfm  
alpha:  graph1=0.4, graph2=0.4, graph3=0.4  
decay=1e-3  
lr=0.001  

* ml-100k
alpha: graph1=0.4, graph2=0.4, graph3=0.4 
decay=1e-5  
lr=0.001  

* ml-1m
alpha: graph1=0.4, graph2=0.4, graph3=0.4
decay=1e-4  
lr=0.001  

