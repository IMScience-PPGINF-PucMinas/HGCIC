Hierarchical Graph Convolutional Networks for Image Classification
=====
Implementation of our BRACIS 2023 paper "Hierarchical Graph Convolutional Networks for Image Classification" authored by:
[João Pedro Oliveira Batisteli](https://lattes.cnpq.br/8128547685252443), [Silvio Jamil F. Guimarães](http://lattes.cnpq.br/8522089151904453) and 
[Zenilton K. G. Patrocínio Jr](http://lattes.cnpq.br/8895634496108399),


Graph-based image representation is a promising research direction that can capture the structure and semantics of images. How- ever, existing methods for converting images to graphs often fail to pre- serve the hierarchical information of the image elements and produce sub-optimal or poor regions. To address these limitations, we propose a novel approach that uses a hierarchical image segmentation technique to generate graphs at multiple segmentation scales, capturing the hierar- chical relationships between image elements. We also propose and train a Hierarchical Graph Convolutional Network for Image Classification (HGCIC) model that leverages the hierarchical information with three different adjacency setups on the CIFAR-10 database. Experimental re- sults show that the proposed approach can achieve competitive or supe- rior performance compared to other state-of-the-art methods while using smaller graphs.

## Getting started
### Prerequisites
0. Clone this repository
```
# no need to add --recursive as all dependencies are copied into this repo.
git clone "https://github.com/JPBatisteli/HGCIC.git"
cd "HGCIC"
```

1. Create and activate the environment
```bash 
conda env create -f environment.yml
conda activate hgcic
```

2. Prepare feature files

The graphs will be generated and saved in the `data/processed` folder the first time the training or test script is executed. 
To create datasets in one of the configurations modify the following line of code in the scripts: 

-Hierarchy adjacency:
```python:
 Cifar10_graphs(root="data/", nodes=20, k_neighbors=8, use_knn=False, complete_graph=True)
```

-KNN adjacency:
```python:
 Cifar10_graphs(root="data/", nodes=20, k_neighbors=8, use_knn=True, complete_graph=False)
```

-Complete adjacency:
```python:
 Cifar10_graphs(root="data/", nodes=20, k_neighbors=8, use_knn=True, complete_graph=True)
```

### Training and Inference

1. To perform model training run:
```bash 
mlflow ui
```

And in another terminal:
```bash
python train.py
```

You can change the name of the experiment and the url that mlflow will use in the header of the training script, which is previously defined as:

```python:
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Experiment 1")
```

The weight directory and file name can be changed and its default setting is:

```python:
torch.save(model.state_dict(), "weights/hierarchy_adj.pth")
```

2. To perform model test run:

```bash 
python test.py
```

Don't forget to change directory to the model weights you want to test. The weights to reproduce the results reported in table 2 of the article are available in the folder `weights/reproduce`.


## Citations
If you find this code useful for your research, consider cite our paper:
```
Not available yet!
```


## Contact
João Pedro Oliveira Batisteli: joao.batisteli@sga.pucminas.br

