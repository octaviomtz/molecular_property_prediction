# Overview
Use GNN to determine the ability for a chemical compound to inhibit HIV replication.       
Data:   
- chemical structure under SMILES format   
- HIV-activity: (0) innactive or (1) active compound
![hydra_folders](/github_images/output.png?raw=true)

## Installing requirments to use colab GPU via ssh
```bash
pip3 install -r requirements.txt
bash install_geometric.sh
```

## Train (using hydra and mlflow)
```bash
python3 train.py
```


## Comments about the code
This is the code for this video series: https://www.youtube.com/watch?v=nAEb1lOf_4o

## Further things
- Its highly recommended to setup a GPU (including CUDA) for this code. 
- Here is where I found ideas for node / edge features: https://www.researchgate.net/figure/Descriptions-of-node-and-edge-features_tbl1_339424976
- There is also a Kaggle competition that used this dataset (from a University):
https://www.kaggle.com/c/iml2019/overview

