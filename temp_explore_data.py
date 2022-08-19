#%%
import numpy as np
import pandas as pd
# from dataset import MoleculeDataset
from dataset_featurizer import MoleculeDataset
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

#%%
data = pd.read_csv("data/raw/HIV_train.csv")
data.head()

#%%
data = pd.read_csv("data/raw/HIV.csv")
data.head()
data.shape

#%%
print(data.shape)
print(data['HIV_active'].value_counts())

# %%
ii=12
sample_smiles = data['smiles'][ii:ii+9].values
sample_mols = [Chem.MolFromSmiles(smiles) for smiles in sample_smiles]
grid = Draw.MolsToGridImage(sample_mols, molsPerRow=3, subImgSize=(200,200))
grid

# %%
#to call dataset_featurizer.py (it used to be dataset) use:
dataset = MoleculeDataset(root='data', filename='HIV.csv')

# %%
dataset[0].edge_index.t()
dataset[0].x
dataset[0].edge_attr
dataset[0].y