#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will look at a few different ways of visualizing the chemical space covered by a set of molecules. We will cover principal component anlaysis (PCA) and t-distributed stochastic neighbor embedding (t-sne)

# We will start by importing a the necessary Python libraries

# In[1]:

#from IPython import get_ipython
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import SaltRemover
import umap
# Read the input data.  This is a set of ERK2 inhibitors, and associated decoy molecules, from the DUD-E database. 
df = pd.read_csv("supporting_4_combine_oecd.csv")

from rdkit.Chem import PandasTools
PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Mol')


# Define a couple of functions to convert a list SMILES to a list of fingerprints.
def fp_list_from_smiles_list(smiles_list,n_bits=1024):
    fp_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fp_list.append(fp_as_array(mol,n_bits))
    #print(fp_list)
    return fp_list

def fp_as_array(mol,n_bits=1024):
    #fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2,
    #        n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1,), np.int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    #print(arr)
    return arr


# Convert the SMILES from our dataframe to fingerprints.
fp_list = fp_list_from_smiles_list(df.SMILES)
#print(fp_list)

# Perform principal component analysis (PCA) on the fingerprints.
reducer = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=2, random_state=42)
crds_embedded = reducer.fit_transform(fp_list)
#
#
tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])
tsne_df['SMILES'] = df['SMILES']
tsne_df['PCBA1030'] = df['PCBA1030']
tsne_df['PCBA1461'] = df['PCBA1461']
tsne_df['PCBA1468'] = df['PCBA1468']
tsne_df['PCBA1688'] = df['PCBA1688']
tsne_df['PCBA2546'] = df['PCBA2546']
tsne_df['PCBA2551'] = df['PCBA2551']
tsne_df['PCBA504332'] = df['PCBA504332']
tsne_df['PCBA504339'] = df['PCBA504339']
tsne_df['PCBA504444'] = df['PCBA504444']
tsne_df['PCBA504467'] = df['PCBA504467']
tsne_df['PCBA540276'] = df['PCBA540276']
tsne_df['PCBA588855'] = df['PCBA588855']
tsne_df['PCBA624288'] = df['PCBA624288']
tsne_df['PCBA624296'] = df['PCBA624296']
tsne_df['PCBA624417'] = df['PCBA624417']
tsne_df['PCBA651635'] = df['PCBA651635']
tsne_df['PCBA686970'] = df['PCBA686970']
tsne_df['PCBA686978'] = df['PCBA686978']
tsne_df['PCBA686979'] = df['PCBA686979']
tsne_df['PCBA720504'] = df['PCBA720504']
tsne_df['PCBA720579'] = df['PCBA720579']
tsne_df['PCBA720580'] = df['PCBA720580']
tsne_df['PCBA883'] = df['PCBA883']
tsne_df['PCBA884'] = df['PCBA884']
tsne_df['PCBA891'] = df['PCBA891']
tsne_df['PCBA938'] = df['PCBA938']


# Look at the first few lines of the dataframe.
print('tsne_df=',tsne_df.head())
tsne_df.to_csv('tsne_df.csv',index=False)
#
