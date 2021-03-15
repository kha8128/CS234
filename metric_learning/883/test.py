import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
from tqdm import tqdm
import seaborn as sns
import umap

from sklearn.manifold import TSNE
import metric_learn
import numpy as np
from sklearn.datasets import make_classification, make_regression

sns.set(rc={'figure.figsize': (10, 10)})
sns.set(font_scale=1.5)
sns.set_style('whitegrid')

# visualisation imports
import matplotlib.pyplot as plt
np.random.seed(42)


df = pd.read_csv("supporting_4.csv")

from rdkit.Chem import PandasTools
PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Mol')


def fp_list_from_smiles_list(smiles_list,n_bits=2048):
    fp_list = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        fp_list.append(fp_as_array(mol,n_bits))
    return fp_list

def fp_as_array(mol,n_bits=2048):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr = np.zeros((1,), np.int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


fp_list = fp_list_from_smiles_list(df.SMILES)
df['fingerprint'] = fp_list

reducer = umap.UMAP(n_neighbors=20, min_dist=0.0, n_components=2, random_state=42)
crds_embedded = reducer.fit_transform(fp_list)
print("umap x y", crds_embedded)
tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])
tsne_df['SMILES'] = df['SMILES']
tsne_df['PCBA883'] = df['PCBA883']

print('umap_df=',tsne_df.head())
tsne_df.to_csv('umap_df.csv',index=False)


# setting up LMNN
lmnn = metric_learn.LMNN(k=5, learn_rate=1e-6)

# fit the data!
lmnn_df = pd.DataFrame()
lmnn_df['fingerprint']=df['fingerprint']
lmnn_df['SMILES'] = df['SMILES']
lmnn_df['PCBA883'] = df['PCBA883']
lmnn_df=lmnn_df.dropna()
lmnn_df.reset_index(drop=True,inplace=True)
X = lmnn_df['fingerprint']
y = lmnn_df['PCBA883']

print(X.values.tolist())
print(y.values.tolist())

lmnn.fit(X.values.tolist(),y.values.tolist())

# transform our input space
X_lmnn = lmnn.transform(X.values.tolist())
crds_embedded_2 = reducer.fit_transform(X_lmnn)
print("lmnn embedded", crds_embedded_2)
metric_df = pd.DataFrame(crds_embedded_2,columns=["X","Y"])
metric_df['PCBA883'] = lmnn_df['PCBA883'] 
metric_df['SMILES'] = lmnn_df['SMILES']
print('metric_df=',metric_df.head())
metric_df.to_csv('metric_df.csv',index=False)

print(X.values.tolist())
print(y.values.tolist())

