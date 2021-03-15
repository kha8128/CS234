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


# We set a few parameters to improve the appearance of our plots

# In[2]:


sns.set(rc={'figure.figsize': (10, 10)})
sns.set(font_scale=1.5)
sns.set_style('whitegrid')


# Read the input data.  This is a set of ERK2 inhibitors, and associated decoy molecules, from the DUD-E database. 

# In[3]:


df = pd.read_csv("supporting_4.csv")


# Take a look at how many rows and columns we have in the data

# In[4]:


print('shape=',df.shape)


# Look at the first few lines in the dataframe.

# In[5]:


print('df=',df.head())


# In[6]:


rslt_df = df.loc[df['SMILES'].str.match('O=C1N/C(=C/c2ccco2)C(=O)N1Cc1ccccc1F')]
print('rslt_df=',rslt_df)


# In[7]:


from rdkit.Chem import PandasTools
PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Mol')


# Define a couple of functions to convert a list SMILES to a list of fingerprints.

# In[9]:


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


# Convert the SMILES from our dataframe to fingerprints.

# In[10]:


fp_list = fp_list_from_smiles_list(df.SMILES)


# Perform principal component analysis (PCA) on the fingerprints.

# In[11]:


#for i in range(0,2048):
#    print(fp_list[15][i], end = '')


# In[12]:


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
crds = pca.fit_transform(fp_list)


# Run the t-sne on the 50 principal component database we created above.

# In[ ]:


from sklearn.manifold import TSNE
crds_embedded = TSNE(n_components=2).fit_transform(crds)

# Put the t-sne dataset into a dataframe to simplify plotting.

# In[ ]:


tsne_df = pd.DataFrame(crds_embedded,columns=["X","Y"])

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

# In[ ]:


print('tsne_df=',tsne_df.head())
tsne_df.to_csv('tsne_df.csv',index=False)

# Plot the distribution of the active and decoy molecule with t-sne.  Note that, as above, we plot in two steps to avoid obscuring the red points. 

# In[ ]:


#ax = sns.scatterplot(data=tsne_df.query("PCBA883 == 0"),x="X",y="Y",color='lightblue')
#ax = sns.scatterplot(data=tsne_df.query("PCBA883 == 1"),x="X",y="Y",color='red')

#ax = sns.scatterplot(data=tsne_df.query("is_active == 0 & X < -10 & Y < -20"),x="X",y="Y",color='lightblue')
#ax = sns.scatterplot(data=tsne_df.query("is_active == 1 & X < -10 & Y < -20"),x="X",y="Y",color='red')

#ax = sns.scatterplot(data=tsne_df.query("is_active ==0 & X < -15 & X > -16 & Y > -29 & Y < -25"),x="X",y="Y",color='lightblue')
#ax = sns.scatterplot(data=tsne_df.query("is_active == 1 & X < -15 & X > -16 & Y > -29 & Y < -25"),x="X",y="Y",color='red')


# In[ ]:


#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.stats import gaussian_kde
#
## Generate fake data
#x = tsne_df.query("PCBA1030 == 1")["X"]
#y = tsne_df.query("PCBA1030 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA1030 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='',cmap="YlOrRd",alpha=0.8)
#plt.title("PCBA-1030")
#plt.xlim(-82,82)
#plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.show()
#
#
