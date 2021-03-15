import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df = pd.read_csv("scaled.txt",header=None,names=['X','Y'],delimiter=r"\s+")
labels = pd.read_csv("labels.txt",header=None,names=["label"])
df['label']=labels.iloc[:,0]
print(labels)
print(df)
plt.scatter(df['X'],df['Y'],c=df['label'],cmap='rainbow',s=15,edgecolor='black')
plt.show()
plt.savefig("4.png")
