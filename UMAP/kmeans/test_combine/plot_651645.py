import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns

sns.set(rc={'figure.figsize': (10, 10)})
sns.set(font_scale=1.5)
sns.set_style('whitegrid')


tsne_df = pd.read_csv("tsne_df.csv")

# Generate fake data
#x = tsne_df.query("PCBA1030 == 1")["X"]
#y = tsne_df.query("PCBA1030 == 1")["Y"]

# Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)


fig, ax = plt.subplots()
ax = sns.scatterplot(data=tsne_df.query("PCBA651635!=1"),x="X",y="Y", s=30, color='lightblue')
ax = sns.scatterplot(data=tsne_df.query("PCBA651635==1"),x="X",y="Y", s=30,
        color='red')

#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)

#plt.title("PCBA-1030 dim = 50")
plt.xlim(2.0,22.5)
plt.ylim(-5,22.5)
#fig.colorbar(ax)
#plt.colorbar()
plt.savefig('1030.png')
plt.show()


## Generate fake data
#x = tsne_df.query("PCBA1461 == 1")["X"]
#y = tsne_df.query("PCBA1461 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA1461 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-1461 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('1461.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA1468 == 1")["X"]
#y = tsne_df.query("PCBA1468 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA1468 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-1468 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('1468.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA1688 == 1")["X"]
#y = tsne_df.query("PCBA1688 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA1688 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-1688 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('1688.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA2546 == 1")["X"]
#y = tsne_df.query("PCBA2546 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA2546 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-2546 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('2546.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA2551 == 1")["X"]
#y = tsne_df.query("PCBA2551 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA2551 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-2551 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('2551.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA504332 == 1")["X"]
#y = tsne_df.query("PCBA504332 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA504332 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-504332 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('504332.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA504339 == 1")["X"]
#y = tsne_df.query("PCBA504339 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA504339 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-504339 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('504339.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA504444 == 1")["X"]
#y = tsne_df.query("PCBA504444 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA504444 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-504444 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('504444.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA504467 == 1")["X"]
#y = tsne_df.query("PCBA504467 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA504467 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-504467 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('504467.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA540276 == 1")["X"]
#y = tsne_df.query("PCBA540276 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA540276 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-540276 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('540276.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA588855 == 1")["X"]
#y = tsne_df.query("PCBA588855 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA588855 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-588855 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('588855.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA624288 == 1")["X"]
#y = tsne_df.query("PCBA624288== 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA624288 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-624288 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('624288.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA624296 == 1")["X"]
#y = tsne_df.query("PCBA624296 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA624296 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-624296 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('624296.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA624417 == 1")["X"]
#y = tsne_df.query("PCBA624417 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA624417 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-624417 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('624417.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA651635 == 1")["X"]
#y = tsne_df.query("PCBA651635 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA651635 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-651635 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('651635.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA686970 == 1")["X"]
#y = tsne_df.query("PCBA686970 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA686970 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-686970 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('686970.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA686978 == 1")["X"]
#y = tsne_df.query("PCBA686978 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA686978 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-686978 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('686978.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA686979 == 1")["X"]
#y = tsne_df.query("PCBA686979 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA686979 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-686979 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('686979.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA720504 == 1")["X"]
#y = tsne_df.query("PCBA720504 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA720504 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-720504 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('720504.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA720579 == 1")["X"]
#y = tsne_df.query("PCBA720579 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA720579 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-720579 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('720579.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA720580 == 1")["X"]
#y = tsne_df.query("PCBA720580 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA720580 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-720580 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('720580.png')
#plt.show()
#
#
#
## Generate fake data
#x = tsne_df.query("PCBA883 == 1")["X"]
#y = tsne_df.query("PCBA883 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA883 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-883 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('883.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA884 == 1")["X"]
#y = tsne_df.query("PCBA884 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA884 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-884 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('884.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA891 == 1")["X"]
#y = tsne_df.query("PCBA891 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA891 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-891 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('891.png')
#plt.show()
#
#
## Generate fake data
#x = tsne_df.query("PCBA938 == 1")["X"]
#y = tsne_df.query("PCBA938 == 1")["Y"]
#
## Calculate the point density
#xy = np.vstack([x,y])
#z = gaussian_kde(xy)(xy)
#
#fig, ax = plt.subplots()
#ax = sns.scatterplot(data=tsne_df.query("PCBA938 == 0"),x="X",y="Y", s=50, color='lightblue')
#ax.scatter(x, y, c=z, s=50, edgecolor='white',cmap='YlOrRd',alpha=0.8)
#
#plt.title("PCBA-938 dim = 50")
##plt.xlim(-82,82)
##plt.ylim(-79,78)
##fig.colorbar(ax)
##plt.colorbar()
#plt.savefig('938.png')
#plt.show()
#
