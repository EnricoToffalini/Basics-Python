
################################

# import packages
import pandas as pd
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt
import seaborn as sns

# import data
dfTot = pd.read_csv("../data/dataForCluster.csv")

# retain only numeric variables
df = dfTot.select_dtypes(include=np.number).dropna()

################################

## K-MEANS

# standardize each feature

x = skl.preprocessing.StandardScaler().fit_transform(df)

# initialize loop to compute silhouette values at various k

silhouetteValues = []
Ks = range(2,7)

# run loop

for k in Ks:
  kmeans = skl.cluster.KMeans(n_clusters=k)
  classification = kmeans.fit_predict(x)
  silhouetteValues.append(
    skl.metrics.silhouette_score(X=x, labels=classification)
  )

# plot silhouette values

plt.figure(figsize=(12,8),dpi=400)
sns.pointplot(x=Ks, y=silhouetteValues)
plt.title("Silhouette scores vs Number of clusters (k)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.show()
plt.savefig("Fig_sil2.png")
plt.clf()

# plot profiles at k = 2

kmeans = skl.cluster.KMeans(n_clusters=2)
classification = kmeans.fit_predict(x)
clusterMeans = pd.DataFrame(x).groupby(classification).mean()
clusterMeans.columns = df.columns

plt.figure(figsize=(12,8),dpi=400)
sns.lineplot(data=clusterMeans.T, dashes=False, markers=True)
plt.title("Cluster profiles (with k-means), k = 2")
plt.ylabel("z-score")
plt.show()
plt.savefig("Fig_Profiles2.png")
plt.clf()

# or even better, with actual values and CIs

dfx = pd.DataFrame(x, columns=df.columns)
dfx["cluster"] = classification
dfxLong = dfx.melt(id_vars="cluster", var_name="Variable", value_name = "Score")

plt.figure(figsize=(12,8),dpi=400)
sns.pointplot(data=dfxLong, x="Variable", y="Score",
    hue="cluster")
plt.title("Cluster profiles (with k-means), k = 2")
plt.ylabel("z-score")
plt.show()
plt.savefig("Fig_Profiles2CIs.png")
plt.clf()

################################

## GAUSSIAN MIXTURE MODELS

# initialize loop to compute BIC for various k

BICs = []
Ks = range(1,7)

# run loop

for k in Ks:
  gmm = skl.mixture.GaussianMixture(n_components=k)
  gmm.fit(x)
  BICs.append(gmm.bic(x))

# plot BIC values

plt.figure(figsize=(12,8),dpi=400)
sns.pointplot(x=Ks, y=BICs)
plt.title("BIC vs Number of clusters (k)")
plt.xlabel("Number of clusters (k)")
plt.ylabel("BIC")
plt.show()
plt.savefig("Fig_BICs.png")
plt.clf()

# plot profiles at k = 2

gmm = skl.mixture.GaussianMixture(n_components=2).fit(x)
clusterMeans = pd.DataFrame(gmm.means_)
clusterMeans.columns = df.columns

plt.figure(figsize=(12,8),dpi=400)
sns.lineplot(data=clusterMeans.T, dashes=False, markers=True)
plt.title("Cluster profiles (with GMM), k = 2")
plt.ylabel("z-score")
plt.show()
plt.savefig("Fig_GMMprofiles2.png")
plt.clf()


################################

