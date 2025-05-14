
###########################################

# !pip install sentence-transformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

df = pd.read_csv("../data/emotionalTexts1.csv")

###########################################

# import model

model = SentenceTransformer("all-MiniLM-L6-v2")

###########################################

# extract embeddings

embeddings = model.encode(df["text"], show_progress_bar=True)
embeddings.shape

###########################################

# full pca for demonstration

pca = PCA()
pca_emb = pca.fit_transform(embeddings)
pca_expl_var = pca.explained_variance_ratio_
pca_cum_var = cumulative_variance = np.cumsum(pca_expl_var)

plt.figure(figsize=(10, 6))
plt.plot(range(1,len(pca_cum_var)+1),pca_cum_var)
plt.axhline(y=0.90, color='r', linestyle=':')
plt.show()

###########################################

# feasible pca for our purposes

pca10 = PCA(n_components=10)
pca10_emb = pca.fit_transform(embeddings)

###########################################

# grouping into a few clusters of texts

kmeans = KMeans(n_clusters=6)
df["cluster"] = kmeans.fit_predict(pca10_emb)
df["cluster"] = df["cluster"].astype(str)

###########################################

# plot into 2-D

pca2D = PCA(n_components=2)
pca2D_emb = pca2D.fit_transform(embeddings)

plt.figure(figsize=(20, 12),dpi=500)
sns.scatterplot(x=pca2D_emb[:, 0], y=pca2D_emb[:, 1], hue=df["cluster"], s=120, alpha=0.8)
plt.title("Emotional Texts Clusters in 2D PCA space", fontsize=20)
plt.xlabel("PC1", fontsize=20)
plt.ylabel("PC2", fontsize=20)
plt.xticks(fontsize=16);
plt.yticks(fontsize=16);
plt.legend(title="Cluster", fontsize=12)
plt.show()

###########################################


