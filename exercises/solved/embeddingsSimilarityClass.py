
###########################################

# !pip install sentence-transformers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

df = pd.read_csv("../data/emotionalTexts1.csv")

###########################################

# import model

model = SentenceTransformer("all-MiniLM-L6-v2")

###########################################

# extract embeddings

sentenceEmbeddings = model.encode(df["text"], show_progress_bar=True)
sentenceEmbeddings.shape

###########################################

# compute similarities 

emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
targetEmotions = ["a person feeling " + e for e in emotions]
emotionEmbeddings = model.encode(emotions, show_progress_bar=True)

similarityMatrix = cosine_similarity(sentenceEmbeddings, emotionEmbeddings)

plt.figure(figsize=(12, 20))
plt.imshow(similarityMatrix[:25,:], aspect='auto', cmap='Blues')
plt.colorbar()
plt.title('Cosine Similarity Heatmap')
plt.ylabel('Sentence #')
plt.xticks(ticks=range(0,len(emotions)),labels=emotions)
plt.yticks(ticks=range(0,25))
plt.show()
plt.clf(); plt.close()

###########################################

# predict closest emotion

closestEmotion = similarity_matrix.argmax(axis=1)
df["predictedEmotion"] = [emotions[i] for i in closestEmotion]
df.head()

###########################################

# see accuracy from labelled 

dfLabelled = pd.read_csv("../data/emotionalTexts1Labelled.csv")
df["labelledEmotion"] = dfLabelled["emotion"]

confusionMatrix = pd.crosstab(pd.Series(df["labelledEmotion"]),
                              pd.Series(df["predictedEmotion"]))

plt.figure(figsize=(6, 5))
sns.heatmap(confusionMatrix, annot=True, cmap='Blues')
plt.title("Predicted-Labelled Confusion Matrix")
plt.show()
plt.clf(); plt.close()

###########################################
