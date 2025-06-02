
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

model = SentenceTransformer("all-MiniLM-L12-v2")

input_sentence = "That was absolutely terrifying. My hands were shaking, my heart pounded so loudly I could barely hear my own thoughts, and each creak of the floorboards seemed to scream danger. I stood frozen in place, staring at the door, every nerve in my body bracing for something awful to emerge from the darkness behind it."

emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
emotion_sentences = ["a person feeling " + e  for e in emotions]

input_embedding = model.encode([input_sentence])
emotion_embeddings = model.encode(emotion_sentences)

similarities = cosine_similarity(input_embedding, emotion_embeddings)[0]

df = pd.DataFrame({
  "target": emotions,
  "cosineSimilarity": similarities
  })
df = df.sort_values(by="cosineSimilarity",ascending=False).reset_index(drop=True)
df.cosineSimilarity = df.cosineSimilarity
print(df)

df.to_dict()


