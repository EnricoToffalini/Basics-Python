
#######################################

# !pip install transformers
# !pip install torch
# !pip install hf_xet

# import packages and functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import transformers
from torch.nn.functional import softmax

# import data from excel
df = pd.read_excel("../data/panasonicTrimmer.xlsx")

#######################################

modelName = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = transformers.AutoTokenizer.from_pretrained(modelName)
model = transformers.AutoModelForSequenceClassification.from_pretrained(modelName)
classifier = transformers.pipeline(task="sentiment-analysis", model=model, tokenizer=tokenizer)

#######################################

pred = [classifier(t)[0] for t in df["Text"]]
pred = pd.DataFrame(pred)

conditions = [pred["label"]=="LABEL_0", pred["label"]=="LABEL_1", pred["label"]=="LABEL_2"]
labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
df["predictedSentiment"] = np.select(conditions, labels, default="")

df["trueSentiment"] == df["predictedSentiment"]
CM = confusion_matrix(df["trueSentiment"], df["predictedSentiment"], labels=labels)
print(CM)
print(CM / CM.sum(1))

#######################################

df["valence"] = np.nan
for i in range(len(df["Text"])):
  sentence = tokenizer(df["Text"][i], return_tensors="pt")
  logits = model(**sentence).logits
  probs = softmax(logits, dim=1)
  df.loc[i,"valence"] = float(probs[0][0]*(-1) + probs[0][1]*0 + probs[0][2]*(+1))

plt.figure(figsize=(12,6))
sns.scatterplot(x=df["Stars"],y=df["valence"], alpha=0.4, s=100, color="darkblue")
plt.show()
plt.clf(); plt.close()

df["valenceRescaled"] = (df["valence"]+1)/2

plt.figure(figsize=(12,6))
sns.regplot(x=df["Stars"],y=df["valenceRescaled"], logistic=True, scatter_kws={"alpha": 0.4, "s":100}, color="darkblue")
plt.show()
plt.clf(); plt.close()

#######################################


