
###############################################

import openai
import pandas as pd
import random
import time

random.seed(30)

# Set your own OpenAI API key... You need to have an OPENAI billing plan!
client = openai.OpenAI(api_key="sk-proj-4slS9XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
model = "gpt-4o"

###############################################

# Define emotion categories and sample size
N = 550
Emotions = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

# Initialize a dataframe with random emotions and empty texts
df = pd.DataFrame({
    "id": range(1, N+1),
    "text": ""*N,
    "emotion": [random.choice(Emotions) for i in range(N)]
})

###############################################

# Loop over the dataframe and generate texts using OPENAI's GPT-4o-mini
for i in range(N):
  
    emotion = df.loc[i, "emotion"]
  
    try:
         response = client.chat.completions.create(
             model = model,
             messages = [
                 {"role": "system", "content": "Account simple, everyday human experiences talking in first person"},
                 {"role": "user", "content": f"Write a short paragraph (1–2 sentences) that conveys {emotion} without explicitly naming the emotion."}
              ]
          )
         
         text = response.choices[0].message.content
         df.loc[i, "text"] = text
         print(str(i)+" "+emotion+" --- "+text+"\n\n")

         time.sleep(0.1)
         df.to_csv("emotionalTexts.csv", index=False)

    except Exception as e:
         print("ERROR; ROW: " + str(i) + str(e))

###############################################

# Export to CSV

df.to_csv("emotionalTexts.csv", index=False)

###############################################

