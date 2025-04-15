import BagOfWords
import pandas as pd
import os

bow = BagOfWords.BagOfWords(extra_stopwords=["movie", "film"])
df = None
if os.path.exists("train_pos.csv") and os.path.exists("train_neg.csv"):
    df = pd.concat(
        [
            pd.read_csv("train_pos.csv", encoding="utf-8"), 
            pd.read_csv("train_neg.csv", encoding="utf-8")
        ],
        ignore_index=True
    )

bow.adapt(df["content"])

text = "bad good movie great well fantastic terrible horrible"
b = bow.bag([text])[0]
print(b)
print(sum(b))