import pandas as pd

threshold = 0.55

df = pd.read_csv("PATH_TO_READ")
print(df.head(10))
print(len(df))

print(len(df[(df["label"] == 4) & (df["predict"] != 4) & (df["confidence"] > threshold)]))
df["label"][(df["label"] == 4) & (df["predict"] != 4) & (df["confidence"] > threshold)] = df["predict"]
print(len(df))
print(df.head(10))
df.to_csv("PATH_TO_SAVE")