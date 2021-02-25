import pandas as pd
import os

threshold = 0.55

df = pd.read_csv("./mendeleyframe.csv")
print(df.head())
print(len(df))

print(len(df[(df["label"] == -1) & (df["predict"] != 4) & (df["confidence"] > threshold)]))
df["label"][(df["label"] == -1) & (df["predict"] != 4) & (df["confidence"] > threshold)] = df["predict"]
df = df[df["label"] != -1]
print(len(df))
print(df.head())
df[["image_id", "label"]].to_csv("./pl.csv")

for file in os.listdir("./mendeley_leaf_data"):
    if file not in df["image_id"].values:
        os.remove(os.path.join("./mendeley_leaf_data", file))

source = pd.read_csv("./kd.csv")
df["source"] = [2019 for i in range(len(df))]
print(df.head())
final = pd.concat([source, df], ignore_index=True)
final.to_csv("./train.csv")