import pandas as pd

df = pd.read_csv("cereal.csv")

df.loc[df["type"] == "C", 'type'] = 0
df.loc[df["type"] == "H", 'type'] = 1

X = df[[
    "type", "calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass",
    "vitamins", "shelf", "weight", "cups"
]]

y = df["name"]

print(X)
print(y)
