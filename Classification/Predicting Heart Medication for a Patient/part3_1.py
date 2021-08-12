import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Drug Classification.csv")

columns = ["Sex", "BP", "Cholesterol"]

one_hot_encoder = OneHotEncoder(sparse=False)

one_hot_encoded = one_hot_encoder.fit_transform(df[columns])
labels = one_hot_encoder.get_feature_names(columns)

for i, label in enumerate(labels):
    df[label] = one_hot_encoded[:, i]

print(df)
