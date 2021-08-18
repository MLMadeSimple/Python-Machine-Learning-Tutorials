import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth

df = pd.read_csv("cereal.csv")

df.loc[df["type"] == "C", 'type'] = 0
df.loc[df["type"] == "H", 'type'] = 1

X = df[[
    "type", "calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass",
    "vitamins", "shelf", "weight", "cups"
]]

y = df["name"]

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
print(bandwidth)

model = MeanShift(bandwidth=bandwidth)
model.fit(X)

labels = model.labels_

labels_unique = np.unique(labels)
n_clusters = len(labels_unique)

print(n_clusters)

final_df = df.merge(pd.DataFrame(labels, columns=["result"]), left_index=True, right_index=True)
final_df.to_csv("cereal_prediction.csv")
