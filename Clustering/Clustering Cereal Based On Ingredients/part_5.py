import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA as sklearnPCA

df = pd.read_csv("cereal.csv")

df.loc[df["type"] == "C", 'type'] = 0
df.loc[df["type"] == "H", 'type'] = 1

X = df[[
    "type", "calories", "protein", "fat", "sodium", "fiber", "carbo", "sugars", "potass",
    "vitamins", "shelf", "weight", "cups"
]]

y = df["name"]

bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

model = MeanShift(bandwidth=bandwidth)
model.fit(X)

labels = model.labels_
cluster_centers = model.cluster_centers_

labels_unique = np.unique(labels)
n_clusters = len(labels_unique)

# Principal Component Analysis (PCA)
X_norm = (X - X.min())/(X.max() - X.min())
pca = sklearnPCA(n_components=2)
transformed = pd.DataFrame(pca.fit_transform(X_norm))

for k in range(n_clusters):
    class_points = transformed[labels[transformed.index] == k]
    plt.scatter(class_points[0], class_points[1], label='Class {}'.format(k))

plt.legend()
plt.show()

