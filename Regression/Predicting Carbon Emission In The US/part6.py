import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("owid-co2-data.csv")

mask = df["iso_code"] == "USA"
df = df.loc[mask]

x = df["year"]
y = df["cumulative_co2"]

model = np.poly1d(np.polyfit(x, y, 2))

r2 = r2_score(y, model(x))

x_hat = x.append(pd.Series(range(x.max() + 1, x.max() + 50)))
y_hat = model(x_hat)

plt.plot(x, y)
plt.plot(x_hat, y_hat)
plt.show() 
