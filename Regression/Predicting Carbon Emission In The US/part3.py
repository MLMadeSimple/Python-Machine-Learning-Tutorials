import pandas as pd

df = pd.read_csv("owid-co2-data.csv")

mask = df["iso_code"] == "USA"
print(mask)
df = df.loc[mask]

print(df)
