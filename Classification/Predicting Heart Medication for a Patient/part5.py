import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Drug Classification.csv")

columns = ["Sex", "BP", "Cholesterol"]

one_hot_encoder = OneHotEncoder(sparse=False)

one_hot_encoded = one_hot_encoder.fit_transform(df[columns])
labels = one_hot_encoder.get_feature_names(columns)

for i, label in enumerate(labels):
    df[label] = one_hot_encoded[:, i]

X = df[['Sex_F', 'Sex_M', 'BP_HIGH', 'BP_LOW', 'BP_NORMAL', 'Cholesterol_HIGH', 'Cholesterol_NORMAL', 'Na_to_K']]
y = df["Drug"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier(max_depth=None)
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(model.predict([
    [1, 0, 1, 0, 0, 1, 0, 1.36],
    [0, 1, 0, 1, 0, 1, 0, 5.6],
    [1, 0, 0, 0, 1, 0, 1, 8.5],
    [0, 1, 0, 1, 0, 0, 1, 10.6],
]))
