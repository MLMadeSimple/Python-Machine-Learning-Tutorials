import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def build_model(m):
    m.fit(X_train, y_train)
    print(m.score(X_test, y_test))
    return m


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

print("KNeighborsClassifier")
model = KNeighborsClassifier(n_neighbors=6)
build_model(model)

print("\nGaussianNB")
model = GaussianNB()
build_model(model)

print("\nDecisionTreeClassifier")
model = DecisionTreeClassifier(max_depth=None)
build_model(model)

print("\nRandomForestClassifier")
model = RandomForestClassifier(max_depth=None)
build_model(model)

print("\nLogisticRegression")
model = LogisticRegression(max_iter=500)
build_model(model)
