import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("iris_dataset_week4.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "iris_model.joblib")
print("Saved iris_model.joblib")
