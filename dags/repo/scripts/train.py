import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("/tmp/transformed.csv")

# Обновлённый набор признаков
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, "/tmp/model.joblib")
print("✅ Model trained and saved to /tmp/model.joblib")
