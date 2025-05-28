import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/tmp/transformed.csv")

# Features and target
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

# Split the data with fixed random state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "/tmp/model.joblib")
print("✅ Model trained and saved to /tmp/model.joblib")

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Confusion matrix
print("\n🔍 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print(cm)


cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot(cmap='Blues')
# plt.title("Confusion Matrix")

# # Сохранить изображение
# plt.savefig("/tmp/confusion_matrix.png")
# print("🖼 Confusion matrix saved to /tmp/confusion_matrix.png")

# Показать изображение (например, в Jupyter или при работе с GUI)
# plt.show()
# Optional: Visualize confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
# disp.plot(cmap="Blues", values_format="d")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()
