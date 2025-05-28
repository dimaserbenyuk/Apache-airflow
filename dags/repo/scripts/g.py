from sklearn.datasets import make_classification
import pandas as pd

# Генерация синтетического датасета, похожего на Iris
X, y = make_classification(
    n_samples=1_000_000,   # 1 миллион строк
    n_features=4,
    n_informative=4,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.5,
    random_state=42
)

# Создаем DataFrame
df = pd.DataFrame(X, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
df["species"] = y

# Преобразуем метки в строковые значения, как в Iris
df["species"] = df["species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Сохраняем в CSV (можно сохранить как parquet, если нужно сжать)
csv_path = "/tmp/synthetic_iris_million.csv"
df.to_csv(csv_path, index=False)

csv_path
