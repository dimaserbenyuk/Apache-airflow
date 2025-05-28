import pandas as pd

# df = pd.read_csv("data/iris.csv")
df = pd.read_csv("/opt/airflow/dags/repo/data/iris.csv")
df.to_csv("/tmp/extracted.csv", index=False)
print("✅ Data extracted to /tmp/extracted.csv")
