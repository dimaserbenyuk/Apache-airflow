import pandas as pd
import sqlite3

df = pd.read_csv("/tmp/transformed.csv")
conn = sqlite3.connect("/tmp/iris.db")
df.to_sql("iris_data", conn, if_exists="replace", index=False)
print("✅ Data loaded to /tmp/iris.db table: iris_data")
