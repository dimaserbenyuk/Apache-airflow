import pandas as pd

df = pd.read_csv("/tmp/extracted.csv")
df["petal_length_norm"] = df["petal_length"] / df["petal_length"].max()
df.to_csv("/tmp/transformed.csv", index=False)
print("✅ Data transformed to /tmp/transformed.csv")
