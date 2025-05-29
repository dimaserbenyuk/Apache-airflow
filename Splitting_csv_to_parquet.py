import pandas as pd
import pyarrow.parquet as pq
import os
from pathlib import Path
from tqdm import tqdm

# Настройки
input_csv = Path("/tmp/synthetic_iris_million.csv")
output_dir = Path("/Users/dima/Apache-airflow/iris_parquet_parts")
num_parts = 20

# Чтение исходного CSV
print(f"📥 Loading CSV: {input_csv}")
df = pd.read_csv(input_csv)
print(f"✅ Loaded: {len(df):,} rows")

# Подготовка папки
output_dir.mkdir(parents=True, exist_ok=True)

# Разбиение и сохранение
rows_per_part = len(df) // num_parts
print(f"🪓 Splitting into {num_parts} parts...")

for i in tqdm(range(num_parts), desc="📤 Writing Parquet"):
    start = i * rows_per_part
    end = None if i == num_parts - 1 else (i + 1) * rows_per_part
    chunk = df.iloc[start:end]
    
    filename = f"train-{i:05d}-of-{num_parts:05d}.parquet"
    chunk.to_parquet(output_dir / filename, index=False, engine="pyarrow")

print(f"\n✅ Split complete: {num_parts} files saved in {output_dir}\n")

# Проверка целостности файлов
print("🔍 Validating parquet files...")
files = sorted(output_dir.glob("*.parquet"))
total_rows = 0
first_schema = pq.read_schema(files[0])

for f in tqdm(files, desc="📄 Validating"):
    try:
        df_part = pd.read_parquet(f)
        total_rows += len(df_part)
        current_schema = pq.read_schema(f)
        assert current_schema == first_schema, f"❌ Schema mismatch in {f.name}"
        print(f"✅ OK: {f.name} — shape={df_part.shape}")
    except Exception as e:
        print(f"❌ Error reading {f.name}: {e}")

print(f"\n📊 Total rows across all parts: {total_rows:,}")
print("✅ All schemas match.")
