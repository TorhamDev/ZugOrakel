import gc
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

DATA_FILE = "cleaned_data.parquet"
MAX_ROWS = 10_000_000
RAM_LIMIT = "8gb"

cat_features = ["eva", "station_name", "train_type"]

print("Loading dataset (row-group sampling)...")
pf = pq.ParquetFile(DATA_FILE)
fraction = min(1.0, MAX_ROWS / pf.metadata.num_rows)
rng = np.random.default_rng(42)

tables = []
for i in range(pf.num_row_groups):
    group = pf.read_row_group(i)
    if fraction < 1.0:
        keep = np.flatnonzero(rng.random(group.num_rows) < fraction)
        group = group.take(pa.array(keep))
    tables.append(group)
    del group

df = pa.concat_tables(tables).to_pandas()
del tables
gc.collect()
print(f"Using {len(df):,} of {pf.metadata.num_rows:,} rows")

y = df.pop("delay_in_min")
X = df
for col in cat_features:
    X[col] = X[col].fillna("unknown").astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
del X, y, df
gc.collect()


model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.08,
    depth=6,
    cat_features=cat_features,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    thread_count=-1,
    used_ram_limit=RAM_LIMIT,
    verbose=100,
)

print("Training CatBoost model...")
start_time = time.time()

model.fit(
    X_train,
    y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    use_best_model=True,
)

training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds!")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)

baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.median()))
baseline_rmse = root_mean_squared_error(y_test, np.full(len(y_test), y_train.mean()))

print("\n--- Model Evaluation ---")
print(f"Mean Absolute Error (MAE):  {mae:.2f} minutes (no-feature baseline: {baseline_mae:.2f})")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes (no-feature baseline: {baseline_rmse:.2f})")

model.save_model("train_delay_catboost.cbm")
print("\nModel saved to 'train_delay_catboost.cbm'")
