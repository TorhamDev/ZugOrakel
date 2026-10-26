from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ASSETS_DIR = Path("ai_model/assests")
OUTPUT_FILE = "cleaned_data.parquet"
BATCH_SIZE = 500_000

COLUMNS = [
    "eva",
    "station_name",
    "train_type",
    "arrival_is_canceled",
    "departure_is_canceled",
    "arrival_planned_time",
    "departure_planned_time",
    "train_line_station_num",
    "delay_in_min",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    is_arrival_canceled = df["arrival_is_canceled"].fillna(False).astype(bool)
    is_departure_canceled = df["departure_is_canceled"].fillna(False).astype(bool)
    has_departure = df["departure_planned_time"].notna()
    is_canceled = is_departure_canceled.where(has_departure, is_arrival_canceled)
    df = df[~is_canceled].dropna(subset=["delay_in_min"])

    scheduled_time = df["departure_planned_time"].fillna(df["arrival_planned_time"])
    day_of_week = scheduled_time.dt.dayofweek

    return pd.DataFrame(
        {
            "eva": df["eva"],
            "station_name": df["station_name"],
            "train_type": df["train_type"],
            "hour": scheduled_time.dt.hour.astype("Int8"),
            "day_of_week": day_of_week.astype("Int8"),
            "is_weekend": day_of_week.isin([5, 6]).astype("int8"),
            "station_num_on_line": df["train_line_station_num"],
            "delay_in_min": df["delay_in_min"].clip(lower=0).astype("float32"),
        }
    )


files = sorted(ASSETS_DIR.glob("*.parquet"))
if not files:
    raise FileNotFoundError(f"No .parquet files found in {ASSETS_DIR}")

input_schema = pq.read_schema(files[0])
schema = pa.schema(
    [
        input_schema.field("eva"),
        input_schema.field("station_name"),
        input_schema.field("train_type"),
        pa.field("hour", pa.int8()),
        pa.field("day_of_week", pa.int8()),
        pa.field("is_weekend", pa.int8()),
        input_schema.field("train_line_station_num").with_name("station_num_on_line"),
        pa.field("delay_in_min", pa.float32()),
    ]
)

rows = 0
with pq.ParquetWriter(OUTPUT_FILE, schema) as writer:
    for file in files:
        print(f"Processing {file.name}...")
        for batch in pq.ParquetFile(file).iter_batches(
            batch_size=BATCH_SIZE, columns=COLUMNS
        ):
            cleaned = clean(batch.to_pandas())
            writer.write_table(
                pa.Table.from_pandas(cleaned, schema=schema, preserve_index=False)
            )
            rows += len(cleaned)

print(f"Wrote {rows} rows to {OUTPUT_FILE}")
