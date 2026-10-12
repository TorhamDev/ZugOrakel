# ZugOrakel

ZugOrakel is a machine learning project that predicts how many minutes a train will be late on the German railway network (Deutsche Bahn), based on historical data. It consists of a data cleaning script, a CatBoost training script, a FastAPI backend and a small web frontend.

## How it works

1. **Clean** (`ai_model/clean_data.py`): reads the raw monthly parquet files and keeps one row per train stop with these features: station (`eva`, `station_name`), `train_type`, `hour`, `day_of_week`, `is_weekend` and `station_num_on_line` (the position of the stop on the train's route). The target is `delay_in_min` (early trains count as 0, capped at 120). Canceled stops and rows without a delay are dropped. Output: `cleaned_data.parquet`.
2. **Train** (`ai_model/train.py`): samples up to 10 million rows, trains a CatBoost regressor (RMSE loss, so it predicts the expected delay) and saves it as `train_delay_catboost.cbm`. At the end it prints MAE and RMSE next to a "no-feature" baseline so you can see how much the model actually learned.
3. **Serve** (`server.py`, `ai_model/predict.py`): FastAPI loads the model and exposes a `/predict` endpoint. It also serves the frontend (`frontend/index.html`) and the station list (`stations.json`).

## Project layout

```
ZugOrakel/
├── ai_model/
│   ├── assests/             # put the downloaded .parquet files here (the folder name is spelled like this in the code)
│   ├── clean_data.py        # raw parquet files -> cleaned_data.parquet
│   ├── train.py             # cleaned_data.parquet -> train_delay_catboost.cbm
│   └── predict.py           # loads the model and predicts a delay
├── frontend/index.html      # web UI
├── server.py                # FastAPI app
├── stations.json            # stations offered in the UI (name + EVA number)
├── pyproject.toml / uv.lock # dependencies
└── train_delay_catboost.cbm # trained model (created by train.py)
```

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- For training: disk space for the dataset files and several GB of RAM

## Setup

```bash
git clone <this repository>
cd ZugOrakel
uv sync
```

Without uv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install catboost "fastapi[standard]" pandas pyarrow scikit-learn
```

All commands below must be run from the project root, because the scripts use relative paths. Without uv, drop the `uv run` prefix and use the activated virtual environment.

## Run the web app

The app needs the trained model `train_delay_catboost.cbm` in the project root. If you do not have it yet, follow [Train your own model](#train-your-own-model) first.

```bash
uv run fastapi run server.py
```

Then open <http://127.0.0.1:8000>. For development with auto-reload use `uv run fastapi dev server.py`.

In the UI choose:

- **Station**: some names exist several times with different EVA numbers (for example the S-Bahn and the main-line platforms of a Hbf). The EVA number is shown in brackets, and the delay statistics differ between them.
- **Train type**: ICE, IC, EC, RE, RB or S-Bahn.
- **Station number on line**: the position of the stop on the train's route (1 = first stop).
- **Date and time**: the scheduled time. The model only uses the weekday and the hour.

### API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "station_name": "Frankfurt (Main) Hbf",
        "eva": "08000105",
        "train_type": "ICE",
        "station_num_on_line": 7,
        "datetime": "2026-10-16 18:15:00"
      }'
```

Response (the number is only an example):

```json
{
  "station_name": "Frankfurt (Main) Hbf",
  "train_type": "ICE",
  "datetime": "2026-10-16 18:15:00",
  "predicted_delay_minutes": 12.3
}
```

On errors the API returns HTTP 500 with an `error` and a `message` field.

## Train your own model

### 1. Download the dataset

The data comes from the Hugging Face dataset [piebro/deutsche-bahn-data](https://huggingface.co/datasets/piebro/deutsche-bahn-data) (CC BY 4.0, data by Deutsche Bahn).

1. Open the **Files** tab and go to the `monthly_processed_data/` folder.
2. Download one or more monthly files (`data-YYYY-MM.parquet`). Each file has around 14 to 16 million rows and is about 350 MB.
3. Put the files **directly** into `ai_model/assests/` (not in a sub-folder). The scripts read every `*.parquet` file in that folder.

With the Hugging Face CLI (`pip install -U huggingface_hub`) you can also do:

```bash
hf download piebro/deutsche-bahn-data --repo-type dataset \
  --include "monthly_processed_data/data-2026-05.parquet" --local-dir hf_download
mv hf_download/monthly_processed_data/*.parquet ai_model/assests/
```

The parquet files are ignored by git, so they are never committed.

### 2. Clean the data

```bash
uv run python ai_model/clean_data.py
```

This writes `cleaned_data.parquet`. The files are processed in batches, so it needs little memory.

### 3. Train the model

```bash
uv run python ai_model/train.py
```

This writes `train_delay_catboost.cbm` and prints the evaluation. Training uses all CPU cores. Settings at the top of `ai_model/train.py`:

- `MAX_ROWS` (default 10 million): rows sampled for training. Lower it if you run out of memory or want faster training; a few million rows are usually enough.
- `RAM_LIMIT` (default `8gb`): memory limit handed to CatBoost.

Restart the server afterwards so it loads the new model.

## Notes and limitations

- The model only knows the station, train type, position on the line, weekday and hour. It does not know the weather, construction sites, strikes or the real-time situation, so a prediction is the expected typical delay and individual trains can differ a lot. Compare with the baseline numbers printed by `train.py` to judge how much better than a constant guess the model is.
- Only stations listed in `stations.json` can be selected in the UI, and only the train types in the dropdown were checked against the data.

## License

MIT, see [LICENSE](LICENSE). The dataset is licensed separately (CC BY 4.0).

---

This README was written by AI. The frontend (`frontend/index.html`) was also generated by AI.
