from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

_model = None
MODEL_PATH = "train_delay_catboost.cbm"


def _load_model():
    global _model
    if _model is None:
        _model = CatBoostRegressor(verbose=0)
        model_path = Path(MODEL_PATH)
        if model_path.exists():
            _model.load_model(str(model_path))
        else:
            print(
                f"Warning: Model file not found at {MODEL_PATH}. Using fallback predictions."
            )
            _model = None
    return _model


def predict_delay(
    station_name,
    eva,
    train_type,
    station_num_on_line,
    datetime_str,
):
    dt = pd.to_datetime(datetime_str)
    input_data = pd.DataFrame(
        [
            {
                "eva": str(eva),
                "station_name": str(station_name),
                "train_type": str(train_type),
                "hour": dt.hour,
                "day_of_week": dt.dayofweek,
                "is_weekend": int(dt.dayofweek in [5, 6]),
                "station_num_on_line": station_num_on_line,
            }
        ]
    )

    model = _load_model()

    print("pridcting.........")

    assert model is not None # yea i know :D

    predicted_delay = model.predict(input_data)[0]
    return max(0, round(predicted_delay, 1))
