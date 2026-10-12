from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ai_model.predict import predict_delay

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return FileResponse("frontend/index.html")


@app.get("/stations.json")
def get_stations():
    return FileResponse("stations.json")


class PredictRequest(BaseModel):
    """Request model for prediction endpoint"""

    station_name: str
    eva: str
    train_type: str
    station_num_on_line: int
    datetime: str


@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict train delay for a given station and time.

    Expected input JSON:
    {
        "station_name": "Leipzig Hbf",
        "eva": "08000206",
        "train_type": "ICE",
        "station_num_on_line": 5,
        "datetime": "2026-10-09 17:45:00"
    }

    Returns predicted delay in minutes.
    """
    try:
        predicted_delay_minutes = predict_delay(
            station_name=request.station_name,
            eva=request.eva,
            train_type=request.train_type,
            station_num_on_line=request.station_num_on_line,
            datetime_str=request.datetime,
        )
        return {
            "station_name": request.station_name,
            "train_type": request.train_type,
            "datetime": request.datetime,
            "predicted_delay_minutes": predicted_delay_minutes,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "message": "Failed to generate prediction"},
        )
