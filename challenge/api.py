import os

import fastapi
import pandas as pd

from typing import List
from pydantic import BaseModel, field_validator
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from challenge.model import DelayModel

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "data.csv")
_data = pd.read_csv(DATA_PATH, usecols=["OPERA"])
VALID_OPERA = _data["OPERA"].unique().tolist()

app = fastapi.FastAPI()
model = DelayModel()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @field_validator("OPERA")
    @classmethod
    def validate_opera(cls, v):
        if v not in VALID_OPERA:
            raise ValueError(f"Invalid airline: {v}")
        return v

    @field_validator("TIPOVUELO")
    @classmethod
    def validate_tipovuelo(cls, v):
        if v not in ("I", "N"):
            raise ValueError(f"Invalid flight type: {v}")
        return v

    @field_validator("MES")
    @classmethod
    def validate_mes(cls, v):
        if v < 1 or v > 12:
            raise ValueError(f"Invalid month: {v}")
        return v


class PredictRequest(BaseModel):
    flights: List[FlightData]


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    data = pd.DataFrame([flight.model_dump() for flight in request.flights])
    features = model.preprocess(data=data)
    predictions = model.predict(features=features)
    return {"predict": predictions}
