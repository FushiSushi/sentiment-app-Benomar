
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict, MODELE

app = FastAPI(
    title="Sentiment Analysis API",
    description="API de prédiction du sentiment des avis clients.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    texte: str = Field(min_length=1, description="Avis client à analyser")


@app.get("/")
def accueil():
    return {
        "message": "Bienvenue dans l'API Sentiment Analysis",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODELE.exists(),
    }


@app.post("/predict")
def prediction(request: PredictionRequest):
    try:
        return predict(request.texte)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
