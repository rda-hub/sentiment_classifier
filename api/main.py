import os

from fastapi import FastAPI, Request
from transformers import pipeline
from loguru import logger
import time

os.makedirs("logs", exist_ok=True)
logger.add("logs/predictions.log", rotation="500 KB")

app = FastAPI()
classifier = pipeline("sentiment-analysis")


@app.post("/predict")
async def predict(request: Request):
    start = time.time()
    body = await request.json()
    text = body.get("text", "")
    result = classifier(text)[0]
    end = time.time()

    inference_time = end - start

    logger.info(
        f"Prediction | text='{text[:30]}' | label={result['label']} | score={result['score']:.4f} | time={inference_time:.3f}s"
    )

    return {
        "label": result["label"],
        "score": result["score"],
        "inference_time": inference_time
    }