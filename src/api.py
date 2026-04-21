from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import SMAP_CHANNEL, WINDOW_SIZE, ALERT_LOG_PATH, STREAM_INTERVAL_SEC
from src.detector import AnomalyDetector
from src.simulator import SensorStreamSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_detector: Optional[AnomalyDetector] = None
_simulator: Optional[SensorStreamSimulator] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _detector, _simulator
    _detector = AnomalyDetector(SMAP_CHANNEL)
    _simulator = SensorStreamSimulator(SMAP_CHANNEL, interval_sec=STREAM_INTERVAL_SEC)
    yield

app = FastAPI(
    title="Real-time Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class PredictRequest(BaseModel):
    window: list[list[float]]
    run_shap: bool = True

class PredictResponse(BaseModel):
    timestamp: str
    is_anomaly: bool
    ensemble_score: float
    if_score: float
    ocsvm_score: float
    lstm_score: float
    if_pred: int
    ocsvm_pred: int
    lstm_pred: int
    votes: int
    top_features: list[dict]
    channel_errors: list[float]

@app.get("/health")
async def health():
    if _detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _detector.health()

@app.get("/model-info")
async def model_info():
    if _detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _detector.model_info()

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if _detector is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    window = np.array(req.window, dtype=np.float32)
    if window.shape[0] != WINDOW_SIZE:
        raise HTTPException(status_code=422, detail=f"Window must have {WINDOW_SIZE} timesteps")
    if window.ndim != 2:
        raise HTTPException(status_code=422, detail="Window must be 2D")

    result = _detector.predict(window, run_shap=req.run_shap)
    return PredictResponse(**result.__dict__)

@app.get("/stream")
async def stream_detections():
    if _detector is None or _simulator is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    async def event_generator():
        gen = _simulator.stream()
        loop = asyncio.get_event_loop()
        try:
            while True:
                window, meta = await loop.run_in_executor(None, next, gen)
                result = await loop.run_in_executor(None, _detector.predict, window, False)
                payload = {**result.__dict__, "stream_meta": meta}
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(STREAM_INTERVAL_SEC * 0.1)
        except StopIteration:
            yield "data: {\"event\": \"stream_end\"}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/alerts")
async def get_alerts(last_n: int = 50):
    if not ALERT_LOG_PATH.exists():
        return {"alerts": [], "total": 0}
    lines = [l for l in ALERT_LOG_PATH.read_text().strip().split("\n") if l.strip()]
    alerts = [json.loads(l) for l in lines[-last_n:]]
    return {"alerts": alerts[::-1], "total": len(lines)}

@app.get("/")
async def root():
    return {"service": "Real-time Anomaly Detection API", "version": "1.0.0", "docs": "/docs"}

