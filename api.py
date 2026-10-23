"""
Real-time scoring API.   Run:  uvicorn api:app --port 8000
Docs/test UI:  http://localhost:8000/docs

Note: state starts EMPTY (cold start). Features use up to 1 hour of history, so scores
become fully reliable after the stream has run for ~1 hour (or after you replay history).
"""
import threading, time
from datetime import datetime, timezone
from typing import Optional

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from features import StreamingFeatures
from model_utils import predict_one

art = joblib.load("artifacts/model.joblib")
MODEL, THR = art["model"], art["thr"]

app = FastAPI(title="Real-time Click Fraud Scorer (demo)")
state = StreamingFeatures()
lock = threading.Lock()
stats = {"scored": 0, "flagged": 0}


class Click(BaseModel):
    ip: int
    app: int
    device: int
    os: int
    channel: int
    click_time: Optional[str] = None   # "YYYY-MM-DD HH:MM:SS" (UTC). Default: now.


def _ts(s):
    if not s:
        return time.time()
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


@app.post("/score")
def score(c: Click):
    t0 = time.perf_counter()
    with lock:
        feats = state.step(_ts(c.click_time), c.ip, c.app, c.device, c.os, c.channel)
        p = predict_one(MODEL, feats)
        flagged = p < THR
        stats["scored"] += 1
        stats["flagged"] += int(flagged)
    return {"p_genuine": round(p, 6), "risk_score": round(1 - p, 6), "flagged": bool(flagged),
            "threshold": round(THR, 6), "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
            "features": dict(zip(art["feature_names"], feats))}


@app.get("/stats")
def get_stats():
    return stats


@app.post("/reset")
def reset():
    global state
    with lock:
        state = StreamingFeatures()
        stats.update(scored=0, flagged=0)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok", "model": art["model_name"]}
