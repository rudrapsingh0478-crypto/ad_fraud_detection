"""Replay the held-out TEST clicks in time order as a live stream and measure latency/throughput."""
import argparse, json, time
import numpy as np, joblib
from features import StreamingFeatures
from model_utils import predict_one, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50_000, help="number of test clicks to replay")
a = ap.parse_args()

art = joblib.load("artifacts/model.joblib")
ev = np.load("artifacts/events.npz")
ts, ip, app, dev, os_, ch, y = (ev[k] for k in ("ts", "ip", "app", "device", "os", "channel", "y"))
model, thr, t_start = art["model"], art["thr"], art["test_start"]

# state needs only the last hour of history (all windows <= 3600 s)
warm = int(np.searchsorted(ts, ts[t_start] - 3600))
sf = StreamingFeatures()
for i in range(warm, t_start):
    sf.step(int(ts[i]), int(ip[i]), int(app[i]), int(dev[i]), int(os_[i]), int(ch[i]))
print(f"warm-up: replayed {t_start-warm:,} clicks (last hour before test)")

end = min(len(ts), t_start + a.n)
lat, scores = [], []
wall0 = time.perf_counter()
for i in range(t_start, end):
    t0 = time.perf_counter_ns()
    f = sf.step(int(ts[i]), int(ip[i]), int(app[i]), int(dev[i]), int(os_[i]), int(ch[i]))
    p = predict_one(model, f)
    lat.append((time.perf_counter_ns() - t0) / 1e6)
    scores.append(p)
wall = time.perf_counter() - wall0

lat = np.array(lat)
rep = {"clicks_replayed": int(len(lat)),
       "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                      "p99": float(np.percentile(lat, 99)), "max": float(lat.max())},
       "throughput_clicks_per_sec": float(len(lat) / wall),
       "quality_on_replayed_slice": evaluate(y[t_start:end], np.array(scores), thr)}
print(json.dumps(rep, indent=2))
json.dump(rep, open("artifacts/sim_report.json", "w"), indent=2)
