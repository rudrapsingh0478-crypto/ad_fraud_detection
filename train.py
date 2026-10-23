"""
Train + evaluate with a strict TIME-ORDERED split (70/15/15). Streaming features are
computed once over the whole sorted sequence (state carries over, exactly like production).

Framing (be honest in your demo): TalkingData's label is 'app downloaded after click'
(is_attributed), not a verified fraud label. We predict P(genuine conversion);
low P = suspicious. Metrics are for the RARE positive class (genuine conversion).
"""
import argparse, json, os, time
import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from features import StreamingFeatures, FEATURE_NAMES, NUMERIC_IDX
from data_utils import load_clicks
from model_utils import fit_gbdt, predict_proba_many, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/train.csv")
ap.add_argument("--nrows", type=int, default=1_000_000)
ap.add_argument("--keep", type=float, default=0.95, help="share of genuine conversions to keep")
a = ap.parse_args()
os.makedirs("artifacts", exist_ok=True)

print(f"[1/5] loading {a.data} (nrows={a.nrows}) ...")
ev = load_clicks(a.data, a.nrows)
n = len(ev["ts"])
print(f"      {n:,} clicks, genuine rate {ev['y'].mean():.4%}, "
      f"span {(ev['ts'][-1]-ev['ts'][0])/3600:.1f} h")

print("[2/5] streaming feature extraction (causal) ...")
t0 = time.time()
sf = StreamingFeatures()
X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
for i in range(n):
    X[i] = sf.step(int(ev["ts"][i]), int(ev["ip"][i]), int(ev["app"][i]),
                   int(ev["device"][i]), int(ev["os"][i]), int(ev["channel"][i]))
print(f"      {time.time()-t0:.1f}s  ({n/(time.time()-t0):,.0f} clicks/s)")
y = ev["y"]

i1, i2 = int(0.70 * n), int(0.85 * n)
Xtr, ytr, Xva, yva, Xte, yte = X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:]
print(f"[3/5] time split  train={len(ytr):,}  val={len(yva):,}  test={len(yte):,}")
print(f"      genuine in train/val/test: {ytr.sum()}/{yva.sum()}/{yte.sum()}")

print("[4/5] training models ...")
gbdt, name = fit_gbdt(Xtr, ytr, Xva, yva)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
lr.fit(np.log1p(np.clip(Xtr[:, NUMERIC_IDX], 0, None)), ytr)

def p_lr(Xm): return lr.predict_proba(np.log1p(np.clip(Xm[:, NUMERIC_IDX], 0, None)))[:, 1]
def p_rule(Xm): return 1.0 / (1.0 + Xm[:, 1])            # fewer clicks from IP in 10 min -> more genuine

scorers = {f"GBDT ({name})": lambda Xm: predict_proba_many(gbdt, Xm),
           "Logistic regression": p_lr,
           "Heuristic rule (IP clicks/10min)": p_rule}

print("[5/5] evaluating on the untouched TEST split ...")
results = {}
for label, fn in scorers.items():
    pva = fn(Xva)
    thr = float(np.quantile(pva[yva == 1], 1 - a.keep)) if yva.sum() >= 20 else float(np.quantile(pva, 0.5))
    results[label] = evaluate(yte, fn(Xte), thr)
    results[label]["threshold"] = thr

print(f"\n{'Model':38s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'flagged%':>9s} {'genuine kept':>13s}")
for k, r in results.items():
    print(f"{k:38s} {r['pr_auc']:8.4f} {r['roc_auc']:8.4f} {r['flagged_share_of_clicks']:9.2%} {r['genuine_kept_recall']:13.2%}")
print(f"\nPR-AUC of a random scorer = genuine rate = {yte.mean():.4%}  (compare PR-AUC against this, NOT against 1.0)")

gkey = f"GBDT ({name})"
joblib.dump({"model": gbdt, "model_name": name, "thr": results[gkey]["threshold"],
             "feature_names": FEATURE_NAMES, "val_start": i1, "test_start": i2}, "artifacts/model.joblib")
np.savez("artifacts/events.npz", **ev)
with open("artifacts/train_report.json", "w") as f:
    json.dump({"rows": n, "split": [i1, i2], "results": results}, f, indent=2)
print("saved artifacts/model.joblib, events.npz, train_report.json")
