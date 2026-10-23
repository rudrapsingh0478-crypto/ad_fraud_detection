import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from features import CATEGORICAL_IDX


def fit_gbdt(Xtr, ytr, Xva, yva):
    """LightGBM if installed (recommended); otherwise sklearn fallback (smoke tests only)."""
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=50, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
              categorical_feature=CATEGORICAL_IDX,
              callbacks=[lgb.early_stopping(40, verbose=False)])
        return m, "lightgbm"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           early_stopping=True, random_state=42)
        m.fit(Xtr, ytr)
        return m, "sklearn-HistGB (fallback, install lightgbm)"


def predict_proba_many(model, X):
    return model.predict_proba(X)[:, 1]


def predict_one(model, feats):
    x = np.asarray([feats], dtype=np.float32)
    if hasattr(model, "booster_"):          # LightGBM: much lower overhead
        return float(model.booster_.predict(x)[0])
    return float(model.predict_proba(x)[0, 1])


def evaluate(y, p, thr):
    """
    Positive class = genuine conversion (is_attributed = 1, the RARE class).
    A click is FLAGGED (suspicious) when p(genuine) < thr.
    thr is chosen on VALIDATION so that 95% of genuine conversions are kept.
    """
    y = np.asarray(y); p = np.asarray(p)
    flagged = p < thr
    out = {
        "n": int(len(y)),
        "genuine_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "flagged_share_of_clicks": float(flagged.mean()),
        "genuine_kept_recall": float((~flagged[y == 1]).mean()) if y.sum() else None,
        "flagged_non_download_precision": float(1 - y[flagged].mean()) if flagged.any() else None,
    }
    return out
