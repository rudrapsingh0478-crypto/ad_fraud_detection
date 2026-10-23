
"""
AI-Based Fraudulent Advertisement Detection System
Run: python -m streamlit run dashboard.py
"""
import html
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import StreamingFeatures
from model_utils import predict_one


# -------------------- CONFIGURATION --------------------
st.set_page_config(
    page_title="AI-Based Fraudulent Advertisement Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
EVENTS_PATH = ARTIFACTS / "events.npz"
REPORT_PATH = ARTIFACTS / "train_report.json"


# -------------------- LIGHT THEME --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg,#f8fbff,#f2f7ff,#ffffff);
    color: #10254a;
    font-family: Inter, sans-serif;
}
[data-testid="stSidebar"] {
    background: white;
    border-right: 1px solid #dce8fb;
}
.hero {
    padding: 25px;
    margin-bottom: 18px;
    border: 1px solid #dce8fb;
    border-radius: 18px;
    background: linear-gradient(120deg,#ffffff,#edf4ff);
}
.eyebrow {
    color: #1769ff;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.3px;
}
.hero-title {
    color: #10254a;
    font-size: clamp(25px,3vw,36px);
    font-weight: 800;
    margin: 12px 0;
}
.hero-sub,.section-sub {
    color: #647a9e;
    font-size: 13px;
    line-height: 1.7;
}
.section-title {
    color: #10254a;
    font-size: 18px;
    font-weight: 750;
    margin: 10px 0 4px;
}
.note {
    border: 1px solid #d7e7ff;
    background: #eff6ff;
    border-radius: 12px;
    padding: 13px;
    color: #315987;
    font-size: 12px;
    line-height: 1.7;
}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.85);
    border: 1px solid #dce8fb;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL AND DATA --------------------
@st.cache_resource
def load_assets():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"Event data not found: {EVENTS_PATH}")

    artifact = joblib.load(MODEL_PATH)

    with np.load(EVENTS_PATH, allow_pickle=False) as data:
        events = {key: data[key].copy() for key in data.files}

    required = {"ts", "ip", "app", "device", "os", "channel", "y"}
    missing = required - set(events)
    if missing:
        raise ValueError(f"Missing event arrays: {sorted(missing)}")

    lengths = {len(events[key]) for key in required}
    if len(lengths) != 1:
        raise ValueError("Event arrays have different lengths.")

    return artifact, events


try:
    art, ev = load_assets()
    model = art["model"]
    threshold = float(art["thr"])
    test_start = int(art["test_start"])
    model_name = str(art.get("model_name", "ML model"))
except Exception as exc:
    st.error(f"Could not load the model or data: {exc}")
    st.info(
        "Check artifacts/model.joblib and artifacts/events.npz "
        "inside your project folder."
    )
    st.stop()


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.caption("AI-BASED AD FRAUD ANALYTICS")
    st.success("System ready")
    st.divider()

    st.markdown("### Replay settings")
    n_clicks = st.slider(
        "Clicks to replay", 1000, 50000, 10000, 1000
    )
    batch_size = st.slider(
        "Clicks per refresh", 10, 500, 100, 10
    )
    delay = st.slider(
        "Delay per refresh (seconds)", 0.0, 1.0, 0.15, 0.05
    )

    start = st.button(
        "▶ Start live replay",
        type="primary",
        use_container_width=True,
    )

    if st.button("Clear saved results", use_container_width=True):
        st.session_state.pop("replay_results", None)
        st.session_state.pop("replay_ip_flags", None)
        st.rerun()

    st.divider()
    st.markdown("### Model status")
    st.caption(f"Model: {model_name}")
    st.caption(f"Threshold: {threshold:.6f}")
    st.caption("Stream: simulated test clicks")
    st.caption("Flagged events need human review.")


# -------------------- HEADER --------------------
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">AI SECURITY ANALYTICS · CLICK RISK MONITORING</div>
  <div class="hero-title">
    AI-Based Fraudulent Advertisement Detection System
  </div>
  <div class="hero-sub">
    Monitor suspicious ad clicks · Machine-learning risk scores ·
    Investigate events · Export results
  </div>
  <div class="hero-sub">Loaded model: {html.escape(model_name)}</div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="note">
<b>Dataset note:</b> TalkingData's label indicates whether an app
download followed a click. It is not a verified fraud label.
Risk = 1 − P(download after click). A click is flagged when its
predicted probability is below {threshold:.6f}.
Flagged events are candidates for review, not confirmed fraud.
</div>
""", unsafe_allow_html=True)


# -------------------- LIVE REPLAY --------------------
if start:
    ts, ip, app_, dev, os_, ch, labels = (
        ev[key]
        for key in ("ts", "ip", "app", "device", "os", "channel", "y")
    )

    first = max(0, min(test_start, len(ts)))
    end = min(len(ts), first + n_clicks)

    if end <= first:
        st.error("No clicks are available for replay in the test data.")
        st.stop()

    progress = st.progress(0.0, text="Preparing replay...")
    metric_area = st.empty()
    chart_area = st.empty()
    recent_area = st.empty()

    rows = []
    batch_history = []
    ip_flags = {}
    flagged_total = 0
    latencies = []
    wall_start = time.perf_counter()

    feature_engine = StreamingFeatures()

    # Build past-only streaming state before the selected test segment.
    for i in range(first):
        feature_engine.step(
            ts[i], ip[i], app_[i], dev[i], os_[i], ch[i]
        )

    for batch_start in range(first, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        batch_flagged = 0

        for i in range(batch_start, batch_end):
            feats = feature_engine.step(
                ts[i], ip[i], app_[i], dev[i], os_[i], ch[i]
            )

            tick = time.perf_counter()
            probability = float(predict_one(model, feats))
            latency_ms = (time.perf_counter() - tick) * 1000
            latencies.append(latency_ms)

            probability = float(np.clip(probability, 0.0, 1.0))
            risk = 1.0 - probability
            is_flagged = probability < threshold

            if is_flagged:
                batch_flagged += 1
                ip_key = int(ip[i])
                ip_flags[ip_key] = ip_flags.get(ip_key, 0) + 1

            rows.append({
                "timestamp": int(ts[i]),
                "ip": int(ip[i]),
                "app": int(app_[i]),
                "device": int(dev[i]),
                "os": int(os_[i]),
                "channel": int(ch[i]),
                "actual_label": int(labels[i]),
                "genuine_probability": round(probability, 4),
                "risk": round(risk, 4),
                "status": "FLAGGED" if is_flagged else "NORMAL",
                "latency_ms": round(latency_ms, 3),
            })

        done = len(rows)
        flagged_total += batch_flagged
        batch_history.append({
            "Batch": len(batch_history) + 1,
            "Flagged clicks": batch_flagged,
            "Normal clicks": batch_end - batch_start - batch_flagged,
        })

        elapsed = max(time.perf_counter() - wall_start, 1e-9)
        p50, p95 = np.percentile(latencies, [50, 95])

        with metric_area.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total clicks", f"{done:,}")
            c2.metric("Suspicious clicks", f"{flagged_total:,}")
            c3.metric(
                "Flagged rate",
                f"{100 * flagged_total / done:.2f}%"
            )
            c4.metric("Throughput", f"{done / elapsed:,.0f} clicks/s")
            st.caption(
                f"Prediction latency · p50 {p50:.2f} ms · "
                f"p95 {p95:.2f} ms"
            )

        current = pd.DataFrame(rows)

        with chart_area.container():
            left, right = st.columns([2, 1])
            with left:
                st.markdown(
                    '<div class="section-title">Click activity trend</div>',
                    unsafe_allow_html=True,
                )
                st.line_chart(
                    pd.DataFrame(batch_history).set_index("Batch"),
                    height=240,
                )
            with right:
                st.markdown(
                    '<div class="section-title">Risk distribution</div>',
                    unsafe_allow_html=True,
                )
                st.bar_chart(
                    current["risk"].value_counts(
                        bins=8, sort=False
                    ).sort_index(),
                    height=240,
                )

        with recent_area.container():
            st.markdown(
                '<div class="section-title">Latest click events</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(
                current.tail(10).iloc[::-1],
                use_container_width=True,
                hide_index=True,
            )

        progress.progress(
            done / (end - first),
            text=f"Processing {done:,} / {end - first:,} clicks",
        )

        if delay > 0:
            time.sleep(delay)

    st.session_state["replay_results"] = pd.DataFrame(rows)
    st.session_state["replay_ip_flags"] = pd.DataFrame(
        sorted(ip_flags.items(), key=lambda item: -item[1]),
        columns=["ip", "flagged clicks"],
    )
    progress.progress(1.0, text="Replay complete")
    st.success("Replay complete! Results are available below.")


# -------------------- SUMMARY AND INVESTIGATION --------------------
if "replay_results" in st.session_state:
    results = st.session_state["replay_results"].copy()
    flagged = results[results["status"] == "FLAGGED"]

    st.divider()
    st.markdown(
        '<div class="section-title">Live activity summary</div>',
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)
    a.metric("Clicks analysed", f"{len(results):,}")
    b.metric("Flagged for review", f"{len(flagged):,}")
    c.metric(
        "Flagged percentage",
        f"{100 * len(flagged) / max(1, len(results)):.2f}%"
    )
    d.metric("Average risk score", f"{results['risk'].mean():.3f}")

    st.markdown(
        '<div class="section-title">Investigate flagged clicks</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        status_filter = st.selectbox(
            "Event status",
            ["FLAGGED only", "All events", "NORMAL only"],
        )
    with f2:
        min_risk = st.slider(
            "Minimum risk", 0.0, 1.0, 0.0, 0.05
        )
    with f3:
        ip_search = st.text_input(
            "Filter by IP address",
            placeholder="Enter an IP value",
        )

    view = results.copy()
    if status_filter == "FLAGGED only":
        view = view[view["status"] == "FLAGGED"]
    elif status_filter == "NORMAL only":
        view = view[view["status"] == "NORMAL"]

    view = view[view["risk"] >= min_risk]
    if ip_search.strip():
        view = view[
            view["ip"].astype(str).str.contains(
                ip_search.strip(), na=False
            )
        ]

    st.dataframe(
        view.sort_values("risk", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Export replay results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="ad_click_risk_results.csv",
        mime="text/csv",
    )

    st.markdown(
        '<div class="section-title">Top IPs by flagged clicks</div>',
        unsafe_allow_html=True,
    )
    ip_summary = st.session_state.get(
        "replay_ip_flags",
        pd.DataFrame(columns=["ip", "flagged clicks"]),
    )

    if ip_summary.empty:
        st.info("No clicks were flagged in this replay.")
    else:
        st.dataframe(
            ip_summary.head(20),
            use_container_width=True,
            hide_index=True,
        )
        st.bar_chart(
            ip_summary.head(15).set_index("ip"),
            height=280,
        )
else:
    st.info(
        "Choose replay settings in the sidebar and click "
        "**Start live replay** to begin."
    )


# -------------------- MODEL EVALUATION --------------------
st.divider()
st.markdown(
    '<div class="section-title">Model evaluation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-sub">Metrics from artifacts/train_report.json.</div>',
    unsafe_allow_html=True,
)

if REPORT_PATH.exists():
    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file:
            report = json.load(file)

        model_results = report.get("results", {})
        model_names = [
            "GBDT (lightgbm)",
            "Logistic regression",
            "Heuristic rule (IP clicks/10min)",
        ]

        evaluation_rows = []
        for name in model_names:
            item = model_results.get(name)
            if item:
                evaluation_rows.append({
                    "Model": name,
                    "Test clicks": item.get("n", "—"),
                    "PR-AUC": item.get("pr_auc", "—"),
                    "ROC-AUC": item.get("roc_auc", "—"),
                    "Flagged share": item.get("flagged_share", "—"),
                    "Genuine kept recall": item.get(
                        "genuine_kept_recall", "—"
                    ),
                })

        if evaluation_rows:
            st.dataframe(
                pd.DataFrame(evaluation_rows),
                use_container_width=True,
                hide_index=True,
            )

            gbdt = model_results.get("GBDT (lightgbm)", {})
            if gbdt:
                x, y, z = st.columns(3)
                n = gbdt.get("n")
                x.metric(
                    "Test clicks evaluated",
                    f"{n:,}" if isinstance(n, (int, float)) else str(n or "—"),
                )
                pr_auc = gbdt.get("pr_auc")
                roc_auc = gbdt.get("roc_auc")
                y.metric(
                    "GBDT PR-AUC",
                    f"{pr_auc:.4f}" if isinstance(pr_auc, (int, float)) else "—",
                )
                z.metric(
                    "GBDT ROC-AUC",
                    f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else "—",
                )
        else:
            st.warning("No recognized model results were found in the report.")

    except Exception as exc:
        st.error(f"Could not read the evaluation report: {exc}")
else:
    st.warning("Evaluation report not found: artifacts/train_report.json")

st.caption(
    "Important: TalkingData's is_attributed label records whether an app "
    "download followed a click. It is not a verified fraud label. "
    "These metrics measure conversion prediction, not confirmed fraud detection."
)

