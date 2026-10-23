"""
AI-Based Fraudulent Advertisement Detection System
Light-themed Streamlit dashboard.

Run from the project folder:
    python -m streamlit run dashboard.py
"""
import time
import html

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from features import StreamingFeatures
from model_utils import predict_one


st.set_page_config(
    page_title="AI-Based Fraudulent Advertisement Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- LIGHT THEME --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #10254a;
        --muted: #647a9e;
        --blue: #1769ff;
        --border: #dce8fb;
    }

    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #f2f7ff 52%, #ffffff 100%);
        color: var(--ink);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e0eaf8;
    }
    .block-container {
        max-width: 1700px;
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3, p, label { color: var(--ink); }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dce8fb;
        border-radius: 16px;
        padding: 17px 19px;
        box-shadow: 0 5px 18px rgba(39, 91, 160, .045);
        min-height: 115px;
    }
    [data-testid="stMetricLabel"] {
        color: #60779b !important;
        font-size: .88rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #10254a !important;
        font-weight: 800;
        font-size: clamp(1.35rem, 2.1vw, 2.15rem) !important;
        line-height: 1.15;
        overflow-wrap: anywhere;
    }
    [data-testid="stMetric"] {
        overflow: hidden;
    }
    [data-testid="stMetricDelta"] {
        overflow-wrap: anywhere;
    }
    @media (max-width: 900px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 20px; }
        .hero-title { font-size: 27px; }
    }
    .mini-label {
        color: #647a9e;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: .3px;
    }
    .hero-topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }
    .hero-badge {
        display: inline-block;
        background: #e9fbf3;
         color: #07885b;
        border: 1px solid #c5f1de;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        white-space: nowrap;
    }
    [data-testid="stMetricDelta"] { font-size: .78rem; }
    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid #d5e3f8;
        transition: all .18s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #1769ff, #438bff);
        color: white;
        border: none;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #1769ff;
        color: #1769ff;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #dce8fb;
        border-radius: 12px;
        overflow: hidden;
        background: white;
    }
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #fff;
        border-color: #d5e3f8;
        border-radius: 9px;
    }
    .hero {
        background: linear-gradient(115deg, #ffffff 0%, #eef5ff 70%, #f8f5ff 100%);
        border: 1px solid #dce8fb;
        border-radius: 20px;
        padding: 25px 28px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(39, 91, 160, .045);
    }
    .eyebrow {
        color: #1769ff;
        font-size: 11px;
        letter-spacing: 1.8px;
        font-weight: 800;
        text-transform: uppercase;
    }
    .hero-title {
        color: #10254a;
        font-size: clamp(25px, 3vw, 36px);
        line-height: 1.2;
        font-weight: 800;
        margin: 9px 0;
    }
    .hero-sub {
        color: #647a9e;
        font-size: 13px;
        line-height: 1.7;
    }
    .section-title {
        color: #10254a;
        font-size: 18px;
        font-weight: 750;
        margin: 8px 0 3px;
    }
    .section-sub {
        color: #7185a5;
        font-size: 12px;
        margin-bottom: 10px;
    }
    .status-pill {
        display: inline-block;
        background: #e9fbf3;
        color: #07885b;
        border: 1px solid #c5f1de;
        padding: 6px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }
    .note {
        border: 1px solid #d7e7ff;
        background: #eff6ff;
        border-radius: 12px;
        padding: 12px 15px;
        color: #315987;
        font-size: 12px;
        line-height: 1.65;
    }
    hr { border-color: #e2ebf8; }
    </style>
    """,
    unsafe_allow_html=True,
)
# -------------------- LOAD MODEL & EVENT DATA --------------------
@st.cache_resource
def load_assets():
    model_artifact = joblib.load("artifacts/model.joblib")
    with np.load("artifacts/events.npz") as archive:
        event_data = {key: archive[key].copy() for key in archive.files}
    return model_artifact, event_data


try:
    art, ev = load_assets()
except Exception as exc:
    st.error(f"Could not load the model or event data: {exc}")
    st.info("Check that artifacts/model.joblib and artifacts/events.npz exist in the project folder.")
    st.stop()

model = art["model"]
threshold = float(art["thr"])
test_start = int(art["test_start"])
model_name = str(art.get("model_name", "ML model"))

# -------------------- SIDEBAR CONTROLS --------------------
with st.sidebar:
    st.markdown("## 🛡️ FraudGuard")
    st.caption("AI-BASED AD FRAUD ANALYTICS")
    st.markdown('<span class="status-pill">● System ready</span>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### Replay settings")
    n_clicks = st.slider(
        "Clicks to replay", 1000, 50000, 10000, 1000, key="replay_click_count"
    )
    batch_size = st.slider(
        "Clicks per refresh", 10, 500, 100, 10, key="replay_batch_size"
    )
    delay = st.slider(
        "Delay per refresh (seconds)", 0.0, 1.0, 0.15, 0.05, key="replay_delay"
    )

    start = st.button(
        "▶  Start live replay", type="primary", use_container_width=True,
        key="start_replay_button"
    )

    if st.button("Clear saved results", use_container_width=True, key="clear_results_button"):
        st.session_state.pop("replay_results", None)
        st.session_state.pop("replay_ip_flags", None)
        st.rerun()

    st.divider()
    st.markdown("### Model status")
    st.caption(f"Model: {model_name}")
    st.caption(f"Threshold: {threshold:.6f}")
    st.caption("Stream: simulated test clicks")
    st.divider()
    st.caption("Smarter ads. Safer business.")
    st.caption("Flagged events are candidates for review, not confirmed fraud.")
    st.divider()
    st.markdown("### Quick guide")
    st.caption("1. Choose the number of clicks.")
    st.caption("2. Set batch size and replay delay.")
    st.caption("3. Start replay to generate results.")
    st.caption("4. Investigate flagged events and export CSV.")

# -------------------- HEADER --------------------
st.markdown(
    f"""
    <div class="hero">
        <div class="hero-topline">
            <div class="eyebrow">AI SECURITY ANALYTICS · CLICK RISK MONITORING</div>
            <div class="hero-badge">● Model loaded · {html.escape(model_name)}</div>
        </div>
        <div class="hero-title">AI-Based Fraudulent Advertisement Detection System</div>
        <div class="hero-sub">
            Monitor suspicious ad clicks · Score risk with machine learning ·
            Investigate events and export results
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="note"><b>Dataset note:</b> The TalkingData label means an app was downloaded after a click. '
    f'It is a proxy for genuine traffic, not a verified fraud label. Risk = 1 − P(genuine); '
    f'clicks are flagged when P(genuine) &lt; {threshold:.6f}.</div>',
    unsafe_allow_html=True,
)

# -------------------- REPLAY --------------------
if start:
    ts, ip, app_, dev, os_, ch, y = (
        ev[key] for key in ("ts", "ip", "app", "device", "os", "channel", "y")
    )
    end = min(len(ts), test_start + n_clicks)

    if end <= test_start:
        st.error("No clicks are available for replay in the test data.")
        st.stop()
        done = len(rows)
        flagged_total += batch_flagged
        batch_history.append({
            "Batch": len(batch_history) + 1,
            "Flagged clicks": batch_flagged,
            "Normal clicks": (batch_end - b0) - batch_flagged,
        })

        elapsed = max(time.perf_counter() - wall_start, 1e-9)
        p50, p95 = np.percentile(latencies, [50, 95])

        with metrics_slot.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total clicks", f"{done:,}")
            c2.metric("Suspicious clicks", f"{flagged_total:,}")
            c3.metric("Flagged rate", f"{100 * flagged_total / done:.2f}%")
            c4.metric("Throughput", f"{done / elapsed:,.0f} clicks/s")
            st.caption(f"Prediction latency · p50 {p50:.2f} ms · p95 {p95:.2f} ms")

        with chart_slot.container():
            chart_left, chart_right = st.columns([2, 1])
            with chart_left:
                st.markdown('<div class="section-title">Click activity trend</div>', unsafe_allow_html=True)
                st.line_chart(pd.DataFrame(batch_history).set_index("Batch"), height=240)
            with chart_right:
                st.markdown('<div class="section-title">Risk distribution</div>', unsafe_allow_html=True)
                current = pd.DataFrame(rows)
                st.bar_chart(current["risk"].value_counts(bins=8, sort=False).sort_index(), height=240)

        with recent_slot.container():
            st.markdown('<div class="section-title">Latest click events</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rows[-10:][::-1]), use_container_width=True, hide_index=True)

        progress.progress(done / (end - test_start), text=f"Processing {done:,} / {end-test_start:,} clicks")
        time.sleep(delay)

    results = pd.DataFrame(rows)
    st.session_state["replay_results"] = results
    st.session_state["replay_ip_flags"] = pd.DataFrame(
        sorted(ip_flags.items(), key=lambda item: -item[1]),
        columns=["ip", "flagged clicks"],
    )
    progress.progress(1.0, text="Replay complete")
    st.success("Replay complete! Results are available below.")

# -------------------- SUMMARY & INVESTIGATION --------------------
if "replay_results" in st.session_state:
    results = st.session_state["replay_results"].copy()
    flagged = results[results["status"] == "FLAGGED"]

    st.divider()
    st.markdown('<div class="section-title">Live activity summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">A quick overview of the selected replay. '
        'Flagged events should be reviewed before drawing conclusions.</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Clicks analysed", f"{len(results):,}")
    s2.metric("Flagged for review", f"{len(flagged):,}")
    s3.metric("Flagged percentage", f"{100 * len(flagged) / max(1, len(results)):.2f}%")
    s4.metric("Average risk score", f"{results['risk'].mean():.3f}")

    chart_a, chart_b = st.columns([3, 2])
    with chart_a:
        st.markdown('<div class="section-title">Risk score distribution</div>', unsafe_allow_html=True)
        st.bar_chart(results["risk"].value_counts(bins=10, sort=False).sort_index(), height=260)
    with chart_b:
        st.markdown('<div class="section-title">Top suspicious IP addresses</div>', unsafe_allow_html=True)
        ip_summary = st.session_state["replay_ip_flags"]
        if ip_summary.empty:
            st.info("No clicks were flagged during this replay.")
        else:
            st.dataframe(ip_summary.head(10), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-title">Click investigation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Search events, filter statuses, inspect risk scores and export results.</div>', unsafe_allow_html=True)

    tab_all, tab_flagged, tab_ips = st.tabs(["All clicks", "Flagged clicks", "IP analysis"])

    with tab_all:
        filter_a, filter_b = st.columns([2, 1])
        ip_search = filter_a.text_input("Search IP address", key="investigation_ip_search")
        status_filter = filter_b.selectbox(
            "Status filter", ["All", "FLAGGED", "NORMAL"], key="investigation_status_filter"
        )
        filtered = results.copy()
        if ip_search.strip():
            filtered = filtered[filtered["ip"].astype(str).str.contains(ip_search.strip(), regex=False)]
        if status_filter != "All":
            filtered = filtered[filtered["status"] == status_filter]

        st.caption(f"{len(filtered):,} matching records")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download filtered results (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="ad_click_fraud_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_all_results",
        )

    with tab_flagged:
        st.metric("Clicks flagged for investigation", f"{len(flagged):,}")
        st.dataframe(flagged, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇ Download flagged clicks (CSV)",
            data=flagged.to_csv(index=False).encode("utf-8"),
            file_name="flagged_clicks.csv",
            mime="text/csv",
            key="download_flagged_results",
        )

    with tab_ips:
        st.dataframe(
            st.session_state["replay_ip_flags"],
            use_container_width=True,
            hide_index=True,
        )
else:
    st.divider()
    st.markdown('<div class="section-title">Ready to monitor</div>', unsafe_allow_html=True)
    st.write("Click **Start live replay** in the sidebar to view the summary, charts, click explorer, and CSV exports.")

# -------------------- MODEL EVALUATION --------------------
import json
from pathlib import Path

st.divider()
st.header("Model Evaluation")
st.caption(
    "Performance metrics from the saved test report. "
    "The target label indicates an app download after a click, "
    "not confirmed advertising fraud."
)

report_path = Path(__file__).parent / "artifacts" / "train_report.json"

if report_path.exists():
    with open(report_path, "r", encoding="utf-8") as report_file:
        report = json.load(report_file)

    st.metric("Test clicks evaluated", f"{report['results']['GBDT (lightgbm)']['n']:,}")

    model_results = report.get("results", {})
    rows = []

    for model_name, metrics in model_results.items():
        rows.append({
            "Model": model_name,
            "PR-AUC": metrics.get("pr_auc"),
            "ROC-AUC": metrics.get("roc_auc"),
            "Flagged Click Share": metrics.get("flagged_share_of_clicks"),
            "Genuine Downloads Retained": metrics.get("genuine_kept_recall"),
        })

    evaluation_df = pd.DataFrame(rows)

    for column in [
        "PR-AUC",
        "ROC-AUC",
        "Flagged Click Share",
        "Genuine Downloads Retained",
    ]:
        evaluation_df[column] = evaluation_df[column].map(
            lambda value: f"{value:.2%}" if pd.notna(value) else "N/A"
        )

    st.dataframe(evaluation_df, use_container_width=True, hide_index=True)

    st.info(
        "PR-AUC is useful when genuine downloads are rare. "
        "A high percentage of flagged clicks that did not produce downloads "
        "does not prove those clicks were fraudulent."
    )
else:
    st.warning(
        "Training report not found. Expected file: artifacts/train_report.json"
    )
