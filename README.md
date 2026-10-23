# Real-time Ad Click Fraud Scoring Demo

Streaming pipeline + API + live dashboard on the TalkingData AdTracking data.

## Setup (10 min)
```bash
pip install -r requirements.txt
```
Download `train.csv` from Kaggle (needs a Kaggle account and accepting the competition rules):
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
Put `train.csv` (or `train.csv.zip`) in `data/`.

## Run order
```bash
python train.py --data data/train.csv --nrows 1000000   # ~2-5 min. Prints the results table
python simulate.py --n 50000                            # latency / throughput report
uvicorn api:app --port 8000                             # API  -> http://localhost:8000/docs
streamlit run dashboard.py                              # live dashboard
```
Quick API test:
```bash
curl -X POST localhost:8000/score -H "Content-Type: application/json" \
  -d '{"ip":5348,"app":12,"device":1,"os":13,"channel":497,"click_time":"2017-11-07 09:30:00"}'
```
No Kaggle data yet? `python gen_synthetic.py` makes a FAKE file to check the code runs.
Never report numbers from it.

If your laptop has little RAM, use `--nrows 500000`.

## What the demo does
1. Streaming features per click, computed from past events only (counts of clicks per IP, IP+app, IP+device+OS, IP+channel in 1/10/60 min; distinct apps/channels per IP; time since last click).
2. LightGBM scores P(genuine conversion). Low P = suspicious.
3. Time-ordered 70/15/15 split (no random shuffling, no future leakage).
4. Compared against logistic regression and a one-line heuristic rule.
5. Threshold chosen on validation to keep 95% of genuine conversions; test shows what share of traffic that flags.
6. Replay of held-out clicks as a live stream with measured latency.

## Say this honestly during the presentation
- The label is "app downloaded after click" (is_attributed), NOT a verified fraud label. We treat "no download" as a proxy.
- Only ~0.2-0.3% of clicks are positive, so accuracy is meaningless. Use PR-AUC (random baseline = genuine rate), ROC-AUC, flagged share and genuine-kept recall.
- The stream is a replay of historical data, not live ad traffic.
- Results depend on your sample (first N rows by time). Don't quote numbers until you've run it.

## Files
features.py (streaming features) | train.py | simulate.py | api.py | dashboard.py | model_utils.py | data_utils.py | gen_synthetic.py
