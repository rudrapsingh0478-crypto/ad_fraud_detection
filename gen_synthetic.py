"""
SMOKE-TEST ONLY. Generates a fake file in TalkingData format so you can verify the
pipeline runs end-to-end before downloading the real Kaggle data.
Results on this file mean NOTHING (labels come from my own generator).
Never report these numbers.
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=200_000)
ap.add_argument("--out", default="data/synthetic_train.csv")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

T0 = 1509926400  # 2017-11-06 00:00:00 UTC
SPAN = 4 * 86400
n_bot = int(a.rows * 0.35)
n_norm = a.rows - n_bot

# normal traffic
t_n = rng.integers(T0, T0 + SPAN, n_norm)
ip_n = rng.zipf(1.6, n_norm) % 30000 + 100000
app_n = rng.zipf(1.5, n_norm) % 300
ch_n = rng.integers(0, 200, n_norm)
dev_n = rng.choice([1, 2, 0, 3], n_norm, p=[.9, .05, .03, .02])
os_n = rng.integers(0, 40, n_norm)
y_n = (rng.random(n_norm) < 0.03).astype(np.int8)

# bursty bot traffic: few IPs, tight bursts, same app/channel, almost never convert
bot_ips = rng.integers(500000, 500300, 300)
t_b, ip_b, app_b, ch_b = [], [], [], []
left = n_bot
while left > 0:
    size = int(min(left, rng.integers(10, 60)))
    ip = rng.choice(bot_ips); c = T0 + rng.integers(0, SPAN)
    t_b += list(c + rng.integers(0, 60, size)); ip_b += [ip] * size
    app_b += [int(rng.integers(0, 4))] * size; ch_b += [int(rng.integers(0, 6))] * size
    left -= size
t_b = np.array(t_b); n_bot = len(t_b)
dev_b = np.ones(n_bot, int); os_b = rng.integers(0, 3, n_bot)
y_b = (rng.random(n_bot) < 0.0005).astype(np.int8)

df = pd.DataFrame({
    "ip": np.r_[ip_n, ip_b], "app": np.r_[app_n, app_b], "device": np.r_[dev_n, dev_b],
    "os": np.r_[os_n, os_b], "channel": np.r_[ch_n, ch_b],
    "click_time": pd.to_datetime(np.r_[t_n, t_b], unit="s").strftime("%Y-%m-%d %H:%M:%S"),
    "attributed_time": "", "is_attributed": np.r_[y_n, y_b],
}).sample(frac=1, random_state=1).reset_index(drop=True)
import os; os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
df.to_csv(a.out, index=False)
print(f"wrote {len(df)} rows -> {a.out}  (genuine rate {df.is_attributed.mean():.3%})  SYNTHETIC")
