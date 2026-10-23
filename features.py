"""
Causal (past-only) streaming features.

The SAME class is used for training, simulation, API and dashboard, so there is
no train/serve mismatch and no future information leaks into any feature:
features for a click are computed BEFORE that click is added to the state.

Windows are capped by `maxlen` events per key (default 200), so counts saturate
at 200 for very heavy keys. That is fine for a demo and keeps latency bounded.
"""
from collections import deque, defaultdict

FEATURE_NAMES = [
    "cnt_ip_60s", "cnt_ip_600s", "cnt_ip_3600s",
    "uniq_app_ip_600s", "uniq_chan_ip_600s",
    "cnt_ipapp_600s", "cnt_ipapp_3600s",
    "cnt_iposdev_600s", "cnt_ipchan_600s",
    "sec_since_last_ip", "sec_since_last_ipapp",
    "hour", "app", "device", "os", "channel",
]
NUMERIC_IDX = list(range(0, 12))
CATEGORICAL_IDX = [12, 13, 14, 15]
CAP_SECONDS = 3600.0  # "seconds since last click" is capped at 1 hour


def _count_window(dq, now, window):
    c = 0
    for t, _, _ in reversed(dq):
        if now - t > window:
            break
        c += 1
    return c


class StreamingFeatures:
    def __init__(self, maxlen=200, prune_every=100_000):
        self.maxlen = maxlen
        self.prune_every = prune_every
        mk = lambda: deque(maxlen=maxlen)
        self.ip = defaultdict(mk)       # ip -> (ts, app, channel)
        self.ipapp = defaultdict(mk)
        self.iposdev = defaultdict(mk)
        self.ipchan = defaultdict(mk)
        self.n = 0

    def step(self, ts, ip, app, device, os_, channel):
        """Return features for this click, then add the click to the state."""
        d_ip = self.ip[ip]
        d_ipapp = self.ipapp[(ip, app)]
        d_iposdev = self.iposdev[(ip, os_, device)]
        d_ipchan = self.ipchan[(ip, channel)]

        c60 = c600 = c3600 = 0
        apps, chans = set(), set()
        for t, a, c in reversed(d_ip):
            age = ts - t
            if age > 3600:
                break
            c3600 += 1
            if age <= 600:
                c600 += 1
                apps.add(a)
                chans.add(c)
                if age <= 60:
                    c60 += 1

        ipapp_600 = _count_window(d_ipapp, ts, 600)
        ipapp_3600 = _count_window(d_ipapp, ts, 3600)
        iposdev_600 = _count_window(d_iposdev, ts, 600)
        ipchan_600 = _count_window(d_ipchan, ts, 600)

        sec_ip = min(ts - d_ip[-1][0], CAP_SECONDS) if d_ip else CAP_SECONDS
        sec_ipapp = min(ts - d_ipapp[-1][0], CAP_SECONDS) if d_ipapp else CAP_SECONDS

        feats = [c60, c600, c3600, len(apps), len(chans),
                 ipapp_600, ipapp_3600, iposdev_600, ipchan_600,
                 sec_ip, sec_ipapp,
                 int(ts // 3600) % 24, app, device, os_, channel]

        ev = (ts, app, channel)
        d_ip.append(ev); d_ipapp.append(ev); d_iposdev.append(ev); d_ipchan.append(ev)
        self.n += 1
        if self.n % self.prune_every == 0:
            self._prune(ts)
        return feats

    def _prune(self, now):
        for d in (self.ip, self.ipapp, self.iposdev, self.ipchan):
            dead = [k for k, v in d.items() if v and now - v[-1][0] > 3600]
            for k in dead:
                del d[k]
