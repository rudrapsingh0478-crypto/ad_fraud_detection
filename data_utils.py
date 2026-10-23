import zipfile
import numpy as np
import pandas as pd


def _open(path):
    """For .zip files, stream the CSV inside instead of loading it all into RAM."""
    if str(path).lower().endswith(".zip"):
        z = zipfile.ZipFile(path)
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        name = next((n for n in names if n.lower().endswith("train.csv")), names[0])
        return z.open(name)
    return path


def load_clicks(path, nrows=None):
    """Load TalkingData train.csv (or .zip) and return time-sorted numpy arrays."""
    df = pd.read_csv(
        _open(path), nrows=nrows,
        usecols=["ip", "app", "device", "os", "channel", "click_time", "is_attributed"],
        dtype={"ip": "int64", "app": "int32", "device": "int32", "os": "int32",
               "channel": "int32", "is_attributed": "int8"},
    )
    ts = (pd.to_datetime(df["click_time"]) - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    df["ts"] = ts.astype("int64")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    return {
        "ts": df["ts"].to_numpy(), "ip": df["ip"].to_numpy(), "app": df["app"].to_numpy(),
        "device": df["device"].to_numpy(), "os": df["os"].to_numpy(),
        "channel": df["channel"].to_numpy(), "y": df["is_attributed"].to_numpy(),
    }