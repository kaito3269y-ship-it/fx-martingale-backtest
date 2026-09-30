import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

PAIRS = ["eurjpy", "usdjpy", "gbpusd", "audusd", "eurgbp"]

TOTAL_CAPITAL = 2_000_000
DEPLOYED_CAPITAL = 1_500_000
INITIAL_RESERVE = 500_000

# Initial account allocations = 1.5M total
ALLOCATIONS = {
    "eurjpy": 200_000,
    "usdjpy": 225_000,
    "gbpusd": 275_000,
    "audusd": 350_000,
    "eurgbp": 450_000,
}

MULTIPLIERS = {
    "eurjpy": 1.20,
    "usdjpy": 1.28,
    "gbpusd": 1.35,
    "audusd": 1.42,
    "eurgbp": 1.50,
}

LEVELS = {
    "eurjpy": 5,
    "usdjpy": 6,
    "gbpusd": 7,
    "audusd": 8,
    "eurgbp": 10,
}

# Conservative approximate trading costs in pips
SPREAD_PIPS = {
    "eurjpy": 1.0,
    "usdjpy": 0.8,
    "gbpusd": 1.0,
    "audusd": 1.0,
    "eurgbp": 1.2,
}

SLIPPAGE_PIPS = 0.2

LEVERAGE = 1000
STOP_OUT_LEVEL = 0.20

LOTS = [0.01, 0.015, 0.02, 0.025, 0.03]
GRIDS = [30, 40, 60, 80, 100, 130]
TPS = [10, 15, 20, 25, 30]


# ============================================================
# DATA
# ============================================================

def pip_size(pair):
    return 0.01 if pair.endswith("jpy") else 0.0001


def download(pair, timeframe, start, end):
    directory = Path("data") / pair
    directory.mkdir(parents=True, exist_ok=True)

    cmd = [
        "npx",
        "dukascopy-node",
        "-i", pair,
        "-from", start,
        "-to", end,
        "-t", timeframe,
        "-f", "csv",
        "-v", "true",
        "-dir", str(directory),
    ]

    subprocess.run(cmd, check=True)

    files = sorted(
        directory.rglob("*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise RuntimeError(f"No CSV downloaded for {pair}")

    return files[0]


def load_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    time_col = next(
        (
            c for c in df.columns
            if c in ("timestamp", "time", "date", "datetime")
        ),
        df.columns[0],
    )

    # Dukascopy timestamps are normally milliseconds since Unix epoch.
    raw_ts = pd.to_numeric(df[time_col], errors="coerce")

    if raw_ts.notna().mean() > 0.90:
        median_ts = raw_ts.dropna().median()

        if median_ts > 10**11:
            df["time"] = pd.to_datetime(
                raw_ts,
                unit="ms",
                utc=True,
                errors="coerce",
            )
        elif median_ts > 10**9:
            df["time"] = pd.to_datetime(
                raw_ts,
                unit="s",
                utc=True,
                errors="coerce",
            )
        else:
            df["time"] = pd.to_datetime(
                df[time_col],
                utc=True,
                errors="coerce",
            )
    else:
        df["time"] = pd.to_datetime(
            df[time_col],
            utc=True,
            errors="coerce",
        )

    for field in ["open", "high", "low", "close"]:
        if field not in df.columns:
            candidate = next(
                (c for c in df.columns if field in c),
                None,
            )

            if candidate is None:
                raise RuntimeError(
                    f"Could not find {field} column in {file_path}"
                )

            df[field] = df[candidate]

        df[field] = pd.to_numeric(df[field], errors="coerce")

    df = (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )

    if len(df) == 0:
        raise RuntimeError(f"No usable rows in {file_path}")

    return df


# ============================================================
# SIGNALS
# ============================================================

def signals(df, strategy_id):
    close = df["close"]

    # EURJPY: Z-score mean reversion
    if strategy_id == 0:
        mean = close.rolling(96).mean()
        std = close.rolling(96).std()
        z = (close - mean) / std

        return np.where(
            z < -2.0,
            1,
            np.where(z > 2.0, -1, 0),
        )

    # USDJPY: Bollinger mean reversion