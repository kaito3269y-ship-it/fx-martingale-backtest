import argparse, json, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ["eurjpy", "usdjpy", "gbpusd", "audusd", "eurgbp"]

ALLOC = {
    "eurjpy": 200000,
    "usdjpy": 225000,
    "gbpusd": 275000,
    "audusd": 350000,
    "eurgbp": 450000,
}

MULT = {
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

SPREAD = {
    "eurjpy": 1.0,
    "usdjpy": 0.8,
    "gbpusd": 1.0,
    "audusd": 1.0,
    "eurgbp": 1.2,
}

TOTAL = 2000000
RESERVE = 500000
SLIP = 0.2


def pip(pair):
    return 0.01 if pair.endswith("jpy") else 0.0001


def download(pair, tf, start, end):
    d = Path("data") / pair
    d.mkdir(parents=True, exist_ok=True)

    cmd = [
        "npx", "dukascopy-node",
        "-i", pair,
        "-from", start,
        "-to", end,
        "-t", tf,
        "-f", "csv",
        "-v", "true",
        "-dir", str(d),
    ]

    subprocess.run(cmd, check=True)

    files = sorted(
        d.rglob("*.csv"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise RuntimeError(f"No data: {pair}")

    return files[0]


def load(path):
    df = pd.read_csv(path)
    df.columns = [str(c).lower().strip() for c in df.columns]

    tc = next(
        (c for c in df.columns if c in ("timestamp", "time", "date", "datetime")),
        df.columns[0],
    )

    raw = pd.to_numeric(df[tc], errors="coerce")

    if raw.notna().mean() > 0.9:
        med = raw.dropna().median()

        if med > 1e11:
            df["time"] = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce")
        elif med > 1e9:
            df["time"] = pd.to_datetime(raw, unit="s", utc=True, errors="coerce")
        else:
            df["time"] = pd.to_datetime(df[tc], utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(df[tc], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            alt = next((x for x in df.columns if c in x), None)
            if alt is None:
                raise RuntimeError(f"Missing {c}")
            df[c] = df[alt]

        df[c] = pd.to_numeric(df[c], errors="coerce")

    return (
        df.dropna(subset=["time", "open", "high", "low", "close"])
        .sort_values("time")
        .drop_duplicates("time")
        .reset_index(drop=True)
    )


def signal(df, k):
    c = df.close

    if k == 0:
        m = c.rolling(96).mean()
        s = c.rolling(96).std()
        z = (c - m) / s
        return np.where(z < -2, 1, np.where(z > 2, -1, 0))

    if k == 1:
        m = c.rolling(80).mean()
        s = c.rolling(80).std()
        return np.where(c < m - 2*s, 1, np.where(c > m + 2*s, -1, 0))

    if k == 2:
        d = c.diff()
        up = d.clip(lower=0).rolling(14).mean()
        dn = (-d.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        return np.where(rsi < 28, 1, np.where(rsi > 72, -1, 0))

    if k == 3:
        m = c.ewm(span=120, adjust=False).mean()
        dev = (c - m) / c
        return np.where(dev < -0.008, 1, np.where(dev > 0.008, -1, 0))

    hi = c.rolling(96).max().shift(1)
    lo = c.rolling(96).min().shift(1)

    return np.where(c < lo, 1, np.where(c > hi, -1, 0))


def pip_value(pair, lot):
    if pair.endswith("jpy"):
        return 1000 * lot

    if pair in ("gbpusd", "audusd"):
        return 1500 * lot

    if pair == "eurgbp":
        return 1900 * lot

    return 1500 * lot


def run(df, pair, k, capital, lot, grid, tp):
    sig = signal(df, k)

    cash = float(capital)
    withdrawn = 0.0
    pos = []
    side = 0

    baskets = 0
    fills = 0
    peak = float(capital)
    maxdd = 0.0
    month = None

    p = pip(pair)

    for i in range(len(df)):
        r = df.iloc[i]
        px = float(r.close)
        ym = (r.time.year, r.time.month)

        if month is not None and ym != month and not pos and cash > capital:
            withdrawn += cash - capital
            cash = float(capital)

        month = ym

        if not pos and sig[i] != 0:
            side = int(sig[i])
            cost = (SPREAD[pair] / 2 + SLIP) * p
            pos = [(px + side * cost, lot)]
            baskets += 1
            fills += 1

        elif pos:
            total = sum(q for _, q in pos)
            avg = sum(e*q for e, q in pos) / total

            target = avg + side * tp * p
            next_grid = pos[-1][0] - side * grid * p

            grid_hit = (
                (side == 1 and r.low <= next_grid)
                or
                (side == -1 and r.high >= next_grid)
            )

            tp_hit = (
                (side == 1 and r.high >= target)
                or
                (side == -1 and r.low <= target)
            )

            if grid_hit and len(pos) < LEVELS[pair]:
                q = lot * (MULT[pair] ** len(pos))
                cost = (SPREAD[pair] / 2 + SLIP) * p
                pos.append((next_grid + side * cost, q))
                fills += 1

                # Conservative: adverse move first
                tp_hit = False

            if tp_hit:
                cost = (SPREAD[pair] / 2 + SLIP) * p
                ex = target - side * cost

                pnl = sum(
                    side * (ex - e) / p * pip_value(pair, q)
                    for e, q in pos
                )

                cash += pnl
                pos = []
                side = 0

        unreal = sum(
            side * (px - e) / p * pip_value(pair, q)
            for e, q in pos
        )

        eq = max(0.0, cash + unreal)

        wealth = eq + withdrawn

        peak = max(peak, wealth)

        if peak > 0:
            maxdd = max(maxdd, (peak - wealth) / peak)

    final_eq = cash

    if pos:
        px = float(df.iloc[-1].close)

        final_eq = max(
            0.0,
            cash + sum(
                side * (px - e) / p * pip_value(pair, q)
                for e, q in pos
            ),
        )

    return {
        "final_equity": round(final_eq, 2),
        "withdrawals": round(withdrawn, 2),
        "wealth": round(final_eq + withdrawn, 2),
        "baskets": baskets,
        "fills": fills,
        "max_dd": round(maxdd, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="m15")
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default="2026-09-01")
    args = ap.parse_args()

    Path("results").mkdir(exist_ok=True)

    rows = []

    lots = [0.01, 0.02, 0.03]
    grids = [40, 80, 120]
    tps = [15, 25]

    for k, pair in enumerate(PAIRS):
        print(f"Downloading {pair}")

        df = load(download(pair, args.tf, args.start, args.end))

        print(
            pair,
            len(df),
            df.time.iloc[0],
            "->",
            df.time.iloc[-1],
        )

        cut = int(len(df) * 0.7)

        train = df.iloc[:cut].reset_index(drop=True)
        oos = df.iloc[cut:].reset_index(drop=True)

        best = None

        for lot in lots:
            for grid in grids:
                for tp in tps:
                    r = run(
                        train,
                        pair,
                        k,
                        ALLOC[pair],
                        lot,
                        grid,
                        tp,
                    )

                    score = (
                        r["wealth"]
                        - ALLOC[pair]
                        - ALLOC[pair] * 4 * r["max_dd"]
                    )

                    candidate = (score, lot, grid, tp, r)

                    if best is None or candidate[0] > best[0]:
                        best = candidate

        _, lot, grid, tp, train_result = best

        oos_result = run(
            oos,
            pair,
            k,
            ALLOC[pair],
            lot,
            grid,
            tp,
        )

        rows.append({
            "pair": pair,
            "allocation": ALLOC[pair],
            "lot": lot,
            "grid": grid,
            "tp": tp,
            "rows": len(df),
            "start": str(df.time.iloc[0]),
            "end": str(df.time.iloc[-1]),
            **{f"train_{x}": y for x, y in train_result.items()},
            **{f"oos_{x}": y for x, y in oos_result.items()},
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        "results/summary.csv",
        index=False,
    )

    account_wealth = sum(
        r["oos_wealth"]
        for r in rows
    )

    summary = {
        "starting_capital": TOTAL,
        "reserve": RESERVE,
        "timeframe": args.tf,
        "accounts_wealth": round(account_wealth, 2),
        "total_with_reserve": round(account_wealth + RESERVE, 2),
        "accounts": rows,
    }

    Path("results/summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(out.to_string(index=False))
    print()
    print("TOTAL INCLUDING RESERVE:", round(account_wealth + RESERVE))


if __name__ == "__main__":
    main()