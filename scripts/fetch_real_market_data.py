from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from quant_research.market_data import fetch_one, resolve_universe


def main():
    parser = argparse.ArgumentParser(description="Fetch real market data and save parquet")
    parser.add_argument("--symbols", default=None, help="MARKET:CODE list, comma separated")
    parser.add_argument("--pool-size", type=int, default=50, choices=[50, 100])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output", default="data/raw/daily_bars_real.parquet")
    args = parser.parse_args()

    pairs = resolve_universe(args.symbols, args.pool_size)
    frames: list[pd.DataFrame] = []

    for market, code in pairs:
        part = fetch_one(market, code, args.start, args.end)
        frames.append(part)
        print(f"fetched {market}:{code}, rows={len(part)}")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved: {out}, rows={len(df)}, symbols={df['symbol'].nunique()}")


if __name__ == "__main__":
    main()
