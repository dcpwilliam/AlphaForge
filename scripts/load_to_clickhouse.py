from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_research.db import get_client, insert_df
from quant_research.factors import calc_technical_factors


def purge_existing(symbols: list[str], min_date: str, max_date: str):
    client = get_client()
    try:
        symbols_csv = ", ".join(f"'{s}'" for s in symbols)
        client.command(
            f"""
            ALTER TABLE daily_bars
            DELETE WHERE symbol IN ({symbols_csv})
              AND trade_date BETWEEN toDate('{min_date}') AND toDate('{max_date}')
            """
        )
        client.command(
            f"""
            ALTER TABLE factor_values
            DELETE WHERE symbol IN ({symbols_csv})
              AND trade_date BETWEEN toDate('{min_date}') AND toDate('{max_date}')
            """
        )
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load parquet bars into ClickHouse")
    parser.add_argument("--input", default="data/raw/daily_bars_real.parquet")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件: {path}，请先运行 scripts/fetch_real_market_data.py")

    bars = pd.read_parquet(path)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date

    symbols = sorted(bars["symbol"].unique().tolist())
    min_date = str(bars["trade_date"].min())
    max_date = str(bars["trade_date"].max())

    purge_existing(symbols, min_date, max_date)

    insert_df("daily_bars", bars)
    print(f"inserted daily_bars: {len(bars)}, symbols={len(symbols)}, range={min_date}~{max_date}")

    factors = calc_technical_factors(bars)
    factors["trade_date"] = pd.to_datetime(factors["trade_date"]).dt.date
    insert_df("factor_values", factors)
    print(f"inserted factor_values: {len(factors)}")
