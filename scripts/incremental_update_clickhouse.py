from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import pandas as pd

from quant_research.db import get_client, insert_df, query_df
from quant_research.factors import calc_technical_factors
from quant_research.market_data import fetch_one, resolve_universe


def _to_ymd(value: datetime) -> str:
    return value.strftime("%Y-%m-%d")


def _symbol_key(market: str, code: str) -> str:
    return f"{market}:{code}"


def _quote_symbol(sym: str) -> str:
    return "'" + sym.replace("'", "''") + "'"


def purge_existing(symbols: list[str], min_date: str, max_date: str):
    client = get_client()
    try:
        symbols_csv = ", ".join(_quote_symbol(s) for s in symbols)
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


def load_last_dates(symbols: list[str]) -> dict[str, str]:
    if not symbols:
        return {}
    symbols_csv = ", ".join(_quote_symbol(s) for s in symbols)
    df = query_df(
        f"""
        select symbol, max(trade_date) as last_date
        from daily_bars
        where symbol in ({symbols_csv})
        group by symbol
        """
    )
    if df.empty:
        return {}
    return {row["symbol"]: str(row["last_date"]) for _, row in df.iterrows()}


def main():
    parser = argparse.ArgumentParser(description="Incremental update for multi-market symbols")
    parser.add_argument("--symbols", default=None, help="MARKET:CODE list, comma separated")
    parser.add_argument("--pool-size", type=int, default=50, choices=[50, 100])
    parser.add_argument("--history-start", default="2020-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--lookback-days", type=int, default=30)
    args = parser.parse_args()

    pairs = resolve_universe(args.symbols, args.pool_size)
    symbol_names = [_symbol_key(market, code) for market, code in pairs]
    last_dates = load_last_dates(symbol_names)

    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    history_start_dt = datetime.strptime(args.history_start, "%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    for market, code in pairs:
        symbol = _symbol_key(market, code)
        if symbol in last_dates:
            start_dt = pd.to_datetime(last_dates[symbol]).to_pydatetime() - timedelta(days=args.lookback_days)
            if start_dt < history_start_dt:
                start_dt = history_start_dt
        else:
            start_dt = history_start_dt

        if start_dt >= end_dt:
            print(f"skip {symbol}: start {start_dt.date()} >= end {end_dt.date()}")
            continue

        start = _to_ymd(start_dt)
        end = _to_ymd(end_dt + timedelta(days=1))

        try:
            part = fetch_one(market, code, start=start, end=end)
            frames.append(part)
            print(f"fetched {symbol}, range={start}~{args.end}, rows={len(part)}")
        except Exception as exc:  # noqa: BLE001
            print(f"failed {symbol}: {exc}")

    if not frames:
        print("no data fetched, skip update")
        return

    bars = pd.concat(frames, ignore_index=True)
    bars = bars.sort_values(["symbol", "trade_date"]).drop_duplicates(["symbol", "trade_date"])
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date

    symbols = sorted(bars["symbol"].unique().tolist())
    min_date = str(bars["trade_date"].min())
    max_date = str(bars["trade_date"].max())

    purge_existing(symbols, min_date, max_date)
    insert_df("daily_bars", bars)

    factors = calc_technical_factors(bars)
    factors["trade_date"] = pd.to_datetime(factors["trade_date"]).dt.date
    insert_df("factor_values", factors)

    print(f"updated daily_bars: {len(bars)}, symbols={len(symbols)}, range={min_date}~{max_date}")
    print(f"updated factor_values: {len(factors)}")


if __name__ == "__main__":
    main()
