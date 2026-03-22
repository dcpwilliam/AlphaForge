from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .universe import get_symbol_pool


def normalize_to_yf_ticker(market: str, code: str) -> str:
    mkt = market.upper().strip()
    val = code.upper().strip()

    if mkt == "US":
        return val
    if mkt == "HK":
        return f"{val.zfill(4)}.HK"
    if mkt == "CN":
        if val.startswith("6"):
            return f"{val}.SS"
        return f"{val}.SZ"
    if mkt == "SG":
        return f"{val}.SI"

    raise ValueError(f"Unsupported market: {market}")


def parse_symbols(raw: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid symbol format: {item}, expected MARKET:CODE")
        market, code = item.split(":", 1)
        pairs.append((market.strip().upper(), code.strip().upper()))
    if not pairs:
        raise ValueError("No symbols provided")
    return pairs


def resolve_universe(symbols: str | None, pool_size: int) -> list[tuple[str, str]]:
    if symbols:
        return parse_symbols(symbols)
    return parse_symbols(",".join(get_symbol_pool(pool_size)))


def fetch_one(market: str, code: str, start: str, end: str) -> pd.DataFrame:
    ticker = normalize_to_yf_ticker(market, code)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise ValueError(f"No data fetched for {market}:{code} ({ticker})")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.reset_index().rename(
        columns={
            "Date": "trade_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
    raw["volume"] = raw["volume"].fillna(0.0).astype(float)
    raw["amount"] = raw["close"] * raw["volume"]
    raw["vwap"] = np.where(raw["volume"] > 0, raw["amount"] / raw["volume"], raw["close"])
    raw["ret_1d"] = raw["close"].pct_change().fillna(0.0)
    raw["ret_5d"] = raw["close"].pct_change(5).fillna(0.0)
    raw["symbol"] = f"{market}:{code}"

    cols = [
        "trade_date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "ret_1d",
        "ret_5d",
    ]
    return raw[cols].copy()
