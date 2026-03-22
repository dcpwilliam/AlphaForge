from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


SignalFunc = Callable[[pd.DataFrame, dict], pd.Series]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    display_name: str
    description: str
    default_params: dict
    signal_func: SignalFunc


def _rolling_slope(arr: np.ndarray) -> float:
    x = np.arange(len(arr))
    k, _ = np.polyfit(x, arr, 1)
    return float(k)


def _dual_ma_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    short_win = int(params.get("short_win", 10))
    long_win = int(params.get("long_win", 30))
    sma_short = df["close"].rolling(short_win).mean()
    sma_long = df["close"].rolling(long_win).mean()
    return (sma_short > sma_long).astype(int)


def _sma_regression_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    short_win = int(params.get("short_win", 10))
    long_win = int(params.get("long_win", 30))
    reg_win = int(params.get("reg_win", 20))

    sma_short = df["close"].rolling(short_win).mean()
    sma_long = df["close"].rolling(long_win).mean()
    reg_slope = df["close"].rolling(reg_win).apply(lambda s: _rolling_slope(s.values), raw=False)
    return ((sma_short > sma_long) & (reg_slope > 0)).astype(int)


def _rsi_mean_reversion_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    rsi_period = int(params.get("rsi_period", 14))
    rsi_low = float(params.get("rsi_low", 30.0))
    rsi_high = float(params.get("rsi_high", 70.0))

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))

    long_signal = (rsi < rsi_low).astype(int)
    flat_signal = (rsi > rsi_high).astype(int)

    position = pd.Series(0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        prev = position.iat[i - 1]
        if long_signal.iat[i] == 1:
            position.iat[i] = 1
        elif flat_signal.iat[i] == 1:
            position.iat[i] = 0
        else:
            position.iat[i] = prev

    return position.astype(int)


def _bollinger_breakout_signal(df: pd.DataFrame, params: dict) -> pd.Series:
    bb_window = int(params.get("bb_window", 20))
    bb_k = float(params.get("bb_k", 2.0))

    ma = df["close"].rolling(bb_window).mean()
    std = df["close"].rolling(bb_window).std()
    upper = ma + bb_k * std
    lower = ma - bb_k * std

    # 趋势突破：收盘价突破上轨做多，跌破中轨平仓
    enter = (df["close"] > upper).astype(int)
    exit_ = (df["close"] < ma).astype(int)

    position = pd.Series(0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        prev = position.iat[i - 1]
        if enter.iat[i] == 1:
            position.iat[i] = 1
        elif exit_.iat[i] == 1:
            position.iat[i] = 0
        else:
            position.iat[i] = prev

    return position.astype(int)


STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    "dual_ma": StrategySpec(
        name="dual_ma",
        display_name="Dual MA Crossover",
        description="短均线向上穿越长均线做多，反之空仓。",
        default_params={"short_win": 10, "long_win": 30, "total_capital": 1_000_000.0},
        signal_func=_dual_ma_signal,
    ),
    "sma_regression_filter": StrategySpec(
        name="sma_regression_filter",
        display_name="SMA + Regression Filter",
        description="短均线在长均线上方且回归斜率为正时做多。",
        default_params={"short_win": 10, "long_win": 30, "reg_win": 20, "total_capital": 1_000_000.0},
        signal_func=_sma_regression_signal,
    ),
    "rsi_mean_reversion": StrategySpec(
        name="rsi_mean_reversion",
        display_name="RSI Mean Reversion",
        description="RSI低于阈值买入，RSI高于阈值平仓。",
        default_params={"rsi_period": 14, "rsi_low": 30, "rsi_high": 70, "total_capital": 1_000_000.0},
        signal_func=_rsi_mean_reversion_signal,
    ),
    "bollinger_breakout": StrategySpec(
        name="bollinger_breakout",
        display_name="Bollinger Breakout",
        description="价格突破布林上轨入场，跌破中轨离场。",
        default_params={"bb_window": 20, "bb_k": 2.0, "total_capital": 1_000_000.0},
        signal_func=_bollinger_breakout_signal,
    ),
}


def list_strategies() -> pd.DataFrame:
    rows = []
    for spec in STRATEGY_REGISTRY.values():
        rows.append(
            {
                "strategy_name": spec.name,
                "display_name": spec.display_name,
                "description": spec.description,
                "default_params": str(spec.default_params),
            }
        )
    return pd.DataFrame(rows)


def run_strategy_backtest(df: pd.DataFrame, strategy_name: str, params: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}, choose from {sorted(STRATEGY_REGISTRY)}")

    spec = STRATEGY_REGISTRY[strategy_name]
    merged_params = dict(spec.default_params)
    if params:
        merged_params.update(params)

    total_capital = float(merged_params.get("total_capital", 1_000_000.0))

    out = df.sort_values("trade_date").copy()
    out["ret"] = out["close"].pct_change().fillna(0.0)

    raw_position = spec.signal_func(out, merged_params)
    out["position"] = raw_position.shift(1).fillna(0).astype(int)
    out["strategy_ret"] = out["position"] * out["ret"]

    out["benchmark_curve"] = total_capital * (1 + out["ret"]).cumprod()
    out["strategy_curve"] = total_capital * (1 + out["strategy_ret"]).cumprod()

    out["benchmark_cum_return"] = out["benchmark_curve"] / total_capital - 1
    out["strategy_cum_return"] = out["strategy_curve"] / total_capital - 1

    out["position_change"] = out["position"].diff().fillna(0)
    out["signal_type"] = np.where(out["position_change"] > 0, "BUY", np.where(out["position_change"] < 0, "SELL", "HOLD"))

    days = max(len(out), 1)
    ann_factor = 252 / days
    total_return = float(out["strategy_cum_return"].iloc[-1])
    cagr = (1 + total_return) ** ann_factor - 1
    max_drawdown = float((out["strategy_curve"] / out["strategy_curve"].cummax() - 1).min())
    sharpe = float(np.sqrt(252) * out["strategy_ret"].mean() / (out["strategy_ret"].std() + 1e-12))

    metrics = {
        "strategy_name": strategy_name,
        "total_capital": total_capital,
        "final_amount": float(out["strategy_curve"].iloc[-1]),
        "total_return": total_return,
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }

    return out, metrics


def _calc_curve_metrics(curve: pd.Series) -> dict:
    curve = curve.dropna()
    if curve.empty:
        return {"total_return": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    ret = curve.pct_change().fillna(0.0)
    days = max(len(curve), 1)
    ann_factor = 252 / days
    total_return = float(curve.iloc[-1] / curve.iloc[0] - 1)
    cagr = (1 + total_return) ** ann_factor - 1
    max_drawdown = float((curve / curve.cummax() - 1).min())
    sharpe = float(np.sqrt(252) * ret.mean() / (ret.std() + 1e-12))
    return {
        "total_return": total_return,
        "cagr": float(cagr),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def run_enhanced_breakout_portfolio_backtest(
    bars: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    组合级增强突破策略（非实盘）:
    - 调仓频率控制（默认20日）
    - 个股冷却期（默认10日）
    - 最小交易金额比例
    - 突破信号 +1% 过滤假突破
    - 统一滑点 + 手续费
    - 分级日志（返回到 logs DataFrame）
    """
    cfg = {
        "initial_capital": 1_000_000.0,
        "top_n": 10,
        "rebalance_days": 20,
        "cooldown_days": 10,
        "min_trade_ratio": 0.02,
        "breakout_filter": 1.01,
        "slippage": 0.001,
        "fee_rate": 0.0003,
        "lot_size": 100,
        "atr_window": 14,
        "atr_rank_window": 20,
        "warmup_days": 62,
        "log_level": "INFO",
    }
    if params:
        cfg.update(params)

    required_cols = {"trade_date", "symbol", "open", "high", "close"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    data = bars.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = data.sort_values(["trade_date", "symbol"]).drop_duplicates(["trade_date", "symbol"])

    px_open = data.pivot(index="trade_date", columns="symbol", values="open").sort_index()
    px_high = data.pivot(index="trade_date", columns="symbol", values="high").sort_index()
    px_close = data.pivot(index="trade_date", columns="symbol", values="close").sort_index()
    px_close_ffill = px_close.ffill()

    # 简化 ATR: 用 |close - prev_close| 的 rolling mean 近似，兼容缺少 low 字段时的场景
    tr = (px_close - px_close.shift(1)).abs()
    atr = tr.rolling(int(cfg["atr_window"])).mean()
    atr_rank_signal = (atr / (px_close + 1e-12)).rolling(int(cfg["atr_rank_window"])).mean()

    cash = float(cfg["initial_capital"])
    holdings_shares: dict[str, int] = {s: 0 for s in px_close.columns}
    last_trade_day: dict[str, int] = {s: -999_999 for s in px_close.columns}
    last_rebalance = -999_999

    level_rank = {"DEBUG": 10, "INFO": 20, "TRADE": 25, "PNL": 30}
    chosen_rank = level_rank.get(str(cfg["log_level"]).upper(), 20)
    logs: list[dict] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []

    def add_log(level: str, day_idx: int, msg: str):
        if level_rank.get(level, 99) >= chosen_rank:
            logs.append({"trade_date": px_close.index[day_idx], "level": level, "message": msg})

    min_trade_value = float(cfg["initial_capital"]) * float(cfg["min_trade_ratio"])
    breakout_mul = float(cfg["breakout_filter"])
    rebalance_days = int(cfg["rebalance_days"])
    cooldown_days = int(cfg["cooldown_days"])
    slippage = float(cfg["slippage"])
    fee_rate = float(cfg["fee_rate"])
    lot_size = int(cfg["lot_size"])
    top_n = int(cfg["top_n"])
    warmup = int(cfg["warmup_days"])

    for i in range(len(px_close.index)):
        dt = px_close.index[i]

        # 每日权益统计
        close_row = px_close_ffill.iloc[i]
        pos_value = 0.0
        for s, shares in holdings_shares.items():
            if shares <= 0:
                continue
            c = close_row.get(s)
            if pd.notna(c):
                pos_value += float(c) * shares
        equity_rows.append({"trade_date": dt, "cash": cash, "position_value": pos_value, "equity": cash + pos_value})

        if i < warmup:
            continue
        if i - last_rebalance < rebalance_days:
            continue
        last_rebalance = i
        add_log("INFO", i, "开始调仓")

        buy_candidates: list[str] = []
        sell_candidates: list[str] = []

        # 生成买卖信号
        for s in px_close.columns:
            hs = px_high[s].iloc[:i]  # 到 i-1
            cs = px_close[s].iloc[:i]  # 到 i-1
            if hs.notna().sum() < 22 or cs.notna().sum() < 62:
                continue

            hs = hs.dropna()
            cs = cs.dropna()
            if len(hs) < 22 or len(cs) < 62:
                continue

            recent_high = hs.iloc[-22:]
            recent_close = cs.iloc[-62:]
            prev_high = recent_high.iloc[-1]
            breakout_ref = recent_high.iloc[:-1].max()
            ma_ref = recent_close.iloc[:-1].mean()

            if prev_high > breakout_ref * breakout_mul:
                buy_candidates.append(s)
            elif prev_high < ma_ref:
                sell_candidates.append(s)

        # ATR排序，取前 top_n
        rank_map: dict[str, float] = {}
        for s in buy_candidates:
            v = atr_rank_signal[s].iloc[i - 1] if i - 1 >= 0 else np.nan
            if pd.notna(v):
                rank_map[s] = float(v)
        selected = [k for k, _ in sorted(rank_map.items(), key=lambda kv: kv[1])[:top_n]]
        add_log("INFO", i, f"候选池={len(buy_candidates)}, 入选={len(selected)}")

        # 卖出
        for s in sell_candidates:
            if holdings_shares.get(s, 0) <= 0:
                continue
            if i - last_trade_day[s] < cooldown_days:
                continue

            o = px_open[s].iloc[i]
            if pd.isna(o) or o <= 0:
                continue
            sell_price = float(o) * (1 - slippage)
            shares = holdings_shares[s]
            fee = sell_price * shares * fee_rate
            cash += sell_price * shares - fee
            holdings_shares[s] = 0
            last_trade_day[s] = i
            trades.append(
                {
                    "trade_date": dt,
                    "symbol": s,
                    "side": "SELL",
                    "shares": shares,
                    "price": sell_price,
                    "fee": fee,
                }
            )
            add_log("TRADE", i, f"SELL {s} shares={shares} price={sell_price:.4f}")

        # 买入资金分配（等权）
        if selected:
            alloc = cash / len(selected)
        else:
            alloc = 0.0

        for s in selected:
            if holdings_shares.get(s, 0) > 0:
                continue
            if i - last_trade_day[s] < cooldown_days:
                continue
            if alloc < min_trade_value:
                continue

            o = px_open[s].iloc[i]
            if pd.isna(o) or o <= 0:
                continue
            buy_price = float(o) * (1 + slippage)
            raw_shares = int(alloc / buy_price)
            shares = (raw_shares // lot_size) * lot_size
            if shares <= 0:
                continue

            amount = buy_price * shares
            fee = amount * fee_rate
            total_cost = amount + fee
            if total_cost > cash:
                continue

            cash -= total_cost
            holdings_shares[s] = shares
            last_trade_day[s] = i
            trades.append(
                {
                    "trade_date": dt,
                    "symbol": s,
                    "side": "BUY",
                    "shares": shares,
                    "price": buy_price,
                    "fee": fee,
                }
            )
            add_log("TRADE", i, f"BUY  {s} shares={shares} price={buy_price:.4f}")

        add_log("PNL", i, f"cash={cash:.2f}")

    equity = pd.DataFrame(equity_rows).sort_values("trade_date")
    equity["strategy_return"] = equity["equity"].pct_change().fillna(0.0)
    equity["strategy_curve"] = equity["equity"] / float(cfg["initial_capital"])
    equity["strategy_cum_return"] = equity["strategy_curve"] - 1

    # 基准：全市场等权收盘收益
    benchmark_ret = px_close_ffill.pct_change(fill_method=None).mean(axis=1).reindex(equity["trade_date"]).fillna(0.0).values
    equity["benchmark_return"] = benchmark_ret
    equity["benchmark_curve"] = (1 + equity["benchmark_return"]).cumprod()
    equity["benchmark_cum_return"] = equity["benchmark_curve"] - 1

    metrics = _calc_curve_metrics(equity["equity"])
    metrics.update(
        {
            "strategy_name": "enhanced_breakout_portfolio",
            "initial_capital": float(cfg["initial_capital"]),
            "final_amount": float(equity["equity"].iloc[-1]) if not equity.empty else float(cfg["initial_capital"]),
            "num_trades": int(len(trades)),
        }
    )

    trade_df = pd.DataFrame(trades)
    log_df = pd.DataFrame(logs)
    # 日志表不作为主返回，挂到 attrs 便于 notebook 取用
    equity.attrs["logs"] = log_df
    return equity, trade_df, metrics
