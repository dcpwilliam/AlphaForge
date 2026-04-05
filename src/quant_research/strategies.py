from __future__ import annotations

import ast
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


_DISALLOWED_CALLS = {
    "eval",
    "exec",
    "open",
    "__import__",
    "compile",
    "input",
}

_DISALLOWED_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("subprocess", "run"),
    ("subprocess", "Popen"),
}

_ALLOWED_NAME_CALLS = {
    "init",
    "handlebar",
    "sell_open",
    "buy_open",
    "buy_close_tdayfirst",
    "sell_close_tdayfirst",
    "timetag_to_datetime",
    "print",
    "int",
    "float",
    "len",
    "range",
    "abs",
}

_ALLOWED_IMPORTS = {"numpy"}


def validate_context_python_strategy(code: str) -> dict:
    report = {
        "syntax_ok": False,
        "has_init": False,
        "has_handlebar": False,
        "illegal_calls": [],
        "illegal_imports": [],
        "errors": [],
    }
    try:
        tree = ast.parse(code)
        report["syntax_ok"] = True
    except SyntaxError as exc:
        report["errors"].append(f"SyntaxError: line {exc.lineno}, col {exc.offset}, {exc.msg}")
        return report

    fn_defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    report["has_init"] = "init" in fn_defs
    report["has_handlebar"] = "handlebar" in fn_defs
    if not report["has_init"]:
        report["errors"].append("缺少函数: init(ContextInfo)")
    if not report["has_handlebar"]:
        report["errors"].append("缺少函数: handlebar(ContextInfo)")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_IMPORTS:
                    report["illegal_imports"].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in _ALLOWED_IMPORTS:
                report["illegal_imports"].append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                fn = node.func.id
                if fn in _DISALLOWED_CALLS:
                    report["illegal_calls"].append(fn)
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                owner = node.func.value.id
                attr = node.func.attr
                if (owner, attr) in _DISALLOWED_ATTR_CALLS:
                    report["illegal_calls"].append(f"{owner}.{attr}")

    # 仅允许 ContextInfo 的常见方法调用，其他对象方法默认放行（例如 numpy/pandas）
    allowed_ctx_methods = {
        "set_universe",
        "get_bar_timetag",
        "get_market_data",
        "paint",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "ContextInfo":
                if node.func.attr not in allowed_ctx_methods:
                    report["illegal_calls"].append(f"ContextInfo.{node.func.attr}")

    # 附加 name call 合法性检查（不过于苛刻，未识别的 Name 调用给提示）
    unknown_name_calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn not in _ALLOWED_NAME_CALLS and fn not in fn_defs and fn not in {"np"}:
                # 对 numpy 别名函数调用通过 Attribute 识别，不在这里处理
                if fn not in _DISALLOWED_CALLS:
                    unknown_name_calls.add(fn)
    if unknown_name_calls:
        report["errors"].append(f"发现未识别函数调用: {sorted(unknown_name_calls)}")

    if report["illegal_imports"]:
        report["errors"].append(f"非法 import: {sorted(set(report['illegal_imports']))}")
    if report["illegal_calls"]:
        report["errors"].append(f"非法调用: {sorted(set(report['illegal_calls']))}")

    report["valid"] = report["syntax_ok"] and report["has_init"] and report["has_handlebar"] and not report["illegal_calls"] and not report["illegal_imports"]
    return report


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


def run_pair_spread_bollinger_backtest(
    bars: pd.DataFrame,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    配对价差布林回测（非实盘）:
    - spread = price_a - price_b
    - 过去N窗口均值 ± k*std 为上下轨
    - spread > up: 做空价差（A空/B多）
    - spread < down: 做多价差（A多/B空）
    - spread 回归轨道内平仓
    """
    cfg = {
        "symbol_a": "rb00.SF",
        "symbol_b": "hc00.SF",
        "window": 30,
        "band_k": 0.5,
        "initial_capital": 1_000_000.0,
        "trade_notional": 200_000.0,
        "slippage": 0.0005,
        "fee_rate": 0.0003,
    }
    if params:
        cfg.update(params)

    required_cols = {"trade_date", "symbol", "close"}
    missing = required_cols - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")

    symbol_a = str(cfg["symbol_a"])
    symbol_b = str(cfg["symbol_b"])
    df = bars.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["symbol"].isin([symbol_a, symbol_b])]
    px = df.pivot(index="trade_date", columns="symbol", values="close").sort_index().ffill()
    if symbol_a not in px.columns or symbol_b not in px.columns:
        raise ValueError(f"bars 中缺少标的: {symbol_a} 或 {symbol_b}")

    spread = px[symbol_a] - px[symbol_b]
    window = int(cfg["window"])
    k = float(cfg["band_k"])
    ma = spread.rolling(window).mean()
    sd = spread.rolling(window).std()
    up = ma + k * sd
    down = ma - k * sd

    cash = float(cfg["initial_capital"])
    pos_a = 0  # >0 多A, <0 空A
    pos_b = 0  # >0 多B, <0 空B
    state = "flat"  # flat/long_spread/short_spread
    slippage = float(cfg["slippage"])
    fee_rate = float(cfg["fee_rate"])
    target_notional = float(cfg["trade_notional"])

    trades: list[dict] = []
    rows: list[dict] = []

    for i, dt in enumerate(px.index):
        pa = float(px.at[dt, symbol_a])
        pb = float(px.at[dt, symbol_b])
        s = float(spread.iat[i]) if pd.notna(spread.iat[i]) else np.nan
        u = float(up.iat[i]) if pd.notna(up.iat[i]) else np.nan
        d = float(down.iat[i]) if pd.notna(down.iat[i]) else np.nan

        # 每日盯市
        position_value = pos_a * pa + pos_b * pb
        equity = cash + position_value

        rows.append(
            {
                "trade_date": dt,
                "price_a": pa,
                "price_b": pb,
                "spread": s,
                "up": u,
                "down": d,
                "state": state,
                "pos_a": pos_a,
                "pos_b": pos_b,
                "cash": cash,
                "equity": equity,
            }
        )

        if i < window or np.isnan(s) or np.isnan(u) or np.isnan(d):
            continue

        qty_a = max(int(target_notional / max(pa, 1e-12)), 1)
        qty_b = max(int(target_notional / max(pb, 1e-12)), 1)

        def _trade(symbol: str, side: str, qty: int, price: float):
            nonlocal cash
            px_exec = price * (1 + slippage if side == "BUY" else 1 - slippage)
            amt = px_exec * qty
            fee = amt * fee_rate
            if side == "BUY":
                cash -= amt + fee
            else:
                cash += amt - fee
            trades.append(
                {
                    "trade_date": dt,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": px_exec,
                    "fee": fee,
                    "state": state,
                }
            )

        # 先平后开，保持与示例逻辑一致
        if state == "short_spread":
            if s <= u:
                # 平空价差: A买平 / B卖平
                _trade(symbol_a, "BUY", abs(pos_a), pa)
                _trade(symbol_b, "SELL", abs(pos_b), pb)
                pos_a, pos_b = 0, 0
                state = "flat"
            if s < d and state == "flat":
                # 反手做多价差
                _trade(symbol_a, "BUY", qty_a, pa)
                _trade(symbol_b, "SELL", qty_b, pb)
                pos_a, pos_b = qty_a, -qty_b
                state = "long_spread"

        elif state == "long_spread":
            if s >= d:
                # 平多价差: A卖平 / B买平
                _trade(symbol_a, "SELL", abs(pos_a), pa)
                _trade(symbol_b, "BUY", abs(pos_b), pb)
                pos_a, pos_b = 0, 0
                state = "flat"
            if s > u and state == "flat":
                # 反手做空价差
                _trade(symbol_a, "SELL", qty_a, pa)
                _trade(symbol_b, "BUY", qty_b, pb)
                pos_a, pos_b = -qty_a, qty_b
                state = "short_spread"

        else:  # flat
            if s > u:
                _trade(symbol_a, "SELL", qty_a, pa)
                _trade(symbol_b, "BUY", qty_b, pb)
                pos_a, pos_b = -qty_a, qty_b
                state = "short_spread"
            elif s < d:
                _trade(symbol_a, "BUY", qty_a, pa)
                _trade(symbol_b, "SELL", qty_b, pb)
                pos_a, pos_b = qty_a, -qty_b
                state = "long_spread"

    curve = pd.DataFrame(rows).sort_values("trade_date")
    curve["strategy_return"] = curve["equity"].pct_change().fillna(0.0)
    curve["strategy_curve"] = curve["equity"] / float(cfg["initial_capital"])
    curve["strategy_cum_return"] = curve["strategy_curve"] - 1

    bench = (px[symbol_a].pct_change(fill_method=None).fillna(0.0) + px[symbol_b].pct_change(fill_method=None).fillna(0.0)) / 2
    curve["benchmark_return"] = bench.reindex(curve["trade_date"]).fillna(0.0).values
    curve["benchmark_curve"] = (1 + curve["benchmark_return"]).cumprod()
    curve["benchmark_cum_return"] = curve["benchmark_curve"] - 1

    trade_df = pd.DataFrame(trades)
    metrics = _calc_curve_metrics(curve["equity"])
    metrics.update(
        {
            "strategy_name": "pair_spread_bollinger",
            "initial_capital": float(cfg["initial_capital"]),
            "final_amount": float(curve["equity"].iloc[-1]) if not curve.empty else float(cfg["initial_capital"]),
            "num_trades": int(len(trades)),
            "symbol_a": symbol_a,
            "symbol_b": symbol_b,
            "window": window,
            "band_k": k,
        }
    )
    return curve, trade_df, metrics
