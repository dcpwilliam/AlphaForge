from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_research.db import query_df
from quant_research.strategies import (
    list_strategies,
    run_enhanced_breakout_portfolio_backtest,
    run_strategy_backtest,
)


st.set_page_config(page_title="AlphaForge Strategy Web", layout="wide")
st.title("AlphaForge 策略研究看板")


STRATEGY_LOGIC_TEXT = {
    "dual_ma": "短均线 > 长均线则持有，否则空仓。",
    "sma_regression_filter": "在 dual_ma 基础上叠加线性回归斜率 > 0 过滤弱趋势。",
    "rsi_mean_reversion": "RSI 低于阈值买入，RSI 高于阈值卖出，偏均值回归。",
    "bollinger_breakout": "收盘突破布林上轨入场，跌破中轨离场，偏趋势突破。",
    "enhanced_breakout_portfolio": "组合增强突破：20日调仓、10日冷却、最小交易金额、1%突破过滤、滑点+手续费。",
}


def load_bars(date_from: str, date_to: str, symbol_filter: str | None = None) -> pd.DataFrame:
    where = [f"trade_date >= toDate('{date_from}')", f"trade_date <= toDate('{date_to}')"]
    if symbol_filter:
        where.append(f"symbol = '{symbol_filter}'")
    where_sql = " and ".join(where)
    return query_df(
        f"""
        select trade_date, symbol, open, high, low, close, volume, amount, ret_1d, ret_5d
        from daily_bars
        where {where_sql}
        order by trade_date, symbol
        """
    )


def sidebar_filters():
    st.sidebar.header("筛选")
    date_from = st.sidebar.date_input("开始日期", value=pd.to_datetime("2023-01-01")).strftime("%Y-%m-%d")
    date_to = st.sidebar.date_input("结束日期", value=pd.Timestamp.today()).strftime("%Y-%m-%d")
    symbol_text = st.sidebar.text_input("股票代码（可选，例 US:AAPL）", value="")
    return date_from, date_to, symbol_text.strip()


def render_strategy_center():
    st.subheader("策略中心")
    info = list_strategies().copy()
    if not info.empty:
        info["交易逻辑"] = info["strategy_name"].map(STRATEGY_LOGIC_TEXT)
    st.dataframe(info, use_container_width=True)

    st.markdown("### 组合策略（增强突破）")
    st.write(STRATEGY_LOGIC_TEXT["enhanced_breakout_portfolio"])
    st.code(
        """
核心参数：
- initial_capital
- top_n
- rebalance_days
- cooldown_days
- min_trade_ratio
- breakout_filter
- slippage
- fee_rate
- lot_size
""".strip()
    )


def render_backtest_board(date_from: str, date_to: str):
    st.subheader("回测看板")

    mode = st.radio("回测模式", ["single_symbol", "portfolio"], horizontal=True)

    if mode == "single_symbol":
        strategy_name = st.selectbox(
            "策略",
            ["dual_ma", "sma_regression_filter", "rsi_mean_reversion", "bollinger_breakout"],
            index=1,
        )
        symbol = st.text_input("回测标的", value="US:AAPL")
        params_text = st.text_area(
            "参数(JSON)",
            value=json.dumps({"total_capital": 1_000_000, "short_win": 10, "long_win": 30, "reg_win": 20}, ensure_ascii=False),
            height=100,
        )

        if st.button("运行单票回测", type="primary"):
            bars = load_bars(date_from, date_to, symbol_filter=symbol)
            if bars.empty:
                st.warning("没有查到该标的数据")
                return

            params = json.loads(params_text)
            bt, metrics = run_strategy_backtest(bars, strategy_name=strategy_name, params=params)

            st.write(pd.DataFrame([{"symbol": symbol, **metrics}]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["trade_date"], y=bt["benchmark_cum_return"], name="Benchmark CumReturn"))
            fig.add_trace(go.Scatter(x=bt["trade_date"], y=bt["strategy_cum_return"], name="Strategy CumReturn"))
            st.plotly_chart(fig, use_container_width=True)

            trades = bt[bt["signal_type"].isin(["BUY", "SELL"])][["trade_date", "signal_type", "close", "position"]].copy()
            trades = trades.rename(columns={"close": "price"})
            st.markdown("#### 交易记录")
            st.dataframe(trades, use_container_width=True)

    else:
        params_text = st.text_area(
            "组合参数(JSON)",
            value=json.dumps(
                {
                    "initial_capital": 1_000_000,
                    "top_n": 10,
                    "rebalance_days": 20,
                    "cooldown_days": 10,
                    "min_trade_ratio": 0.02,
                    "breakout_filter": 1.01,
                    "slippage": 0.001,
                    "fee_rate": 0.0003,
                    "lot_size": 100,
                    "log_level": "INFO",
                },
                ensure_ascii=False,
            ),
            height=180,
        )

        if st.button("运行组合回测", type="primary"):
            bars = load_bars(date_from, date_to)
            if bars.empty:
                st.warning("当前日期区间没有数据")
                return

            params = json.loads(params_text)
            curve, trades, metrics = run_enhanced_breakout_portfolio_backtest(bars, params=params)

            st.write(pd.DataFrame([metrics]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["trade_date"], y=curve["benchmark_cum_return"], name="Benchmark CumReturn"))
            fig.add_trace(go.Scatter(x=curve["trade_date"], y=curve["strategy_cum_return"], name="Strategy CumReturn"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 交易记录")
            st.dataframe(trades, use_container_width=True)

            logs = curve.attrs.get("logs", pd.DataFrame())
            if isinstance(logs, pd.DataFrame) and not logs.empty:
                st.markdown("#### 策略日志")
                st.dataframe(logs.tail(200), use_container_width=True)


def render_data_explorer(date_from: str, date_to: str, symbol_text: str):
    st.subheader("ClickHouse 数据浏览")

    bars = load_bars(date_from, date_to, symbol_filter=symbol_text or None)
    st.markdown("#### 日线数据")
    st.dataframe(bars.tail(1000), use_container_width=True)

    if not bars.empty:
        st.markdown("#### 收盘价走势")
        if symbol_text:
            fig = px.line(bars, x="trade_date", y="close", title=f"{symbol_text} Close")
        else:
            sample = bars[bars["symbol"].isin(sorted(bars["symbol"].unique())[:8])]
            fig = px.line(sample, x="trade_date", y="close", color="symbol", title="Close (sample symbols)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 已存储交易信号")
    signal_df = query_df(
        """
        select trade_date, symbol, strategy_name, signal_type, price, position
        from strategy_signals
        order by trade_date desc
        limit 1000
        """
    )
    st.dataframe(signal_df, use_container_width=True)


def main():
    date_from, date_to, symbol_text = sidebar_filters()

    tab1, tab2, tab3 = st.tabs(["策略中心", "回测看板", "数据浏览"])
    with tab1:
        render_strategy_center()
    with tab2:
        render_backtest_board(date_from, date_to)
    with tab3:
        render_data_explorer(date_from, date_to, symbol_text)


if __name__ == "__main__":
    main()
