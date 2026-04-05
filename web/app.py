from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quant_research.db import query_df
from quant_research.agent_client import chat_completion, load_agent_config
from quant_research.oidc_client import (
    build_authorize_url,
    exchange_code_for_token,
    fetch_me,
    generate_state,
    load_oidc_config,
    normalize_me_profile,
)
from quant_research.strategies import (
    list_strategies,
    run_enhanced_breakout_portfolio_backtest,
    run_pair_spread_bollinger_backtest,
    run_strategy_backtest,
    validate_context_python_strategy,
)
from quant_research.web_store import (
    buy_asset,
    create_asset,
    init_store,
    list_assets,
    purchased_asset_ids,
)


st.set_page_config(page_title="AlphaForge Strategy Web", layout="wide")
logo_path = Path(__file__).resolve().parent / "pic" / "dcp-logo.png"
assets_dir = Path(__file__).resolve().parent / "assets"
st.title("AlphaForge 策略研究看板")


STRATEGY_LOGIC_TEXT = {
    "dual_ma": "短均线 > 长均线则持有，否则空仓。",
    "sma_regression_filter": "在 dual_ma 基础上叠加线性回归斜率 > 0 过滤弱趋势。",
    "rsi_mean_reversion": "RSI 低于阈值买入，RSI 高于阈值卖出，偏均值回归。",
    "bollinger_breakout": "收盘突破布林上轨入场，跌破中轨离场，偏趋势突破。",
    "enhanced_breakout_portfolio": "组合增强突破：20日调仓、10日冷却、最小交易金额、1%突破过滤、滑点+手续费。",
    "pair_spread_bollinger": "配对价差布林：spread=priceA-priceB，突破上下轨开仓，回归轨道内平仓。",
}

DEFAULT_CONTEXT_STRATEGY_CODE = """#coding:gbk
import numpy as np

def init(ContextInfo):
    ContextInfo.trade_pair = ['rb00.SF', 'hc00.SF']
    ContextInfo.position_tag = {'long': False, 'short': False}
    ContextInfo.set_universe(ContextInfo.trade_pair)
    ContextInfo.accid = 'demo'

def handlebar(ContextInfo):
    closes = ContextInfo.get_market_data(['close'], stock_code=ContextInfo.trade_pair, period=ContextInfo.period, count=31)
    up_closes = closes[ContextInfo.trade_pair[0]]['close']
    down_closes = closes[ContextInfo.trade_pair[1]]['close']
    spread = up_closes[:-1] - down_closes[:-1]
    up = np.mean(spread) + 0.5 * np.std(spread)
    down = np.mean(spread) - 0.5 * np.std(spread)
    spread_now = up_closes[-1] - down_closes[-1]
"""

def can_view_asset(viewer_id: str, asset: dict, purchased_ids: set[int], user_ctx: dict) -> bool:
    if viewer_id == asset["owner_id"]:
        return True
    # 策略/因子权限由 OIDC /me 返回的授权 id 决定
    asset_id_num = str(asset.get("asset_id_num", ""))
    asset_id_code = str(asset.get("asset_id", ""))
    if asset["asset_type"] == "strategy":
        allowed = set(user_ctx.get("strategy_ids", set()))
        if asset_id_num in allowed or asset_id_code in allowed:
            return True
        return False
    if asset["asset_type"] == "factor":
        allowed = set(user_ctx.get("factor_ids", set()))
        if asset_id_num in allowed or asset_id_code in allowed:
            return True
        return False
    if asset["visibility"] == "private":
        return False
    if not asset["is_paid"]:
        return True
    return asset["asset_id_num"] in purchased_ids


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


def _sql_quote(s: str) -> str:
    return s.replace("'", "''")


def load_bars_symbols(date_from: str, date_to: str, symbols: list[str]) -> pd.DataFrame:
    safe_symbols = [f"'{_sql_quote(x)}'" for x in symbols if x.strip()]
    if not safe_symbols:
        return pd.DataFrame()
    symbols_sql = ",".join(safe_symbols)
    return query_df(
        f"""
        select trade_date, symbol, open, high, low, close, volume, amount, ret_1d, ret_5d
        from daily_bars
        where trade_date >= toDate('{date_from}')
          and trade_date <= toDate('{date_to}')
          and symbol in ({symbols_sql})
        order by trade_date, symbol
        """
    )


def _handle_oidc_callback() -> None:
    params = st.query_params
    code = params.get("code")
    state = params.get("state")
    if not code:
        return
    expected_state = st.session_state.get("oidc_state", "")
    if expected_state and state and state != expected_state:
        st.sidebar.error("OIDC 登录状态校验失败，请重新登录。")
        return
    try:
        cfg = load_oidc_config("agent.config")
        token = exchange_code_for_token(cfg, code)
        access_token = str(token.get("access_token", "")).strip()
        if not access_token:
            raise RuntimeError(f"token 响应缺少 access_token: {token}")
        me = fetch_me(cfg, access_token)
        st.session_state.oidc_token = token
        st.session_state.oidc_me = me
        st.session_state.oidc_user = normalize_me_profile(me)
        st.query_params.clear()
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"OIDC 登录失败: {exc}")


def sidebar_filters() -> tuple[str, str, str, dict]:
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width=220)
    st.sidebar.header("全局筛选")
    date_from = st.sidebar.date_input("开始日期", value=pd.to_datetime("2023-01-01")).strftime("%Y-%m-%d")
    date_to = st.sidebar.date_input("结束日期", value=pd.Timestamp.today()).strftime("%Y-%m-%d")
    symbol_text = st.sidebar.text_input("股票代码（可选，例 US:AAPL）", value="")
    st.sidebar.markdown("---")
    st.sidebar.subheader("用户模块（OIDC）")

    _handle_oidc_callback()
    cfg = load_oidc_config("agent.config")
    if "oidc_state" not in st.session_state:
        st.session_state.oidc_state = generate_state()
    if "oidc_user" not in st.session_state:
        st.session_state.oidc_user = None

    current_user = st.session_state.oidc_user
    if not current_user:
        auth_url = build_authorize_url(cfg, st.session_state.oidc_state)
        st.sidebar.link_button("OIDC 登录", auth_url, use_container_width=True)
        st.sidebar.caption(f"issuer: {cfg.issuer}")
        st.sidebar.warning("当前未登录，默认只允许创建/查看自己资产。")
        current_user = {
            "user_id": "anonymous",
            "display_name": "anonymous",
            "balance": 0.0,
            "strategy_ids": set(),
            "factor_ids": set(),
            "raw_me": {},
        }
    else:
        st.sidebar.success(f'已登录: {current_user["display_name"]} ({current_user["user_id"]})')
        st.sidebar.metric("账户余额", f'{float(current_user.get("balance", 0.0)):.2f}')
        st.sidebar.caption(f"策略权限数: {len(set(current_user.get('strategy_ids', set())))}")
        st.sidebar.caption(f"因子权限数: {len(set(current_user.get('factor_ids', set())))}")
        c1, c2 = st.sidebar.columns(2)
        with c1:
            if st.button("刷新权限", key="oidc_refresh_me"):
                try:
                    token = st.session_state.get("oidc_token", {})
                    access_token = str(token.get("access_token", "")).strip()
                    me = fetch_me(cfg, access_token)
                    st.session_state.oidc_me = me
                    st.session_state.oidc_user = normalize_me_profile(me)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.sidebar.error(f"刷新失败: {exc}")
        with c2:
            if st.button("退出登录", key="oidc_logout"):
                for k in ("oidc_token", "oidc_me", "oidc_user", "oidc_state"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.query_params.clear()
                st.rerun()

    return date_from, date_to, symbol_text.strip(), current_user


def render_assets_market(asset_type: str, viewer_id: str, title: str, user_ctx: dict) -> None:
    st.markdown(f"### {title}")
    assets = list_assets(asset_type)
    if not assets:
        st.info("暂无可展示资产。")
        return
    purchased_ids = purchased_asset_ids(viewer_id)
    for asset in assets:
        owner = asset["owner_id"]
        paid_text = "付费" if asset["is_paid"] else "免费"
        with st.expander(f'[{asset["asset_id"]}] {asset["title"]} | {owner} | {paid_text} | {asset["visibility"]}'):
            st.caption(f'创建时间: {asset["created_at"]}')
            unlocked = can_view_asset(viewer_id, asset, purchased_ids, user_ctx)
            if not unlocked:
                if asset["asset_type"] in {"strategy", "factor"}:
                    st.warning("当前账号在 OIDC /me 中没有该资产权限。")
                else:
                    st.warning(f'该资产需购买后查看，价格: {asset["price"]:.2f}')
                    if st.button("购买并解锁", key=f'buy_{asset["asset_id"]}'):
                        ok, msg = buy_asset(viewer_id, asset["asset_id_num"])
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.json(asset["content"])


def render_strategy_tab(current_user_id: str, user_ctx: dict) -> None:
    st.subheader("策略")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("### 内置策略")
        info = list_strategies().copy()
        if not info.empty:
            info["交易逻辑"] = info["strategy_name"].map(STRATEGY_LOGIC_TEXT)
        st.dataframe(info, width='stretch')

    with right:
        st.markdown("### 创建策略")
        with st.form("strategy_create_form"):
            name = st.text_input("策略标识", value="my_custom_strategy")
            display = st.text_input("策略名称", value="我的策略")
            category = st.selectbox("策略类型", ["trend", "mean_reversion", "breakout", "multi_factor", "other"], index=0)
            logic = st.text_area("交易逻辑说明", value="输入你的买卖逻辑说明")
            python_code = st.text_area("Python策略代码（可选）", value=DEFAULT_CONTEXT_STRATEGY_CODE, height=260)
            visibility = st.selectbox("可见性", ["private", "public"], index=0)
            is_paid = st.checkbox("设为付费查看", value=False)
            price = st.number_input("付费价格", min_value=0.0, value=19.9, step=1.0, disabled=not is_paid)
            default_params_text = st.text_area(
                "默认参数(JSON)",
                value=json.dumps({"total_capital": 1_000_000, "short_win": 10, "long_win": 30}, ensure_ascii=False),
                height=120,
            )
            submitted = st.form_submit_button("保存策略模板")

        if submitted:
            try:
                params = json.loads(default_params_text)
                validation = None
                if python_code.strip():
                    validation = validate_context_python_strategy(python_code)
                    if not validation.get("valid", False):
                        st.error("Python 策略代码检查未通过，请先修复后再保存。")
                        st.json(validation)
                        return
                create_asset(
                    owner_id=current_user_id,
                    asset_type="strategy",
                    title=display,
                    content={
                        "strategy_name": name,
                        "display_name": display,
                        "category": category,
                        "logic": logic,
                        "default_params": params,
                        "python_code": python_code.strip(),
                        "validation": validation,
                    },
                    is_paid=is_paid,
                    price=price if is_paid else 0.0,
                    visibility=visibility,
                )
                st.success(f"已保存策略模板：{name}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"参数 JSON 解析失败: {exc}")

    st.markdown("### 自定义策略模板")
    strategy_assets = [a for a in list_assets("strategy") if a["owner_id"] == current_user_id]
    custom_rows = []
    for asset in strategy_assets:
        content = asset["content"]
        custom_rows.append(
            {
                "asset_id": asset["asset_id"],
                "strategy_name": content.get("strategy_name"),
                "display_name": content.get("display_name"),
                "category": content.get("category"),
                "logic": content.get("logic"),
                "default_params": content.get("default_params"),
                "has_python_code": bool(content.get("python_code")),
                "visibility": asset["visibility"],
                "is_paid": asset["is_paid"],
                "price": asset["price"],
            }
        )
    custom_df = pd.DataFrame(custom_rows)
    if custom_df.empty:
        st.info("还没有自定义策略模板。")
    else:
        st.dataframe(custom_df, width='stretch')
    render_assets_market("strategy", current_user_id, "策略资产广场", user_ctx)


def render_backtest_tab(date_from: str, date_to: str, current_user_id: str, user_ctx: dict) -> None:
    st.subheader("回测")

    all_strategy_assets = list_assets("strategy")
    purchased_ids = purchased_asset_ids(current_user_id)
    permitted_strategy_assets = [a for a in all_strategy_assets if can_view_asset(current_user_id, a, purchased_ids, user_ctx)]

    st.markdown("### 可用策略（有权限）")
    if permitted_strategy_assets:
        strategy_rows = []
        for a in permitted_strategy_assets:
            c = a["content"]
            strategy_rows.append(
                {
                    "asset_id": a["asset_id"],
                    "owner": a["owner_id"],
                    "display_name": c.get("display_name", a["title"]),
                    "strategy_name": c.get("strategy_name", ""),
                    "category": c.get("category", ""),
                    "has_python_code": bool(c.get("python_code")),
                    "visibility": a["visibility"],
                    "is_paid": a["is_paid"],
                    "price": a["price"],
                }
            )
        st.dataframe(pd.DataFrame(strategy_rows), width='stretch')
    else:
        st.info("当前没有可用策略。你可以在策略页创建，或购买公开付费策略后回来回测。")

    mode = st.radio("回测模式", ["single_symbol", "portfolio", "custom_python_pair"], horizontal=True)

    if mode == "single_symbol":
        strategy_name = st.selectbox(
            "策略",
            ["dual_ma", "sma_regression_filter", "rsi_mean_reversion", "bollinger_breakout"],
            index=1,
        )
        symbol = st.text_input("回测标的", value="US:AAPL")

        preset = {
            "total_capital": 1_000_000,
            "short_win": 10,
            "long_win": 30,
            "reg_win": 20,
        }
        params_text = st.text_area("回测参数(JSON)", value=json.dumps(preset, ensure_ascii=False), height=130)
        save_result = st.checkbox("保存回测结果到我的资产", value=True, key="save_bt_single")
        visibility = st.selectbox("结果可见性", ["private", "public"], index=0, key="bt_single_visibility")
        is_paid = st.checkbox("结果设为付费查看", value=False, key="bt_single_paid")
        price = st.number_input("结果价格", min_value=0.0, value=9.9, step=1.0, disabled=not is_paid, key="bt_single_price")

        if st.button("运行单票回测", type="primary"):
            bars = load_bars(date_from, date_to, symbol_filter=symbol)
            if bars.empty:
                st.warning("没有查到该标的数据")
                return

            try:
                params = json.loads(params_text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"参数 JSON 解析失败: {exc}")
                return

            bt, metrics = run_strategy_backtest(bars, strategy_name=strategy_name, params=params)
            st.write(pd.DataFrame([{"symbol": symbol, **metrics}]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["trade_date"], y=bt["benchmark_cum_return"], name="Benchmark CumReturn"))
            fig.add_trace(go.Scatter(x=bt["trade_date"], y=bt["strategy_cum_return"], name="Strategy CumReturn"))
            st.plotly_chart(fig, width='stretch')

            trades = bt[bt["signal_type"].isin(["BUY", "SELL"])][["trade_date", "signal_type", "close", "position"]].copy()
            trades = trades.rename(columns={"close": "price"})
            st.markdown("#### 交易记录")
            st.dataframe(trades, width='stretch')
            if save_result:
                aid = create_asset(
                    owner_id=current_user_id,
                    asset_type="backtest",
                    title=f"{strategy_name}-{symbol}-{date_from}~{date_to}",
                    content={
                        "mode": "single_symbol",
                        "symbol": symbol,
                        "strategy_name": strategy_name,
                        "params": params,
                        "metrics": metrics,
                        "trade_count": int(len(trades)),
                    },
                    is_paid=is_paid,
                    price=price if is_paid else 0.0,
                    visibility=visibility,
                )
                st.success(f"回测结果已保存到资产库: {aid}")

    elif mode == "portfolio":
        preset = {
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
        }
        params_text = st.text_area("组合参数(JSON)", value=json.dumps(preset, ensure_ascii=False), height=220)
        save_result = st.checkbox("保存组合回测到我的资产", value=True, key="save_bt_port")
        visibility = st.selectbox("结果可见性", ["private", "public"], index=0, key="bt_port_visibility")
        is_paid = st.checkbox("结果设为付费查看", value=False, key="bt_port_paid")
        price = st.number_input("结果价格", min_value=0.0, value=19.9, step=1.0, disabled=not is_paid, key="bt_port_price")

        if st.button("运行组合回测", type="primary"):
            bars = load_bars(date_from, date_to)
            if bars.empty:
                st.warning("当前日期区间没有数据")
                return

            try:
                params = json.loads(params_text)
            except Exception as exc:  # noqa: BLE001
                st.error(f"参数 JSON 解析失败: {exc}")
                return

            curve, trades, metrics = run_enhanced_breakout_portfolio_backtest(bars, params=params)
            st.write(pd.DataFrame([metrics]))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve["trade_date"], y=curve["benchmark_cum_return"], name="Benchmark CumReturn"))
            fig.add_trace(go.Scatter(x=curve["trade_date"], y=curve["strategy_cum_return"], name="Strategy CumReturn"))
            st.plotly_chart(fig, width='stretch')

            st.markdown("#### 交易记录")
            st.dataframe(trades, width='stretch')

            logs = curve.attrs.get("logs", pd.DataFrame())
            if isinstance(logs, pd.DataFrame) and not logs.empty:
                st.markdown("#### 策略日志")
                st.dataframe(logs.tail(200), width='stretch')
            if save_result:
                aid = create_asset(
                    owner_id=current_user_id,
                    asset_type="backtest",
                    title=f"portfolio-{date_from}~{date_to}",
                    content={
                        "mode": "portfolio",
                        "params": params,
                        "metrics": metrics,
                        "trade_count": int(len(trades)),
                    },
                    is_paid=is_paid,
                    price=price if is_paid else 0.0,
                    visibility=visibility,
                )
                st.success(f"组合回测已保存到资产库: {aid}")

    else:
        st.markdown("### 自定义 Python 配对策略回测")
        strategy_assets = [a for a in permitted_strategy_assets if a["content"].get("python_code")]
        if not strategy_assets:
            st.info("当前你有权限的策略中，没有可用于该模式的 Python 策略。")
        else:
            options = {f'{a["asset_id"]} | {a["title"]}': a for a in strategy_assets}
            selected = st.selectbox("选择策略资产", list(options.keys()))
            selected_asset = options[selected]
            python_code = selected_asset["content"].get("python_code", "")

            check = validate_context_python_strategy(python_code)
            if check.get("valid"):
                st.success("策略代码检查通过")
            else:
                st.error("策略代码检查未通过")
            with st.expander("查看语法/合法性检查报告"):
                st.json(check)

            col1, col2 = st.columns(2)
            with col1:
                symbol_a = st.text_input("标的A", value="rb00.SF")
                window = st.number_input("窗口期", min_value=5, max_value=252, value=30)
                initial_capital = st.number_input("总金额", min_value=1000.0, value=1_000_000.0, step=10_000.0)
            with col2:
                symbol_b = st.text_input("标的B", value="hc00.SF")
                band_k = st.number_input("标准差倍数", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
                trade_notional = st.number_input("单边名义金额", min_value=1000.0, value=200_000.0, step=10_000.0)

            run_btn = st.button("运行自定义配对回测", type="primary")
            if run_btn:
                if not check.get("valid"):
                    st.error("代码检查未通过，已阻止回测。")
                    return
                bars = load_bars_symbols(date_from, date_to, [symbol_a, symbol_b])
                if bars.empty:
                    st.warning("当前时间段没有配对标的数据")
                    return
                params = {
                    "symbol_a": symbol_a,
                    "symbol_b": symbol_b,
                    "window": int(window),
                    "band_k": float(band_k),
                    "initial_capital": float(initial_capital),
                    "trade_notional": float(trade_notional),
                }
                curve, trades, metrics = run_pair_spread_bollinger_backtest(bars, params=params)
                st.write(pd.DataFrame([metrics]))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=curve["trade_date"], y=curve["benchmark_cum_return"], name="Benchmark CumReturn"))
                fig.add_trace(go.Scatter(x=curve["trade_date"], y=curve["strategy_cum_return"], name="Strategy CumReturn"))
                st.plotly_chart(fig, width='stretch')
                st.markdown("#### 交易记录")
                st.dataframe(trades, width='stretch')

                aid = create_asset(
                    owner_id=current_user_id,
                    asset_type="backtest",
                    title=f"custom_pair-{symbol_a}-{symbol_b}-{date_from}~{date_to}",
                    content={
                        "mode": "custom_python_pair",
                        "strategy_asset_id": selected_asset["asset_id"],
                        "params": params,
                        "metrics": metrics,
                        "trade_count": int(len(trades)),
                    },
                    is_paid=False,
                    price=0.0,
                    visibility="private",
                )
                st.success(f"自定义回测结果已保存到资产库: {aid}")

    render_assets_market("backtest", current_user_id, "回测资产广场", user_ctx)


def render_factor_tab(date_from: str, date_to: str, symbol_text: str, current_user_id: str, user_ctx: dict) -> None:
    st.subheader("因子研究")

    st.markdown("### 创建因子")
    with st.form("factor_create_form"):
        factor_name = st.text_input("因子名称", value="my_factor")
        factor_type = st.selectbox("因子类型", ["momentum", "volatility", "turnover", "mean_reversion"], index=0)
        window = st.number_input("窗口期", min_value=2, max_value=252, value=20)
        fwd_days = st.number_input("未来收益期(天)", min_value=1, max_value=60, value=5)
        visibility = st.selectbox("因子可见性", ["private", "public"], index=0)
        is_paid = st.checkbox("因子设为付费查看", value=False)
        price = st.number_input("因子价格", min_value=0.0, value=29.9, step=1.0, disabled=not is_paid)
        create_clicked = st.form_submit_button("生成并分析")

    symbol = symbol_text or st.text_input("研究标的", value="US:AAPL")
    bars = load_bars(date_from, date_to, symbol_filter=symbol)

    if bars.empty:
        st.warning("当前筛选条件下没有可用数据。")
        return

    bars = bars.sort_values("trade_date").copy()

    if create_clicked:
        if factor_type == "momentum":
            bars[factor_name] = bars["close"].pct_change(int(window))
        elif factor_type == "volatility":
            bars[factor_name] = bars["close"].pct_change().rolling(int(window)).std()
        elif factor_type == "turnover":
            bars[factor_name] = bars["volume"] / (bars["volume"].rolling(int(window)).mean() + 1e-12)
        else:
            ma = bars["close"].rolling(int(window)).mean()
            bars[factor_name] = (bars["close"] - ma) / (ma + 1e-12)

        bars["forward_ret"] = bars["close"].pct_change(int(fwd_days)).shift(-int(fwd_days))
        factor_df = bars[["trade_date", "close", factor_name, "forward_ret"]].dropna().copy()

        st.markdown("#### 因子样本")
        st.dataframe(factor_df.tail(300), width='stretch')

        ic = factor_df[factor_name].corr(factor_df["forward_ret"], method="spearman")
        st.metric("Spearman IC", f"{ic:.4f}" if pd.notna(ic) else "NaN")

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=factor_df["trade_date"], y=factor_df[factor_name], name=factor_name))
        st.plotly_chart(fig1, width='stretch')

        # 分层收益（5分位）
        tmp = factor_df.copy()
        tmp["bucket"] = pd.qcut(tmp[factor_name].rank(method="first"), 5, labels=False, duplicates="drop")
        grp = tmp.groupby("bucket")["forward_ret"].mean().reset_index()
        fig2 = px.bar(grp, x="bucket", y="forward_ret", title="Factor Quantile Forward Return")
        st.plotly_chart(fig2, width='stretch')
        aid = create_asset(
            owner_id=current_user_id,
            asset_type="factor",
            title=f"{factor_name}-{symbol}",
            content={
                "factor_name": factor_name,
                "factor_type": factor_type,
                "symbol": symbol,
                "window": int(window),
                "fwd_days": int(fwd_days),
                "ic": float(ic) if pd.notna(ic) else None,
                "sample_size": int(len(factor_df)),
            },
            is_paid=is_paid,
            price=price if is_paid else 0.0,
            visibility=visibility,
        )
        st.success(f"因子研究已保存到资产库: {aid}")

    st.markdown("### ClickHouse 股票数据预览")
    st.dataframe(bars.tail(200), width='stretch')
    render_assets_market("factor", current_user_id, "因子资产广场", user_ctx)


def inject_html_resource(resource_name: str) -> None:
    resource_path = assets_dir / resource_name
    if not resource_path.exists():
        return
    st.markdown(resource_path.read_text(encoding="utf-8"), unsafe_allow_html=True)


def render_agent_widget() -> None:
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = [
            {"role": "assistant", "content": "你好，我是 AlphaForge Agent。可以帮你分析策略、回测结果和因子表现。"},
        ]
    if "agent_open" not in st.session_state:
        st.session_state.agent_open = False

    inject_html_resource("agent_widget_effects.html")
    with st.container():
        st.markdown('<div id="af-agent-fab-anchor"></div>', unsafe_allow_html=True)
        if st.button("●", key="agent_fab_toggle", help="AI Agent"):
            st.session_state.agent_open = not st.session_state.agent_open
            st.rerun()

    if st.session_state.agent_open:
        with st.container():
            st.markdown('<div id="af-agent-panel-anchor"></div>', unsafe_allow_html=True)
            cols = st.columns([4, 1])
            with cols[0]:
                st.markdown("### AI Agent")
            with cols[1]:
                if st.button("收起", key="agent_close_btn"):
                    st.session_state.agent_open = False
                    st.rerun()

            cfg = load_agent_config("agent.config")
            st.caption(f"vendor={cfg.vendor} | model={cfg.model} | base_url={cfg.base_url}")

            if cfg.vendor in {"openai", "openai_compatible"} and (not cfg.api_key or cfg.api_key.startswith("${")):
                st.warning("未检测到有效 API Key，请在 `agent.config` 或环境变量 `OPENAI_API_KEY` 中配置。")

            if st.button("清空对话", key="agent_clear_btn"):
                st.session_state.agent_messages = [
                    {"role": "assistant", "content": "对话已清空。你可以继续提问。"},
                ]

            for msg in st.session_state.agent_messages[-12:]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            prompt = st.chat_input("按 Enter 发送消息", key="agent_chat_input")
            if prompt:
                st.session_state.agent_messages.append({"role": "user", "content": prompt.strip()})
                try:
                    with st.spinner("Agent 思考中..."):
                        answer = chat_completion(st.session_state.agent_messages[-20:], cfg)
                except Exception as exc:  # noqa: BLE001
                    answer = f"调用失败: {exc}"
                st.session_state.agent_messages.append({"role": "assistant", "content": answer})
                st.rerun()


def main() -> None:
    init_store()
    date_from, date_to, symbol_text, user_ctx = sidebar_filters()
    current_user_id = str(user_ctx.get("user_id", "anonymous"))

    tab_strategy, tab_backtest, tab_factor = st.tabs(["策略", "回测", "因子研究"])
    with tab_strategy:
        render_strategy_tab(current_user_id, user_ctx)
    with tab_backtest:
        render_backtest_tab(date_from, date_to, current_user_id, user_ctx)
    with tab_factor:
        render_factor_tab(date_from, date_to, symbol_text, current_user_id, user_ctx)
    render_agent_widget()


if __name__ == "__main__":
    main()
