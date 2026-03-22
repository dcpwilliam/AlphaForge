from __future__ import annotations

import numpy as np
import pandas as pd


def calc_technical_factors(df: pd.DataFrame) -> pd.DataFrame:
    """输入包含 symbol/trade_date/close/volume，输出基础因子。"""
    data = df.sort_values(["symbol", "trade_date"]).copy()
    grp = data.groupby("symbol", group_keys=False)

    data["mom_5"] = grp["close"].pct_change(5)
    data["mom_20"] = grp["close"].pct_change(20)

    ret = grp["close"].pct_change()
    data["vol_20"] = ret.groupby(data["symbol"]).rolling(20).std().reset_index(level=0, drop=True)

    data["turnover_z"] = (
        grp["volume"].transform(lambda s: (s - s.rolling(20).mean()) / (s.rolling(20).std() + 1e-12))
    )

    data["forward_ret_5d"] = grp["close"].pct_change(5).shift(-5)

    factor_cols = ["mom_5", "mom_20", "vol_20", "turnover_z"]
    melted = data[["trade_date", "symbol", "forward_ret_5d", *factor_cols]].melt(
        id_vars=["trade_date", "symbol", "forward_ret_5d"],
        value_vars=factor_cols,
        var_name="factor_name",
        value_name="factor_value",
    )

    return melted.replace([np.inf, -np.inf], np.nan).dropna(subset=["factor_value", "forward_ret_5d"])


def cross_sectional_ic(factor_df: pd.DataFrame) -> pd.DataFrame:
    """每日横截面 IC（Spearman）"""
    out = []
    for (dt, fname), part in factor_df.groupby(["trade_date", "factor_name"]):
        if part["factor_value"].nunique() < 3:
            continue
        ic = part["factor_value"].corr(part["forward_ret_5d"], method="spearman")
        out.append({"trade_date": dt, "factor_name": fname, "ic": ic})

    return pd.DataFrame(out)
