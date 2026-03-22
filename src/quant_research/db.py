from __future__ import annotations

import clickhouse_connect
import pandas as pd

from .config import ClickHouseConfig


_cfg = ClickHouseConfig()


def get_client():
    return clickhouse_connect.get_client(
        host=_cfg.host,
        port=_cfg.port,
        username=_cfg.user,
        password=_cfg.password,
        database=_cfg.database,
    )


def query_df(sql: str) -> pd.DataFrame:
    client = get_client()
    try:
        return client.query_df(sql)
    finally:
        client.close()


def insert_df(table: str, df: pd.DataFrame):
    client = get_client()
    try:
        client.insert_df(table=table, df=df)
    finally:
        client.close()
