import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST", "127.0.0.1")
    port: int = int(os.getenv("CLICKHOUSE_PORT", "8123"))
    user: str = os.getenv("CLICKHOUSE_USER", "quant_user")
    password: str = os.getenv("CLICKHOUSE_PASSWORD", "quant_pass")
    database: str = os.getenv("CLICKHOUSE_DATABASE", "quant")
