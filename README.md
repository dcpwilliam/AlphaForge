# AlphaForge Quant Research (ClickHouse + Python + Jupyter)

这个工程用于本地量化研究场景，提供：
- 本地 ClickHouse 数据库（Docker）
- Python 研究工程（可扩展）
- 数据导入脚本
- 数据分析与因子探查 Notebook

## 1. 环境准备

要求：
- Docker / Docker Compose
- Python 3.9+

复制环境变量：
```bash
cp .env.example .env
```

## 2. 启动 ClickHouse

```bash
make up
```

启动后可访问：
- HTTP: `http://127.0.0.1:8123`
- Native TCP: `127.0.0.1:9000`

数据库与表会自动初始化（`clickhouse/init/01_schema.sql`）。

## 3. 安装 Python 依赖

```bash
make venv
make install
```

## 4. 拉取真实行情并导入（50/100 股票池）

```bash
make fetch-real
make load
```

使用 100 只股票池：
```bash
make fetch-real-100
make load
```

可选：指定股票与时间区间（支持美股/港股/沪深/新加坡）
```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/fetch_real_market_data.py \
  --symbols "US:AAPL,HK:0700,CN:600519,SG:D05" \
  --start 2022-01-01 \
  --end 2026-03-19
```

## 5. 增量更新（推荐日更）

手动增量更新（默认 100 池）：
```bash
make update-incremental
```

自定义池和回看天数：
```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/incremental_update_clickhouse.py \
  --pool-size 50 \
  --lookback-days 30
```

安装 macOS 定时任务（`launchd`，每天 08:10 自动更新）：
```bash
make schedule-install
```

卸载定时任务：
```bash
make schedule-uninstall
```

## 6. 打开 Jupyter

```bash
make notebook
```

## 7. 启动 Web 看板（Python / Streamlit）

```bash
make web
```

访问地址：
- Web: `http://127.0.0.1:8501`

说明：
- Web 工程入口：`web/app.py`
- 用户/资产/购买记录长期保存到 `data/app_store.db`
- Agent 配置来自根目录 `agent.config`，默认 vendor 为 `openai`
- 用户模块使用 OIDC（Authing），登录后从 `/oidc/me` 拉取用户信息（如资金、策略权限ID、因子权限ID）

OIDC 环境变量（建议放到 `.env`）：
```bash
OIDC_CLIENT_ID=69ba3eccdb057f1398c742c5
OIDC_CLIENT_SECRET=你的 App Secret
OIDC_REDIRECT_URI=http://127.0.0.1:8501
```

Notebook 说明：
- `notebooks/01_data_import.ipynb`: 数据检查 + 导入验证
- `notebooks/02_data_analysis.ipynb`: 行情分布、收益分析、成交量分析
- `notebooks/03_factor_exploration.ipynb`: 因子 IC 分析、分层收益、可视化
- `notebooks/04_multi_market_fetch_and_backtest.ipynb`: 多市场拉取、统一入库、策略回归/回测与买卖点标记

## 8. 核心目录

```text
clickhouse/
  init/01_schema.sql
notebooks/
scripts/
  fetch_real_market_data.py
  incremental_update_clickhouse.py
  load_to_clickhouse.py
  install_launchd_job.sh
  run_incremental_update.sh
web/
  app.py
src/quant_research/
  config.py
  db.py
  factors.py
  market_data.py
  strategies.py
  universe.py
```

## 9. 后续扩展建议

- 增加交易日历与复权处理
- 增加行业中性、市值中性处理
- 增加回测框架（如 vectorbt / backtrader）
