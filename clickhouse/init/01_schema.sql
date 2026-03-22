CREATE DATABASE IF NOT EXISTS quant;

CREATE TABLE IF NOT EXISTS quant.daily_bars
(
    trade_date Date,
    symbol LowCardinality(String),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    amount Float64,
    vwap Float64,
    ret_1d Float64,
    ret_5d Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (symbol, trade_date);

CREATE TABLE IF NOT EXISTS quant.factor_values
(
    trade_date Date,
    symbol LowCardinality(String),
    factor_name LowCardinality(String),
    factor_value Float64,
    forward_ret_5d Float64,
    created_at DateTime DEFAULT now()
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(trade_date)
ORDER BY (factor_name, symbol, trade_date);
