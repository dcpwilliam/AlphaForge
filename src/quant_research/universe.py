from __future__ import annotations

SYMBOL_POOLS: dict[int, list[str]] = {
    50: [
        "US:AAPL", "US:MSFT", "US:NVDA", "US:AMZN", "US:GOOGL", "US:META", "US:TSLA", "US:BRK-B", "US:JPM", "US:V",
        "US:UNH", "US:XOM", "US:JNJ", "US:WMT", "US:PG", "US:MA", "US:HD", "US:COST", "US:ABBV", "US:BAC",
        "US:KO", "US:AVGO", "US:PEP", "US:MRK", "US:CVX",
        "HK:0700", "HK:9988", "HK:3690", "HK:9618", "HK:1810", "HK:0005", "HK:0939", "HK:1398", "HK:2318", "HK:1211",
        "CN:600519", "CN:601318", "CN:600036", "CN:601166", "CN:600900", "CN:300750", "CN:000858", "CN:002594", "CN:000333", "CN:000001",
        "SG:D05", "SG:O39", "SG:U11", "SG:C6L", "SG:S68",
    ],
    100: [
        "US:AAPL", "US:MSFT", "US:NVDA", "US:AMZN", "US:GOOGL", "US:META", "US:TSLA", "US:BRK-B", "US:JPM", "US:V",
        "US:UNH", "US:XOM", "US:JNJ", "US:WMT", "US:PG", "US:MA", "US:HD", "US:COST", "US:ABBV", "US:BAC",
        "US:KO", "US:AVGO", "US:PEP", "US:MRK", "US:CVX", "US:ORCL", "US:CSCO", "US:NFLX", "US:ADBE", "US:CRM",
        "US:ACN", "US:LIN", "US:TMO", "US:MCD", "US:AMD", "US:QCOM", "US:TXN", "US:HON", "US:INTU", "US:AMAT",
        "HK:0700", "HK:9988", "HK:3690", "HK:9618", "HK:1810", "HK:1024", "HK:2382", "HK:0005", "HK:0939", "HK:1398",
        "HK:3988", "HK:2318", "HK:0941", "HK:0883", "HK:0857", "HK:1211", "HK:9999", "HK:1093", "HK:0669", "HK:0388",
        "CN:600519", "CN:601318", "CN:600036", "CN:601166", "CN:600900", "CN:601398", "CN:601288", "CN:601988", "CN:601857", "CN:600000",
        "CN:600030", "CN:600276", "CN:600887", "CN:601012", "CN:300750", "CN:000858", "CN:002594", "CN:000333", "CN:600309", "CN:601899",
        "CN:601888", "CN:002475", "CN:300760", "CN:002142", "CN:000001", "CN:600703", "CN:603259", "CN:688111", "CN:600809", "CN:002415",
        "SG:D05", "SG:O39", "SG:U11", "SG:Z74", "SG:C6L", "SG:C52", "SG:S68", "SG:F34", "SG:A17U", "SG:BN4",
    ],
}


def get_symbol_pool(size: int) -> list[str]:
    if size not in SYMBOL_POOLS:
        raise ValueError(f"Unsupported pool size: {size}, choose from {sorted(SYMBOL_POOLS)}")
    return SYMBOL_POOLS[size]
