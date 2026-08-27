import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    raise RuntimeError("未读取到 Alpaca API 密钥")

client = StockHistoricalDataClient(api_key, secret_key)

request = StockBarsRequest(
    symbol_or_symbols=["AAPL"],
    timeframe=TimeFrame.Day,
    start=datetime.now(timezone.utc) - timedelta(days=365),
    feed=DataFeed.IEX,
    adjustment=Adjustment.ALL,
)

bars = client.get_stock_bars(request).df.reset_index()

if bars.empty:
    raise RuntimeError("没有获取到 AAPL 的历史行情")

# 排除纽约当天尚未完成的日 K。
today_new_york = datetime.now(ZoneInfo("America/New_York")).date()
new_york_dates = bars["timestamp"].dt.tz_convert(
    "America/New_York"
).dt.date

bars = bars[new_york_dates < today_new_york].copy()
bars = bars.sort_values("timestamp")

# 移动平均线。
bars["sma20"] = bars["close"].rolling(20).mean()
bars["sma50"] = bars["close"].rolling(50).mean()
bars["sma200"] = bars["close"].rolling(200).mean()

# 收益率和年化波动率。
bars["daily_return"] = bars["close"].pct_change()
bars["return_20"] = bars["close"].pct_change(20)
bars["volatility_20"] = (
    bars["daily_return"].rolling(20).std() * np.sqrt(252)
)

# 14日 RSI。
price_change = bars["close"].diff()
gain = price_change.clip(lower=0)
loss = -price_change.clip(upper=0)

average_gain = gain.rolling(14).mean()
average_loss = loss.rolling(14).mean()

relative_strength = average_gain / average_loss.replace(0, np.nan)
bars["rsi14"] = 100 - (100 / (1 + relative_strength))
bars.loc[
    (average_loss == 0) & (average_gain > 0),
    "rsi14",
] = 100

if len(bars) < 200:
    raise RuntimeError("历史数据不足200个交易日")

latest = bars.iloc[-1]
latest_date = latest["timestamp"].tz_convert(
    "America/New_York"
).date()

print(f"股票：AAPL")
print(f"数据日期：{latest_date}")
print(f"收盘价：{latest['close']:.2f}")
print(f"20日均线：{latest['sma20']:.2f}")
print(f"50日均线：{latest['sma50']:.2f}")
print(f"200日均线：{latest['sma200']:.2f}")
print(f"20日收益率：{latest['return_20']:.2%}")
print(f"14日 RSI：{latest['rsi14']:.2f}")
print(f"20日年化波动率：{latest['volatility_20']:.2%}")
print("数据源：Alpaca IEX")
