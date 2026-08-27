import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
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
    start=datetime.now(timezone.utc) - timedelta(days=30),
    feed=DataFeed.IEX,
)

bars = client.get_stock_bars(request).df

timestamps = bars.index.get_level_values("timestamp")
new_york_dates = timestamps.tz_convert("America/New_York").date
today_new_york = datetime.now(ZoneInfo("America/New_York")).date()

completed_bars = bars[new_york_dates < today_new_york]

print("AAPL 最近30天日K线：")
print(completed_bars.tail(10))
