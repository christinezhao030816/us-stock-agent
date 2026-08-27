import os

from dotenv import load_dotenv
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


load_dotenv()

api_key = os.getenv("ALPACA_API_KEY")
secret_key = os.getenv("ALPACA_SECRET_KEY")

if not api_key or not secret_key:
    raise RuntimeError("未读取到 Alpaca API 密钥")

client = StockHistoricalDataClient(api_key, secret_key)

request = StockLatestQuoteRequest(
    symbol_or_symbols=["AAPL"],
    feed=DataFeed.IEX,
)

quotes = client.get_stock_latest_quote(request)
quote = quotes["AAPL"]

print("股票：AAPL")
print(f"买价：{quote.bid_price}")
print(f"卖价：{quote.ask_price}")
print(f"数据时间：{quote.timestamp}")
print("数据源：Alpaca IEX")
