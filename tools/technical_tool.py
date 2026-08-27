from typing import Any

import pandas as pd
import yfinance as yf


def _round_value(
    value: float | int | None,
    digits: int = 2,
) -> float | None:
    if value is None or pd.isna(value):
        return None

    return round(float(value), digits)


def _percentage_change(
    series: pd.Series,
    periods: int,
) -> float | None:
    if len(series) <= periods:
        return None

    previous = float(series.iloc[-periods - 1])
    current = float(series.iloc[-1])

    if previous == 0:
        return None

    return round((current / previous - 1) * 100, 2)


def _calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> float | None:
    if len(close) <= period:
        return None

    change = close.diff()
    gains = change.clip(lower=0)
    losses = -change.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    latest_loss = average_loss.iloc[-1]
    latest_gain = average_gain.iloc[-1]

    if pd.isna(latest_loss) or pd.isna(latest_gain):
        return None

    if latest_loss == 0:
        return 100.0

    relative_strength = latest_gain / latest_loss
    rsi = 100 - 100 / (1 + relative_strength)

    return round(float(rsi), 2)


def _price_position(
    current_price: float,
    level: float | None,
) -> float | None:
    if level in {None, 0}:
        return None

    return round((current_price / level - 1) * 100, 2)


def get_technical_metrics(symbol: str) -> dict[str, Any]:
    symbol = symbol.strip().upper()

    ticker = yf.Ticker(symbol)

    history = ticker.history(
        period="18mo",
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if history.empty:
        raise ValueError(f"未找到 {symbol} 的历史行情数据")

    required_columns = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }

    missing_columns = required_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            f"行情数据缺少字段：{sorted(missing_columns)}"
        )

    history = history.dropna(subset=["Close"]).copy()

    if len(history) < 60:
        raise ValueError(
            f"{symbol} 的有效交易日不足，无法进行完整技术分析"
        )

    close = history["Close"].astype(float)
    high = history["High"].astype(float)
    low = history["Low"].astype(float)
    volume = history["Volume"].fillna(0).astype(float)

    current_price = float(close.iloc[-1])
    previous_close = (
        float(close.iloc[-2])
        if len(close) >= 2
        else None
    )

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = (
        close.rolling(200).mean().iloc[-1]
        if len(close) >= 200
        else None
    )

    rsi14 = _calculate_rsi(close, 14)

    latest_volume = float(volume.iloc[-1])
    average_volume_20 = float(volume.tail(20).mean())
    average_volume_60 = float(volume.tail(60).mean())

    volume_ratio_20 = None

    if average_volume_20 > 0:
        volume_ratio_20 = round(
            latest_volume / average_volume_20,
            2,
        )

    recent_20 = history.tail(20)
    recent_60 = history.tail(60)
    recent_252 = history.tail(252)

    support_20 = float(recent_20["Low"].min())
    resistance_20 = float(recent_20["High"].max())

    support_60 = float(recent_60["Low"].min())
    resistance_60 = float(recent_60["High"].max())

    high_52_week = float(recent_252["High"].max())
    low_52_week = float(recent_252["Low"].min())

    daily_change_percent = None

    if previous_close not in {None, 0}:
        daily_change_percent = round(
            (current_price / previous_close - 1) * 100,
            2,
        )

    volume_signal = "数据不足"

    if volume_ratio_20 is not None:
        if volume_ratio_20 >= 1.5:
            volume_signal = "明显放量"
        elif volume_ratio_20 >= 1.1:
            volume_signal = "温和放量"
        elif volume_ratio_20 <= 0.7:
            volume_signal = "明显缩量"
        else:
            volume_signal = "成交量接近20日均量"

    trend_signal = "中性"

    if (
        sma200 is not None
        and not pd.isna(sma200)
        and current_price > sma20 > sma50 > sma200
    ):
        trend_signal = "多头排列"
    elif (
        sma200 is not None
        and not pd.isna(sma200)
        and current_price < sma20 < sma50 < sma200
    ):
        trend_signal = "空头排列"
    elif current_price > sma20 and current_price > sma50:
        trend_signal = "价格位于短中期均线上方"
    elif current_price < sma20 and current_price < sma50:
        trend_signal = "价格位于短中期均线下方"
    else:
        trend_signal = "短中期趋势分化"

    return {
        "symbol": symbol,
        "data_start": history.index[0].date().isoformat(),
        "data_end": history.index[-1].date().isoformat(),
        "trading_days": int(len(history)),
        "current_price_usd": _round_value(current_price),
        "previous_close_usd": _round_value(previous_close),
        "daily_change_percent": daily_change_percent,
        "return_5d_percent": _percentage_change(close, 5),
        "return_20d_percent": _percentage_change(close, 20),
        "return_60d_percent": _percentage_change(close, 60),
        "sma20_usd": _round_value(sma20),
        "sma50_usd": _round_value(sma50),
        "sma200_usd": _round_value(sma200),
        "rsi14": rsi14,
        "latest_volume": int(latest_volume),
        "average_volume_20d": int(average_volume_20),
        "average_volume_60d": int(average_volume_60),
        "volume_ratio_to_20d": volume_ratio_20,
        "volume_signal": volume_signal,
        "support_20d_usd": _round_value(support_20),
        "resistance_20d_usd": _round_value(resistance_20),
        "support_60d_usd": _round_value(support_60),
        "resistance_60d_usd": _round_value(resistance_60),
        "distance_from_support_20d_percent": (
            _price_position(current_price, support_20)
        ),
        "distance_from_resistance_20d_percent": (
            _price_position(current_price, resistance_20)
        ),
        "high_52_week_usd": _round_value(high_52_week),
        "low_52_week_usd": _round_value(low_52_week),
        "distance_from_52_week_high_percent": (
            _price_position(current_price, high_52_week)
        ),
        "distance_from_52_week_low_percent": (
            _price_position(current_price, low_52_week)
        ),
        "trend_signal": trend_signal,
        "support_resistance_method": (
            "20日和60日区间内的最低价作为支撑参考，"
            "最高价作为阻力参考；这些是机械计算结果，"
            "不代表价格一定会在该位置反转"
        ),
        "price_source": "Yahoo Finance via yfinance",
        "calculation_note": (
            "均线、RSI、收益率、成交量比率及支撑阻力"
            "均根据日线历史行情计算"
        ),
    }
