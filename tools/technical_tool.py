import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import yfinance as yf

from tools.ticker_symbols import canonicalize_symbol, ticker_mapping


PRICE_SOURCE = "Yahoo Finance via yfinance"
PRICE_PERIOD = "18mo"
PRICE_INTERVAL = "1d"
AUTO_ADJUST = False
INCLUDE_ACTIONS = False
MAX_TECHNICAL_WINDOW = 252
ROUND_DIGITS = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVIDENCE_DIRECTORY = (
    PROJECT_ROOT / "evaluation" / "technical_evidence"
)


def _round_value(
    value: float | int | None,
    digits: int = ROUND_DIGITS,
) -> float | None:
    if value is None or pd.isna(value):
        return None

    return round(float(value), digits)


def _json_number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None

    if isinstance(value, int):
        return value

    number = float(value)
    return int(number) if number.is_integer() else number


def _percentage_change_raw(
    series: pd.Series,
    periods: int,
) -> float | None:
    if len(series) <= periods:
        return None

    previous = float(series.iloc[-periods - 1])
    current = float(series.iloc[-1])

    if previous == 0:
        return None

    return (current / previous - 1) * 100


def _calculate_rsi_raw(
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
    return float(100 - 100 / (1 + relative_strength))


def _price_position_raw(
    current_price: float,
    level: float | None,
) -> float | None:
    if level in {None, 0}:
        return None

    return (current_price / level - 1) * 100


def _price_position(
    current_price: float,
    level: float | None,
) -> float | None:
    return _round_value(_price_position_raw(current_price, level))


def _date_at(history: pd.DataFrame, index: int) -> str:
    return pd.Timestamp(history.index[index]).date().isoformat()


def _series_reference(
    history: pd.DataFrame,
    field: str,
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    return {
        "dataset": "raw_bars",
        "field": field,
        "start_index": start_index,
        "end_index": end_index,
        "start_date": _date_at(history, start_index),
        "end_date": _date_at(history, end_index),
        "observation_count": end_index - start_index + 1,
    }


def _evidence_entry(
    *,
    formula: str,
    raw_input: dict[str, Any],
    raw_result: Any,
    rounded_result: Any,
    rounding: str,
) -> dict[str, Any]:
    return {
        "formula": formula,
        "raw_input": raw_input,
        "raw_result": _json_number(raw_result),
        "rounded_result": _json_number(rounded_result),
        "rounding": rounding,
    }


def _normalize_history(
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
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

    normalized = history.copy()

    if not isinstance(normalized.index, pd.DatetimeIndex):
        normalized.index = pd.to_datetime(normalized.index)

    timezone_name = (
        str(normalized.index.tz)
        if normalized.index.tz is not None
        else "timezone-naive"
    )

    original_count = len(normalized)
    normalized = normalized.sort_index(kind="stable")
    duplicate_count = int(
        normalized.index.duplicated(keep="last").sum()
    )
    normalized = normalized[
        ~normalized.index.duplicated(keep="last")
    ]

    missing_close_mask = normalized["Close"].isna()
    dropped_close_dates = [
        pd.Timestamp(value).date().isoformat()
        for value in normalized.index[missing_close_mask]
    ]
    normalized = normalized.loc[~missing_close_mask].copy()

    return normalized, {
        "timezone": timezone_name,
        "fetched_row_count": original_count,
        "duplicate_rows_removed": duplicate_count,
        "rows_missing_close_removed": len(dropped_close_dates),
        "removed_close_missing_dates": dropped_close_dates,
    }


def calculate_technical_metrics_from_history(
    history: pd.DataFrame,
    *,
    symbol: str,
    vendor_symbol: str | None = None,
    retrieved_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calculate all technical metrics from one immutable bar snapshot."""

    canonical_symbol = canonicalize_symbol(symbol)
    symbols = ticker_mapping(canonical_symbol)
    source_symbol = vendor_symbol or symbols["yahoo"]
    retrieved_at = retrieved_at_utc or datetime.now(
        timezone.utc
    ).isoformat()

    history, normalization = _normalize_history(history)

    if history.empty:
        raise ValueError(
            f"未找到 {canonical_symbol} 的有效历史行情数据"
        )

    if len(history) < 60:
        raise ValueError(
            f"{canonical_symbol} 的有效交易日不足，"
            "无法进行技术分析（至少需要60个交易日）"
        )

    close = history["Close"].astype(float)
    calculation_volume = history["Volume"].fillna(0).astype(float)

    current_price = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    daily_change = _percentage_change_raw(close, 1)
    return_5 = _percentage_change_raw(close, 5)
    return_20 = _percentage_change_raw(close, 20)
    return_60 = _percentage_change_raw(close, 60)

    sma20 = float(close.tail(20).mean())
    sma50 = float(close.tail(50).mean())
    sma200 = (
        float(close.tail(200).mean())
        if len(close) >= 200
        else None
    )
    rsi14 = _calculate_rsi_raw(close, 14)

    latest_volume = float(calculation_volume.iloc[-1])
    average_volume_20 = float(
        calculation_volume.tail(20).mean()
    )
    average_volume_60 = float(
        calculation_volume.tail(60).mean()
    )
    volume_ratio_20 = (
        latest_volume / average_volume_20
        if average_volume_20 > 0
        else None
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

    rounded_volume_ratio = _round_value(volume_ratio_20)
    volume_signal = "数据不足"

    if rounded_volume_ratio is not None:
        if rounded_volume_ratio >= 1.5:
            volume_signal = "明显放量"
        elif rounded_volume_ratio >= 1.1:
            volume_signal = "温和放量"
        elif rounded_volume_ratio <= 0.7:
            volume_signal = "明显缩量"
        else:
            volume_signal = "成交量接近20日均量"

    trend_signal = "中性"

    if (
        sma200 is not None
        and current_price > sma20 > sma50 > sma200
    ):
        trend_signal = "多头排列"
    elif (
        sma200 is not None
        and current_price < sma20 < sma50 < sma200
    ):
        trend_signal = "空头排列"
    elif current_price > sma20 and current_price > sma50:
        trend_signal = "价格位于短中期均线上方"
    elif current_price < sma20 and current_price < sma50:
        trend_signal = "价格位于短中期均线下方"
    else:
        trend_signal = "短中期趋势分化"

    metrics = {
        "symbol": canonical_symbol,
        "source_symbol": source_symbol,
        "symbol_mapping": symbols,
        "data_start": _date_at(history, 0),
        "data_end": _date_at(history, -1),
        "trading_days": int(len(history)),
        "current_price_usd": _round_value(current_price),
        "previous_close_usd": _round_value(previous_close),
        "daily_change_percent": _round_value(daily_change),
        "return_5d_percent": _round_value(return_5),
        "return_20d_percent": _round_value(return_20),
        "return_60d_percent": _round_value(return_60),
        "sma20_usd": _round_value(sma20),
        "sma50_usd": _round_value(sma50),
        "sma200_usd": _round_value(sma200),
        "rsi14": _round_value(rsi14),
        "latest_volume": int(latest_volume),
        "average_volume_20d": int(average_volume_20),
        "average_volume_60d": int(average_volume_60),
        "volume_ratio_to_20d": rounded_volume_ratio,
        "volume_signal": volume_signal,
        "support_20d_usd": _round_value(support_20),
        "resistance_20d_usd": _round_value(resistance_20),
        "support_60d_usd": _round_value(support_60),
        "resistance_60d_usd": _round_value(resistance_60),
        "distance_from_support_20d_percent": _price_position(
            current_price, support_20
        ),
        "distance_from_resistance_20d_percent": _price_position(
            current_price, resistance_20
        ),
        "high_52_week_usd": _round_value(high_52_week),
        "low_52_week_usd": _round_value(low_52_week),
        "distance_from_52_week_high_percent": _price_position(
            current_price, high_52_week
        ),
        "distance_from_52_week_low_percent": _price_position(
            current_price, low_52_week
        ),
        "trend_signal": trend_signal,
        "support_resistance_method": (
            "对按日期升序排列、剔除Close为空后的同一份日线数据，"
            "分别取最近20和60个交易日Low最小值为支撑、"
            "High最大值为阻力；不保证价格在该位置反转"
        ),
        "price_source": PRICE_SOURCE,
        "calculation_note": (
            "全部指标由technical evidence文件中的同一份"
            "未复权日线行情机械计算，LLM不参与数值计算"
        ),
    }

    adjusted_close_available = "Adj Close" in history.columns
    raw_bars = []

    for date_value, row in history.iterrows():
        raw_bars.append(
            {
                "date": pd.Timestamp(date_value).date().isoformat(),
                "open": _json_number(row.get("Open")),
                "high": _json_number(row.get("High")),
                "low": _json_number(row.get("Low")),
                "close": _json_number(row.get("Close")),
                "adjusted_close": _json_number(
                    row.get("Adj Close")
                    if adjusted_close_available
                    else None
                ),
                "volume": _json_number(row.get("Volume")),
            }
        )

    final_index = len(history) - 1
    rounding_2 = (
        "Python round(float(value), 2); nearest with ties-to-even "
        "subject to IEEE-754 input representation"
    )
    volume_rounding = (
        "int(float(value)); truncate fractional shares toward zero"
    )
    indicator_evidence = {
        "current_price_usd": _evidence_entry(
            formula="C_t, where t is the last trading observation",
            raw_input=_series_reference(
                history, "close", final_index, final_index
            ),
            raw_result=current_price,
            rounded_result=metrics["current_price_usd"],
            rounding=rounding_2,
        ),
        "previous_close_usd": _evidence_entry(
            formula="C_(t-1)",
            raw_input=_series_reference(
                history, "close", final_index - 1, final_index - 1
            ),
            raw_result=previous_close,
            rounded_result=metrics["previous_close_usd"],
            rounding=rounding_2,
        ),
        "daily_change_percent": _evidence_entry(
            formula="100 * (C_t / C_(t-1) - 1)",
            raw_input=_series_reference(
                history, "close", final_index - 1, final_index
            ),
            raw_result=daily_change,
            rounded_result=metrics["daily_change_percent"],
            rounding=rounding_2,
        ),
        "return_5d_percent": _evidence_entry(
            formula="100 * (C_t / C_(t-5) - 1)",
            raw_input=_series_reference(
                history, "close", final_index - 5, final_index
            ),
            raw_result=return_5,
            rounded_result=metrics["return_5d_percent"],
            rounding=rounding_2,
        ),
        "return_20d_percent": _evidence_entry(
            formula="100 * (C_t / C_(t-20) - 1)",
            raw_input=_series_reference(
                history, "close", final_index - 20, final_index
            ),
            raw_result=return_20,
            rounded_result=metrics["return_20d_percent"],
            rounding=rounding_2,
        ),
        "return_60d_percent": _evidence_entry(
            formula="100 * (C_t / C_(t-60) - 1)",
            raw_input=_series_reference(
                history,
                "close",
                max(0, final_index - 60),
                final_index,
            ),
            raw_result=return_60,
            rounded_result=metrics["return_60d_percent"],
            rounding=rounding_2,
        ),
    }

    for window, raw_result in (
        (20, sma20),
        (50, sma50),
        (200, sma200),
    ):
        start_index = max(0, len(history) - window)
        indicator_evidence[f"sma{window}_usd"] = _evidence_entry(
            formula=(
                f"sum(C_(t-{window - 1}) ... C_t) / {window}; "
                "last N closing observations including C_t"
            ),
            raw_input=_series_reference(
                history, "close", start_index, final_index
            ),
            raw_result=raw_result,
            rounded_result=metrics[f"sma{window}_usd"],
            rounding=rounding_2,
        )

    indicator_evidence["rsi14"] = _evidence_entry(
        formula=(
            "delta_i=C_i-C_(i-1); gain_i=max(delta_i,0); "
            "loss_i=max(-delta_i,0); avg_gain and avg_loss use "
            "pandas ewm(alpha=1/14, adjust=False, min_periods=14, "
            "ignore_na=False); RS=avg_gain/avg_loss; "
            "RSI=100-100/(1+RS); avg_loss=0 yields 100"
        ),
        raw_input={
            **_series_reference(history, "close", 0, final_index),
            "smoothing_period": 14,
            "smoothing": (
                "recursive EWMA with alpha=1/14, initialized by "
                "the first non-null gain/loss; not SMA-seeded"
            ),
        },
        raw_result=rsi14,
        rounded_result=metrics["rsi14"],
        rounding=rounding_2,
    )

    for key, window, raw_result, rounded_result in (
        ("latest_volume", 1, latest_volume, metrics["latest_volume"]),
        (
            "average_volume_20d",
            20,
            average_volume_20,
            metrics["average_volume_20d"],
        ),
        (
            "average_volume_60d",
            60,
            average_volume_60,
            metrics["average_volume_60d"],
        ),
    ):
        indicator_evidence[key] = _evidence_entry(
            formula=(
                "V_t"
                if window == 1
                else f"sum(V last {window})/{window}"
            ),
            raw_input={
                **_series_reference(
                    history,
                    "volume",
                    final_index - window + 1,
                    final_index,
                ),
                "missing_volume_value": "replace with 0 for calculation",
            },
            raw_result=raw_result,
            rounded_result=rounded_result,
            rounding=volume_rounding,
        )

    indicator_evidence["volume_ratio_to_20d"] = _evidence_entry(
        formula="V_t / average_volume_20d_raw",
        raw_input={
            "numerator_indicator": "latest_volume",
            "denominator_indicator": "average_volume_20d",
        },
        raw_result=volume_ratio_20,
        rounded_result=metrics["volume_ratio_to_20d"],
        rounding=rounding_2,
    )

    for window in (20, 60):
        start_index = final_index - window + 1
        support_key = f"support_{window}d_usd"
        resistance_key = f"resistance_{window}d_usd"
        support_raw = support_20 if window == 20 else support_60
        resistance_raw = resistance_20 if window == 20 else resistance_60
        indicator_evidence[support_key] = _evidence_entry(
            formula=f"min(Low) over last {window} trading observations",
            raw_input=_series_reference(
                history, "low", start_index, final_index
            ),
            raw_result=support_raw,
            rounded_result=metrics[support_key],
            rounding=rounding_2,
        )
        indicator_evidence[resistance_key] = _evidence_entry(
            formula=f"max(High) over last {window} trading observations",
            raw_input=_series_reference(
                history, "high", start_index, final_index
            ),
            raw_result=resistance_raw,
            rounded_result=metrics[resistance_key],
            rounding=rounding_2,
        )

    high_low_start = max(0, len(history) - 252)
    indicator_evidence["high_52_week_usd"] = _evidence_entry(
        formula="max(High) over last min(252, available) observations",
        raw_input=_series_reference(
            history, "high", high_low_start, final_index
        ),
        raw_result=high_52_week,
        rounded_result=metrics["high_52_week_usd"],
        rounding=rounding_2,
    )
    indicator_evidence["low_52_week_usd"] = _evidence_entry(
        formula="min(Low) over last min(252, available) observations",
        raw_input=_series_reference(
            history, "low", high_low_start, final_index
        ),
        raw_result=low_52_week,
        rounded_result=metrics["low_52_week_usd"],
        rounding=rounding_2,
    )

    for key, level_key, raw_level in (
        (
            "distance_from_support_20d_percent",
            "support_20d_usd",
            support_20,
        ),
        (
            "distance_from_resistance_20d_percent",
            "resistance_20d_usd",
            resistance_20,
        ),
        (
            "distance_from_52_week_high_percent",
            "high_52_week_usd",
            high_52_week,
        ),
        (
            "distance_from_52_week_low_percent",
            "low_52_week_usd",
            low_52_week,
        ),
    ):
        indicator_evidence[key] = _evidence_entry(
            formula=(
                f"100 * (current_price_raw / {level_key}_raw - 1)"
            ),
            raw_input={
                "current_price_indicator": "current_price_usd",
                "level_indicator": level_key,
            },
            raw_result=_price_position_raw(current_price, raw_level),
            rounded_result=metrics[key],
            rounding=rounding_2,
        )

    reproducibility_missing = []
    if len(history) < MAX_TECHNICAL_WINDOW:
        reproducibility_missing.append(
            f"at_least_{MAX_TECHNICAL_WINDOW}_daily_bars"
        )

    reproducibility_metrics = (
        "current_price_usd",
        "return_20d_percent",
        "sma20_usd",
        "sma50_usd",
        "sma200_usd",
        "rsi14",
        "latest_volume",
        "support_20d_usd",
        "resistance_20d_usd",
        "support_60d_usd",
        "resistance_60d_usd",
        "high_52_week_usd",
        "low_52_week_usd",
    )
    reproducibility_missing.extend(
        key for key in reproducibility_metrics if metrics.get(key) is None
    )

    technical_evidence = {
        "schema_version": "1.0",
        "source": PRICE_SOURCE,
        "canonical_symbol": canonical_symbol,
        "vendor_symbol": source_symbol,
        "retrieved_at_utc": retrieved_at,
        "timezone": normalization["timezone"],
        "date_range": {
            "start": metrics["data_start"],
            "end": metrics["data_end"],
        },
        "last_trading_date": metrics["data_end"],
        "price_basis": {
            "column": "Close",
            "description": "vendor unadjusted daily close",
            "adjusted_close_saved_when_available": (
                adjusted_close_available
            ),
            "adjusted_close_used_for_metrics": False,
            "auto_adjust": AUTO_ADJUST,
            "actions": INCLUDE_ACTIONS,
        },
        "request_parameters": {
            "period": PRICE_PERIOD,
            "interval": PRICE_INTERVAL,
            "auto_adjust": AUTO_ADJUST,
            "actions": INCLUDE_ACTIONS,
        },
        "observation_count": len(history),
        "maximum_indicator_window_trading_days": (
            MAX_TECHNICAL_WINDOW
        ),
        "missing_data_handling": {
            **normalization,
            "sort_order": "date ascending",
            "duplicate_timestamp_rule": "keep last",
            "missing_close_rule": "drop row before every calculation",
            "missing_volume_rule": (
                "preserve null in raw_bars; use 0 in volume calculations"
            ),
            "missing_high_low_rule": (
                "pandas min/max skip null values; null if no valid value"
            ),
            "non_trading_days": (
                "not reindexed or imputed; windows count saved trading rows"
            ),
        },
        "rounding_rule": {
            "prices_percentages_rsi_ratios": rounding_2,
            "reported_volume_counts": volume_rounding,
            "raw_bars": "no application rounding after float conversion",
        },
        "calculation_method": {
            "return_20d": (
                "100 * (C_t / C_(t-20) - 1); 21 closes, 20 intervals"
            ),
            "moving_averages": (
                "arithmetic mean of the last 20/50/200 Close values, "
                "including the last trading observation"
            ),
            "rsi14": indicator_evidence["rsi14"]["formula"],
            "support_resistance": (
                "support=min(Low), resistance=max(High), separately "
                "over the last 20 and 60 trading observations"
            ),
        },
        "indicator_evidence": indicator_evidence,
        "reproducible": not reproducibility_missing,
        "reproducibility_missing": reproducibility_missing,
        "raw_bars": raw_bars,
    }

    return metrics, technical_evidence


def _evidence_directory() -> Path:
    configured = os.getenv("TECHNICAL_EVIDENCE_DIR")
    return (
        Path(configured).expanduser()
        if configured
        else DEFAULT_EVIDENCE_DIRECTORY
    )


def _save_technical_evidence(
    technical_evidence: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    directory = _evidence_directory()
    directory.mkdir(parents=True, exist_ok=True)
    retrieved = str(technical_evidence["retrieved_at_utc"])
    compact_time = "".join(
        character for character in retrieved if character.isdigit()
    )[:20]
    safe_symbol = str(technical_evidence["canonical_symbol"]).replace(
        ".", "_"
    )
    path = directory / (
        f"{safe_symbol}_{technical_evidence['last_trading_date']}_"
        f"{compact_time}_{uuid4().hex[:8]}.json"
    )
    absolute_path = path.resolve()

    file_evidence = {
        **technical_evidence,
        "evidence_path": str(absolute_path),
        "technical_metrics": metrics,
    }
    temporary_path = absolute_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            file_evidence,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(absolute_path)
    return absolute_path


def get_technical_metrics(symbol: str) -> dict[str, Any]:
    canonical_symbol = canonicalize_symbol(symbol)
    symbols = ticker_mapping(canonical_symbol)
    source_symbol = symbols["yahoo"]
    retrieved_at_utc = datetime.now(timezone.utc).isoformat()

    ticker = yf.Ticker(source_symbol)
    history = ticker.history(
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,
        auto_adjust=AUTO_ADJUST,
        actions=INCLUDE_ACTIONS,
    )

    if history.empty:
        raise ValueError(
            f"未找到 {canonical_symbol} 的历史行情数据"
        )

    metrics, technical_evidence = (
        calculate_technical_metrics_from_history(
            history,
            symbol=canonical_symbol,
            vendor_symbol=source_symbol,
            retrieved_at_utc=retrieved_at_utc,
        )
    )
    evidence_path = _save_technical_evidence(
        technical_evidence,
        metrics,
    )
    metrics["technical_evidence"] = {
        key: value
        for key, value in technical_evidence.items()
        if key != "raw_bars"
    }
    metrics["technical_evidence"]["evidence_path"] = str(
        evidence_path
    )

    return metrics
