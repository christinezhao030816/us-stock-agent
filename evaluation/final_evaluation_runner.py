from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from tools.financial_amount import normalize_financial_amount_display
from tools.news_evidence import classify_article_evidence, classify_filing_evidence

RUBRIC_PATH = PROJECT_ROOT / "evaluation" / "final_evaluation_rubric_v3.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "evaluation" / "final_evaluation_regression"
PREFLIGHT_TICKERS = ("AAPL", "BRK.B", "DUK", "TSM")

FACT_FIELDS = [
    "ticker",
    "claim_id",
    "category",
    "claim_type",
    "metric_name",
    "agent_claim",
    "ground_truth",
    "accuracy_label",
    "support_label",
    "numerical_label",
    "citation_label",
    "evidence_source",
    "evidence_url",
    "notes",
    "unique_claim_key",
    "occurrence_id",
    "evidence_depth",
    "full_text_available",
    "claim_scope_limited",
    "limitation_disclosed",
]

CLAIM_INVENTORY_FIELDS = [
    "item_id",
    "ticker",
    "section",
    "sentence_text",
    "atomic_claim",
    "claim_type",
    "item_category",
    "unique_claim_key",
    "contains_number",
    "contains_date",
    "contains_event",
    "contains_company_action",
    "registry_claim_id",
    "internal_provenance_id",
    "internal_traceability_status",
    "premise_claim_ids",
    "premise_citation_status",
    "is_required_external_citation",
    "is_not_claimed_marker",
    "exclusion_reason",
    "evidence_depth",
    "full_text_available",
    "claim_scope_limited",
    "limitation_disclosed",
    "citation_in_body",
    "citation_valid",
    "ground_truth_available",
    "evaluation_status",
    "extraction_method",
    "notes",
]

REPORT_FIELDS = [
    "run_id",
    "ticker",
    "technical",
    "fundamental",
    "news_events",
    "risks",
    "catalysts",
    "bull_case",
    "bear_case",
    "final_synthesis",
    "completed_modules",
    "completeness_score",
    "confidence_score",
    "technical_agent_status",
    "fundamental_agent_status",
    "news_agent_status",
    "strategy_agent_status",
    "freshness",
    "factual_correct",
    "factual_incorrect",
    "factual_unverifiable",
    "factual_not_claimed",
    "factual_accuracy_denominator",
    "factual_accuracy_pct",
    "unsupported_claims",
    "total_factual_claims",
    "unsupported_claim_rate_pct",
    "numerically_correct",
    "numerically_incorrect",
    "unsupported_numerical_claims",
    "unverifiable_numerical_claims",
    "numerical_accuracy_denominator",
    "numerical_accuracy_pct",
    "citation_correct",
    "citation_incorrect",
    "citation_unverifiable",
    "citation_not_cited",
    "citation_accuracy_denominator",
    "citation_accuracy_pct",
    "citation_required_claims",
    "validly_cited_claims",
    "citation_coverage_pct",
    "citation_integrity",
    "duplicate_claim_ids",
    "unknown_claim_ids",
    "unregistered_urls",
    "unused_source_ids",
    "total_body_factual_claims",
    "registry_matched_claims",
    "unmatched_body_claims",
    "claim_inventory_coverage_pct",
    "cited_body_claims",
    "uncited_body_claims",
    "invalid_citation_claims",
    "registry_unused_claims",
    "unique_external_claims",
    "external_claim_occurrences",
    "system_derived_claims",
    "operational_disclosures",
    "analytical_judgments",
    "not_claimed_markers",
    "legitimately_excluded_non_claims",
    "total_extracted_items",
    "reconciliation_closed",
    "internal_system_claims",
    "traceable_internal_claims",
    "internal_traceability_coverage_pct",
    "opinion_premises_required",
    "opinion_premises_cited",
    "opinion_premise_citation_coverage_pct",
    "report_path",
    "technical_evidence_path",
    "source_registry_path",
]

_CLAIM_MARKER = re.compile(r"\[([TFN]\d+)\]")
_ANY_CLAIM_MARKER = re.compile(r"\[([A-Z]\d+)\]")
_URL = re.compile(r"https?://[^\s<>\])]+")
_SOURCE_HEADING = re.compile(
    r"^#{1,4}\s+(?:\d+\.\s*)?(?:来源与证据|Sources and Evidence)\s*$",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ANCHOR = re.compile(
    r"facts\.([^.\s;]+)\.([^.\s;]+)\.units\.([^\s;]+)\s+"
    r"period=([^\s;.]+)\.\.([^\s;]+)"
    r"(?:\s+accession=([0-9-]+))?"
)
_COMPARE_ANCHOR = re.compile(
    r"compare\s+facts\.([^.\s;]+)\.([^.\s;]+)\.units\.([^\s;]+)"
    r"\s+period_end=([^\s;]+)"
)
_NUMBER = re.compile(r"(?<![A-Za-z])\(?-?\d[\d,]*(?:\.\d+)?(?:%|倍)?\)?")
_DATE = re.compile(r"\b(?:19|20)\d{2}(?:[-/.年]\d{1,2})?(?:[-/.月]\d{1,2}日?)?\b")
_DISPLAY_AMOUNT = re.compile(
    r"(?P<open>\()?\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<currency>美元|新台币|日元|USD|TWD|JPY)"
    r"(?P<per_share>\s*/\s*(?:股|share|shares))?\s*(?P<close>\))?",
    re.IGNORECASE,
)
_FORBIDDEN_AMOUNT_DISPLAY = re.compile(
    r"十亿美元|百万美元|\bbillion\b|\bmillion\b",
    re.IGNORECASE,
)

_SYSTEM_DERIVED_TERMS = (
    "置信度", "最终分数", "原始分数", "数据质量上限", "得分", "扣分",
    "freshness", "新鲜度", "period alignment", "期间对齐", "historical_fallback",
    "历史回退", "agent状态", "agent status", "系统评分", "系统诊断",
    "current_analysis_usable", "period_lag", "score cap",
    "综合分析日期", "技术面行情截止", "新闻信息截止", "证据文件路径",
    "未复权日收盘价", "保留5篇", "保留的新闻", "新闻来源质量",
    "价格基础", "行情数据", "系统选中期间",
    "历史区间机械计算", "历史价格区间参考", "时间窗口差异",
    "模块最终保留", "最终保留", "时间范围", "分析时间",
    "可能数据延迟", "15分钟", "新闻相关性评分", "新闻来源质量",
    "技术数据：", "新闻数据：", "财报提交日期距分析日", "选中财务期间距分析日",
    "定期披露时差",
)
_OPERATIONAL_TERMS = (
    "数据不可用", "数据缺失", "未取得", "未返回", "读取错误", "访问受限",
    "正文截断", "结构化事实缺失", "不能使用", "无法使用", "模块不可用",
    "missing", "unavailable", "truncated", "read error", "failed", "partial",
    "缺失项", "缺少摘要", "没有独立高质量", "没有被标记", "分析不完整",
    "正文部分内容缺失", "只能部分支持", "来源访问",
    "正文不可用", "待核实信息", "未读取完整正文", "附件内容不完整",
    "没有保留文章", "没有筛选出新闻", "新闻验证不足", "独立新闻验证",
    "没有独立高质量新闻", "缺少独立高质量新闻", "备用正文", "并非完整10-q",
    "无法进行新闻媒体", "无法通过独立媒体", "不存在可用于判断",
    "没有被工具评为较高质量",
)
_NOT_CLAIMED_TERMS = (
    "未提出", "未披露", "没有提出", "没有可用", "无可用", "not claimed",
    "no claim", "未找到可用", "无完整支持项",
)
_JUDGMENT_TERMS = (
    "看多", "看空", "风险", "催化剂", "综合判断", "偏积极", "偏谨慎", "中性",
    "可能", "或将", "倾向", "建议关注", "值得关注", "前景", "我认为", "判断为",
    "结论", "推断", "不能假定", "不确定", "偏强", "偏弱", "稳健", "需要观察",
    "依赖", "不宜", "有利", "不利", "制约因素", "限制因素",
    "正面因素", "存在矛盾", "关注后续", "跟踪公司",
    "下一期财报", "后续是否", "投资者应结合", "进一步核实",
    "公司经营信息", "历史盈利和现金流能力", "尚未确认趋势反转",
    "bull", "bear", "risk", "catalyst", "outlook", "could", "may",
)

_METRIC_ALIASES = {
    "current_price": ("当前价", "现价", "收盘价", "current price"),
    "previous_close": ("前收盘", "previous close"),
    "daily_change": ("单日", "日涨跌", "daily change"),
    "return_5d": ("5日", "5-day"),
    "return_20d": ("20日回报", "20日收益", "20-day return"),
    "return_60d": ("60日回报", "60日收益", "60-day return"),
    "sma20": ("sma20", "20日均线", "20-day moving average"),
    "sma50": ("sma50", "50日均线", "50-day moving average"),
    "sma200": ("sma200", "200日均线", "200-day moving average"),
    "rsi14": ("rsi", "相对强弱"),
    "latest_volume": ("最新成交量", "当日成交量"),
    "average_volume_20d": ("20日平均成交量",),
    "average_volume_60d": ("60日平均成交量",),
    "support_20d": ("20日支撑",),
    "resistance_20d": ("20日阻力",),
    "support_60d": ("60日支撑",),
    "resistance_60d": ("60日阻力",),
    "high_52_week": ("52周高",),
    "low_52_week": ("52周低",),
    "revenue": ("收入", "营收", "revenue"),
    "net_income": ("净利润", "profit", "net income"),
    "EPS": ("eps", "每股收益"),
    "operating_cash_flow": ("经营现金流", "operating cash flow"),
    "capital_expenditure": ("资本支出", "capital expenditure"),
    "free_cash_flow": ("自由现金流", "free cash flow"),
    "total_assets": ("总资产", "资产总额", "total assets"),
    "total_liabilities": ("总负债", "total liabilities"),
    "total_debt": ("总债务", "total debt"),
    "cash": ("现金", "cash"),
    "reporting_period": ("报告期", "财务期间", "期末"),
    "filing_date": ("提交日期", "申报日期", "filed"),
    "event": ("事件", "公告", "新闻", "filing"),
}


INDEPENDENT_CONCEPTS: dict[str, set[str]] = {
    "revenue": {
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "RegulatedAndUnregulatedOperatingRevenue",
        "OperatingRevenues",
        "Revenues",
        "SalesRevenueNet",
        "Revenue",
    },
    "net_income": {
        "NetIncomeLoss",
        "ProfitLossAttributableToOwnersOfParent",
        "ProfitLoss",
    },
    "EPS": {
        "EarningsPerShareDiluted",
        "EarningsPerShareBasic",
        "DilutedEarningsLossPerShare",
        "BasicEarningsLossPerShare",
    },
    "operating_cash_flow": {
        "NetCashProvidedByUsedInOperatingActivities",
        "CashFlowsFromUsedInOperatingActivities",
    },
    "capital_expenditure": {
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets",
        "PurchaseOfPropertyPlantAndEquipment",
    },
    "cash": {
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndCashEquivalents",
    },
    "total_assets": {"Assets"},
    "total_liabilities": {"Liabilities"},
    "total_debt": {
        "ShortTermBorrowings",
        "ShortTermDebtCurrent",
        "LongTermDebtCurrent",
        "LongTermDebtAndFinanceLeaseObligationsCurrent",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        "Borrowings",
        "CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings",
        "CurrentPortionOfLongtermBorrowings",
        "ShorttermBorrowings",
        "LongtermBorrowings",
        "NoncurrentBorrowings",
    },
    "equity": {
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "Equity",
    },
}

TECHNICAL_KEYS = {
    "current_price": ("current_price_usd", "USD", "price"),
    "previous_close": ("previous_close_usd", "USD", "price"),
    "daily_change": ("daily_change_percent", "percent", "percent"),
    "return_5d": ("return_5d_percent", "percent", "percent"),
    "return_20d": ("return_20d_percent", "percent", "percent"),
    "return_60d": ("return_60d_percent", "percent", "percent"),
    "sma20": ("sma20_usd", "USD", "price"),
    "sma50": ("sma50_usd", "USD", "price"),
    "sma200": ("sma200_usd", "USD", "price"),
    "rsi14": ("rsi14", "index", "rsi"),
    "latest_volume": ("latest_volume", "shares", "volume"),
    "average_volume_20d": ("average_volume_20d", "shares", "volume"),
    "average_volume_60d": ("average_volume_60d", "shares", "volume"),
    "volume_ratio_to_20d": ("volume_ratio_to_20d", "ratio", "ratio"),
    "support_20d": ("support_20d_usd", "USD", "price"),
    "resistance_20d": ("resistance_20d_usd", "USD", "price"),
    "support_60d": ("support_60d_usd", "USD", "price"),
    "resistance_60d": ("resistance_60d_usd", "USD", "price"),
    "distance_from_support_20d": (
        "distance_from_support_20d_percent",
        "percent",
        "percent",
    ),
    "distance_from_resistance_20d": (
        "distance_from_resistance_20d_percent",
        "percent",
        "percent",
    ),
    "high_52_week": ("high_52_week_usd", "USD", "price"),
    "low_52_week": ("low_52_week_usd", "USD", "price"),
    "distance_from_52_week_high": (
        "distance_from_52_week_high_percent",
        "percent",
        "percent",
    ),
    "distance_from_52_week_low": (
        "distance_from_52_week_low_percent",
        "percent",
        "percent",
    ),
}


def load_rubric(path: Path = RUBRIC_PATH) -> dict[str, Any]:
    rubric = json.loads(path.read_text(encoding="utf-8"))
    if rubric.get("rubric_version") != "3.0":
        raise ValueError("Final evaluation rubric must be version 3.0")
    expected = {
        "factual_accuracy",
        "unsupported_claim_rate",
        "numerical_accuracy",
        "citation_accuracy",
        "citation_coverage",
        "completeness",
    }
    missing = expected - set(rubric.get("metrics") or {})
    if missing:
        raise ValueError(f"Rubric is missing metrics: {sorted(missing)}")
    return rubric


def _safe_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("/", "_")


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _pct(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 2) if denominator else None


def split_report_body(report: str) -> str:
    lines = []
    for line in str(report or "").splitlines():
        if _SOURCE_HEADING.match(line.strip()):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def parse_display_amount(text: str) -> dict[str, Any] | None:
    """解析完整会计数字金额；支持中文/ISO币种、括号负数和每股单位。"""

    match = _DISPLAY_AMOUNT.search(str(text or ""))
    if not match:
        return None
    value = float(match.group("value").replace(",", ""))
    if match.group("open") and match.group("close"):
        value = -abs(value)
    currency = {
        "美元": "USD",
        "新台币": "TWD",
        "日元": "JPY",
    }.get(match.group("currency"), match.group("currency").upper())
    return {
        "value": value,
        "currency": currency,
        "unit": f"{currency}/share" if match.group("per_share") else currency,
    }


def forbidden_financial_displays(report: str) -> list[str]:
    return [match.group(0) for match in _FORBIDDEN_AMOUNT_DISPLAY.finditer(str(report or ""))]


def _sentence_units(report: str) -> list[tuple[str, str]]:
    section = "preamble"
    units: list[tuple[str, str]] = []
    for raw_line in split_report_body(report).splitlines():
        line = raw_line.strip()
        heading = _HEADING.match(line)
        if heading:
            section = re.sub(r"[*_`]", "", heading.group(2)).strip()
            continue
        if not line or re.fullmatch(r"[-:|\s]+", line):
            continue
        cleaned = re.sub(r"^(?:[-*+] |\d+[.)]\s+|>\s*)", "", line).strip()
        markers = _ANY_CLAIM_MARKER.findall(cleaned)
        marker_text = "".join(f"[{marker}]" for marker in markers)
        narrative = _ANY_CLAIM_MARKER.sub("", cleaned).strip()
        parts = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s*", narrative)
            if part.strip()
        ]
        for part in parts or [narrative]:
            candidate = f"{part}{marker_text}" if marker_text else part
            if candidate and not re.fullmatch(r"[-:|\s]+", candidate):
                units.append((section, candidate))
    return units


def _fact_signals(sentence: str) -> dict[str, bool]:
    lowered = sentence.lower()
    contains_number = bool(_NUMBER.search(sentence))
    contains_date = bool(_DATE.search(sentence))
    contains_event = any(
        token in lowered
        for token in (
            "新闻", "事件", "公告", "申报", "filing", "10-q", "10-k", "20-f",
            "6-k", "8-k", "监管", "发布", "披露",
        )
    )
    contains_company_action = any(
        token in lowered
        for token in (
            "回购", "分红", "派息", "收购", "合并", "任命", "辞任", "增发",
            "buyback", "dividend", "acquisition", "appoint", "resign",
        )
    )
    domain_fact = any(
        token in lowered
        for token in (
            "价格", "收盘", "均线", "rsi", "成交量", "支撑", "阻力", "回报",
            "收益率", "趋势", "多头", "空头", "上涨", "下跌", "高于", "低于",
            "收入", "营收", "利润", "现金流", "资产", "负债", "债务", "eps",
            "每股收益", "同比", "环比", "最新", "最近", "截至", "期间", "提交日期",
            "price", "moving average", "volume", "support", "resistance", "revenue",
            "income", "cash flow", "assets", "liabilities", "increased", "decreased",
        )
    )
    return {
        "contains_number": contains_number,
        "contains_date": contains_date,
        "contains_event": contains_event,
        "contains_company_action": contains_company_action,
        "is_factual": contains_number or contains_date or contains_event or contains_company_action or domain_fact,
    }


def classify_extracted_item(
    sentence: str,
    section: str = "",
) -> str:
    """Classify one body item without using the source registry as the universe."""

    lowered = f"{section} {sentence}".lower()
    signals = _fact_signals(sentence)
    cleaned = re.sub(r"[*_`]", "", _ANY_CLAIM_MARKER.sub("", sentence)).strip()
    if _CLAIM_MARKER.search(sentence) and (
        "标题级信息" in sentence or "附件内容不完整" in sentence
    ):
        return "external_fact"
    if (
        (cleaned.endswith(("：", ":")) and len(cleaned) <= 40)
        or cleaned.startswith(("/Users/", "/private/", "C:\\"))
        or "不构成投资建议" in cleaned
        or "独立决策" in cleaned
        or "投资者应结合" in cleaned
        or "进一步核实" in cleaned
    ):
        return "legitimately_excluded_non_claim"
    if any(term in lowered for term in _NOT_CLAIMED_TERMS):
        return "not_claimed_marker"
    if any(term in lowered for term in _OPERATIONAL_TERMS):
        return "operational_disclosure"
    if any(term in lowered for term in _SYSTEM_DERIVED_TERMS):
        return "system_derived"
    if any(term in lowered for term in ("下一期", "后续是否", "后续需要关注")):
        return "analytical_judgment"
    if (
        not signals["contains_number"]
        and not signals["contains_date"]
        and cleaned.rstrip("。；;.! ").endswith(("偏强", "偏弱", "偏正面", "偏负面"))
    ):
        return "analytical_judgment"
    # A concrete number, date, event or company action can never be relabelled
    # as opinion merely because it appears in a strategy section.
    if (
        signals["contains_number"]
        or signals["contains_date"]
        or signals["contains_event"]
        or signals["contains_company_action"]
    ):
        return "external_fact"
    if any(term in lowered for term in _JUDGMENT_TERMS):
        return "analytical_judgment"
    if signals["is_factual"]:
        return "external_fact"
    return "legitimately_excluded_non_claim"


def _normalized_claim_key(sentence: str) -> str:
    cleaned = _ANY_CLAIM_MARKER.sub("", sentence).lower()
    cleaned = re.sub(r"[^0-9a-z\u4e00-\u9fff.%]+", " ", cleaned)
    return " ".join(cleaned.split())


def _match_internal_provenance(
    sentence: str,
    category: str,
    records: list[dict[str, Any]],
) -> str:
    lowered = sentence.lower()
    candidates = [
        item
        for item in records
        if str(item.get("claim_category")) == category
        or (category == "not_claimed_marker" and item.get("claim_category") == "operational_disclosure")
    ]
    if category == "system_derived":
        if "置信" in lowered or "分数" in lowered or "得分" in lowered:
            candidates = [item for item in candidates if item.get("metric_name") == "confidence_score"]
        elif any(term in lowered for term in ("freshness", "新鲜度", "期间", "回退", "alignment")):
            candidates = [
                item for item in candidates
                if item.get("metric_name") == "fundamental_freshness_and_alignment"
            ]
    elif category in {"operational_disclosure", "not_claimed_marker"}:
        for role in ("technical", "fundamental", "news", "strategy"):
            if role in lowered or {
                "technical": "技术",
                "fundamental": "基本面",
                "news": "消息",
                "strategy": "策略",
            }[role] in lowered:
                narrowed = [item for item in candidates if item.get("metric_name") == f"{role}_agent_status"]
                if narrowed:
                    candidates = narrowed
                break
    return str(candidates[0].get("provenance_id") or "") if candidates else ""


def _claim_value_tokens(claim: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, bool) or value is None:
            return
        elif isinstance(value, (int, float)):
            tokens.add(str(value))
            tokens.add(f"{value:,}")
            unit = str(claim.get("unit") or "").lower()
            factor = 1_000_000_000 if "billion" in unit else 1_000_000 if "million" in unit else 1
            full = float(value) * factor
            if float(full).is_integer():
                tokens.add(f"{int(full):,}")
                if full < 0:
                    tokens.add(f"({abs(int(full)):,}")
        else:
            item = str(value).strip()
            if len(item) >= 4:
                tokens.add(item)

    visit(claim.get("value"))
    if claim.get("period_or_as_of_date"):
        tokens.add(str(claim["period_or_as_of_date"]))
    if claim.get("source_date"):
        tokens.add(str(claim["source_date"]))
    value = claim.get("value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if claim.get("metric_name") in {"net_margin", "equity_ratio"}:
            tokens.add(str(round(float(value) * 100, 2)))
    return tokens


def _token_present(sentence: str, token: str) -> bool:
    if re.fullmatch(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", token):
        return bool(
            re.search(
                rf"(?<![0-9]){re.escape(token)}(?![0-9])",
                sentence,
                re.IGNORECASE,
            )
        )
    return token.lower() in sentence.lower()


def _semantic_claim_matches(
    sentence: str, citations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lowered = sentence.lower()
    scored: list[tuple[int, dict[str, Any]]] = []
    for claim in citations:
        metric = str(claim.get("metric_name") or "")
        aliases = _METRIC_ALIASES.get(metric, ())
        alias_match = any(alias.lower() in lowered for alias in aliases)
        value_match = any(_token_present(sentence, token) for token in _claim_value_tokens(claim))
        raw_value = claim.get("value")
        claim_has_numeric_value = (
            isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
        )
        if _NUMBER.search(sentence) and claim_has_numeric_value and not value_match:
            continue
        claim_text = str(claim.get("claim_text") or "").lower()
        distinctive = [
            token
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[a-z]{4,}", claim_text)
            if token not in {"数据", "来源", "截至", "公司", "报告", "日期"}
        ]
        text_overlap = sum(token in lowered for token in distinctive[:8])
        score = (4 if value_match else 0) + (3 if alias_match else 0) + min(text_overlap, 3)
        if score >= 5 or (value_match and text_overlap >= 1):
            scored.append((score, claim))
    if not scored:
        return []
    highest = max(score for score, _ in scored)
    return [claim for score, claim in scored if score >= max(5, highest - 1)]


def _asserts_metric_numeric_value(
    sentence: str, metric: str, aliases: tuple[str, ...]
) -> bool:
    cleaned = _ANY_CLAIM_MARKER.sub("", re.sub(r"[*_`]", "", sentence))
    cleaned = _DATE.sub("", cleaned)
    lowered = cleaned.lower()
    if metric in {"revenue", "net_income"} and "同比" in lowered:
        return False
    for alias in aliases:
        if re.fullmatch(r"[a-z0-9 ]+", alias.lower()):
            matched_alias = re.search(
                rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])",
                lowered,
            )
            position = matched_alias.start() if matched_alias else -1
        else:
            position = lowered.find(alias.lower())
        if position < 0:
            continue
        if metric.startswith(("support_", "resistance_")) and "距离" in cleaned[max(0, position - 5) : position]:
            continue
        tail = cleaned[position + len(alias) : position + len(alias) + 18]
        if metric == "latest_volume" and ("均量" in tail or "倍" in tail):
            continue
        tail = re.sub(r"^(?:\s|为|是|达|约|=|:|：|分别)+", "", tail)
        if re.match(r"(?:19|20)\d{2}(?:财年|年)", tail):
            # A period-scope statement (for example, "自由现金流是2024财年
            # 累计数据") does not assert a different monetary value.
            continue
        if re.match(r"\(?-?\d[\d,]*(?:\.\d+)?", tail):
            return True
    return False


def _news_scope_consistent(sentence: str, claim: dict[str, Any]) -> bool:
    depth = str(claim.get("evidence_depth") or "fully_supported")
    lowered = re.sub(r"[*_`]", "", sentence).lower()
    if depth == "headline_only":
        title = normalize_financial_amount_display(
            str((claim.get("source_content") or {}).get("title") or "")
        ).strip().lower()
        return (
            "标题级信息" in sentence
            and "未读取完整正文" in sentence
            and bool(title)
            and title in lowered
        )
    if depth == "truncated_source":
        return "附件内容不完整" in sentence and "已读取片段" in sentence
    if depth == "unavailable":
        return (
            ("待核实" in sentence or "正文不可用" in sentence)
            and not bool(_CLAIM_MARKER.search(sentence))
        )
    return True


def build_claim_inventory(
    ticker: str,
    report: str,
    citations: list[dict[str, Any]],
    internal_provenance: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract and classify every atomic body item, independently of registry."""

    known = {str(claim.get("claim_id") or ""): claim for claim in citations}
    filing_ids = [
        str(claim.get("claim_id") or "")
        for claim in citations
        if claim.get("agent") == "news"
        and claim.get("claim_type") == "sec_filing_event"
    ]
    internal_records = [item for item in (internal_provenance or []) if isinstance(item, dict)]
    inventory: list[dict[str, Any]] = []
    for occurrence_index, (section, sentence) in enumerate(_sentence_units(report), start=1):
        signals = _fact_signals(sentence)
        markers = _ANY_CLAIM_MARKER.findall(sentence)
        category = classify_extracted_item(sentence, section)
        valid_markers = [marker for marker in markers if marker in known]
        invalid_markers = [marker for marker in markers if marker not in known]
        semantic = _semantic_claim_matches(sentence, citations)
        semantic_ids = [str(claim.get("claim_id") or "") for claim in semantic]
        if valid_markers and semantic_ids:
            narrowed = [marker for marker in valid_markers if marker in semantic_ids]
            if narrowed:
                valid_markers = narrowed
        records: list[tuple[str, bool, str, str]] = []
        if category == "external_fact":
            for marker in valid_markers:
                records.append((marker, True, "body_direct_citation", ""))
            for index, marker in enumerate(invalid_markers):
                if marker.startswith("S") and marker[1:].isdigit():
                    ordinal = int(marker[1:])
                    matched = filing_ids[ordinal - 1] if 1 <= ordinal <= len(filing_ids) else ""
                else:
                    matched = semantic_ids[index] if index < len(semantic_ids) else ""
                records.append((matched, False, "body_invalid_citation", f"unknown_marker={marker}"))
            if not records:
                if semantic_ids:
                    records.extend(
                        (claim_id, False, "body_semantic_registry_match", "citation_missing")
                        for claim_id in semantic_ids
                    )
                else:
                    records.append(("", False, "body_unmatched_heuristic", "no_registry_match"))
        else:
            records.append(("", not invalid_markers, "body_item_classification", ""))
        seen: set[tuple[str, str]] = set()
        for claim_id, citation_valid, method, note in records:
            key = (claim_id, method)
            if key in seen:
                continue
            seen.add(key)
            claim = known.get(claim_id, {})
            provenance_id = _match_internal_provenance(
                sentence, category, internal_records
            )
            premise_ids = valid_markers if category == "analytical_judgment" else []
            unique_key = claim_id or _normalized_claim_key(sentence)
            inventory.append(
                {
                    "item_id": f"{ticker}-{occurrence_index:04d}-{len(seen)}",
                    "ticker": ticker,
                    "section": section,
                    "sentence_text": sentence,
                    "atomic_claim": sentence,
                    "claim_type": str(claim.get("claim_type") or "body_fact"),
                    "item_category": category,
                    "unique_claim_key": unique_key,
                    "contains_number": signals["contains_number"],
                    "contains_date": signals["contains_date"],
                    "contains_event": signals["contains_event"],
                    "contains_company_action": signals["contains_company_action"],
                    "registry_claim_id": claim_id,
                    "internal_provenance_id": provenance_id,
                    "internal_traceability_status": (
                        "traceable" if provenance_id else "missing"
                        if category in {"system_derived", "operational_disclosure", "not_claimed_marker"}
                        else "not_applicable"
                    ),
                    "premise_claim_ids": _json_cell(premise_ids),
                    "premise_citation_status": (
                        "supported" if premise_ids else "not_required"
                    ),
                    "is_required_external_citation": category == "external_fact",
                    "is_not_claimed_marker": category == "not_claimed_marker",
                    "exclusion_reason": (
                        "non_factual_narrative" if category == "legitimately_excluded_non_claim" else ""
                    ),
                    "evidence_depth": claim.get("evidence_depth") or "",
                    "full_text_available": claim.get("full_text_available") if claim else "",
                    "claim_scope_limited": claim.get("claim_scope_limited") if claim else "",
                    "limitation_disclosed": claim.get("limitation_disclosed") if claim else "",
                    "citation_in_body": _json_cell(markers),
                    "citation_valid": citation_valid,
                    "ground_truth_available": False,
                    "evaluation_status": "pending",
                    "extraction_method": method,
                    "notes": note,
                }
            )
    return inventory


def evaluate_claim_inventory(
    inventory: list[dict[str, Any]],
    registry_rows: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把独立ground truth核验结果映射到正文claim occurrence。"""

    by_id = {
        row["claim_id"]: row
        for row in registry_rows
        if row.get("claim_id")
    }
    claims_by_id = {
        str(claim.get("claim_id") or ""): claim for claim in citations
    }
    fact_rows: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for item in inventory:
        if item.get("item_category") != "external_fact":
            current = dict(item)
            if item.get("item_category") in {
                "system_derived",
                "operational_disclosure",
                "not_claimed_marker",
            }:
                current["evaluation_status"] = item.get("internal_traceability_status")
            elif item.get("item_category") == "analytical_judgment":
                current["evaluation_status"] = "opinion_not_factual"
            else:
                current["evaluation_status"] = "excluded_non_claim"
            enriched.append(current)
            continue
        claim_id = str(item.get("registry_claim_id") or "")
        source = by_id.get(claim_id)
        structured_claim = claims_by_id.get(claim_id)
        current = dict(item)
        if source:
            sentence = str(item.get("sentence_text") or "")
            expected_tokens = _claim_value_tokens(structured_claim or {})
            raw_value = (structured_claim or {}).get("value")
            numeric_claim = isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool)
            aliases = _METRIC_ALIASES.get(
                str((structured_claim or {}).get("metric_name") or ""), ()
            )
            body_consistent = True
            if numeric_claim and _asserts_metric_numeric_value(
                sentence,
                str((structured_claim or {}).get("metric_name") or ""),
                aliases,
            ):
                body_consistent = any(
                    _token_present(sentence, token) for token in expected_tokens
                )
            if (structured_claim or {}).get("agent") == "news":
                body_consistent = body_consistent and _news_scope_consistent(
                    sentence, structured_claim or {}
                )
            current["ground_truth_available"] = source["accuracy_label"] in {"correct", "incorrect"}
            current["evaluation_status"] = (
                source["accuracy_label"] if body_consistent else "incorrect"
            )
            citation_label = (
                source["citation_label"]
                if item["citation_valid"]
                else "incorrect"
                if "invalid" in item["extraction_method"]
                else "not_cited"
            )
            if not body_consistent:
                citation_label = "incorrect" if item["citation_valid"] else citation_label
            fact_rows.append(
                {
                    **source,
                    "unique_claim_key": item.get("unique_claim_key"),
                    "occurrence_id": item.get("item_id"),
                    "evidence_depth": (structured_claim or {}).get("evidence_depth") or "",
                    "full_text_available": (structured_claim or {}).get("full_text_available"),
                    "claim_scope_limited": (structured_claim or {}).get("claim_scope_limited"),
                    "limitation_disclosed": (
                        ("标题级信息" in sentence and "未读取完整正文" in sentence)
                        or ("附件内容不完整" in sentence and "已读取片段" in sentence)
                        or not (structured_claim or {}).get("claim_scope_limited")
                    ),
                    "agent_claim": item["atomic_claim"],
                    "accuracy_label": (
                        source["accuracy_label"] if body_consistent else "incorrect"
                    ),
                    "support_label": (
                        source["support_label"] if body_consistent else "unsupported"
                    ),
                    "numerical_label": (
                        source["numerical_label"] if body_consistent else "incorrect"
                    ),
                    "citation_label": citation_label,
                    "notes": "; ".join(
                        value
                        for value in (
                            source.get("notes", ""),
                            item.get("notes", ""),
                            "body_value_mismatch" if not body_consistent else "",
                        )
                        if value
                    ),
                }
            )
        else:
            current["evaluation_status"] = "unverifiable"
            fact_rows.append(
                {
                    "ticker": item["ticker"],
                    "claim_id": claim_id or "UNMATCHED",
                    "category": "body",
                    "claim_type": item["claim_type"],
                    "metric_name": "unmatched_body_claim",
                    "agent_claim": item["atomic_claim"],
                    "ground_truth": "null",
                    "accuracy_label": "unverifiable",
                    "support_label": "unsupported",
                    "numerical_label": "unsupported" if item["contains_number"] else "not_applicable",
                    "citation_label": "incorrect" if "invalid" in item["extraction_method"] else "not_cited",
                    "evidence_source": "claim_inventory_unmatched",
                    "evidence_url": "",
                    "notes": item.get("notes", ""),
                    "unique_claim_key": item.get("unique_claim_key"),
                    "occurrence_id": item.get("item_id"),
                    "evidence_depth": item.get("evidence_depth", ""),
                    "full_text_available": item.get("full_text_available", ""),
                    "claim_scope_limited": item.get("claim_scope_limited", ""),
                    "limitation_disclosed": item.get("limitation_disclosed", ""),
                }
            )
        enriched.append(current)
    represented_not_claimed = {
        str(item.get("registry_claim_id") or "")
        for item in inventory
        if item.get("registry_claim_id")
        and by_id.get(str(item.get("registry_claim_id") or ""), {}).get("accuracy_label")
        == "not_claimed"
    }
    fact_rows.extend(
        row
        for row in registry_rows
        if row.get("accuracy_label") == "not_claimed"
        and row.get("claim_id") not in represented_not_claimed
    )
    return fact_rows, enriched


def claim_inventory_summary(
    inventory: list[dict[str, Any]], citations: list[dict[str, Any]]
) -> dict[str, Any]:
    external = [item for item in inventory if item.get("item_category") == "external_fact"]
    matched_ids = {
        str(item.get("registry_claim_id") or "")
        for item in external
        if item.get("registry_claim_id")
    }
    all_ids = {str(item.get("claim_id") or "") for item in citations if item.get("claim_id")}
    total = len(external)
    matched = sum(bool(item.get("registry_claim_id")) for item in external)
    cited = sum(bool(item.get("citation_valid")) for item in external)
    invalid = sum("invalid" in str(item.get("extraction_method")) for item in external)
    uncited = sum(
        not item.get("citation_valid") and "invalid" not in str(item.get("extraction_method"))
        for item in external
    )
    return {
        "total_body_factual_claims": total,
        "registry_matched_claims": matched,
        "unmatched_body_claims": total - matched,
        "claim_inventory_coverage_pct": _pct(matched, total),
        "cited_body_claims": cited,
        "uncited_body_claims": uncited,
        "invalid_citation_claims": invalid,
        "registry_unused_claims": sorted(all_ids - matched_ids),
    }


def evidence_classification_summary(inventory: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(str(item.get("item_category") or "unclassified") for item in inventory)
    external = [item for item in inventory if item.get("item_category") == "external_fact"]
    internal = [
        item
        for item in inventory
        if item.get("item_category") in {"system_derived", "operational_disclosure", "not_claimed_marker"}
    ]
    opinions = [item for item in inventory if item.get("item_category") == "analytical_judgment"]
    premise_required = [
        item
        for item in opinions
        if item.get("premise_claim_ids") not in {"", "[]"}
    ]
    premise_cited = [item for item in premise_required if item.get("premise_citation_status") == "supported"]
    traced = sum(item.get("internal_traceability_status") == "traceable" for item in internal)
    total = len(inventory)
    reconciled = sum(categories.values())
    return {
        "total_extracted_items": total,
        "external_fact": categories["external_fact"],
        "system_derived": categories["system_derived"],
        "analytical_judgment": categories["analytical_judgment"],
        "operational_disclosure": categories["operational_disclosure"],
        "not_claimed_marker": categories["not_claimed_marker"],
        "legitimately_excluded_non_claim": categories["legitimately_excluded_non_claim"],
        "unclassified": categories["unclassified"],
        "reconciliation_total": reconciled,
        "reconciliation_closed": total == reconciled and categories["unclassified"] == 0,
        "external_claim_occurrences": len(external),
        "unique_external_claims": len({str(item.get("unique_claim_key")) for item in external}),
        "internal_system_claims": len(internal),
        "traceable_internal_claims": traced,
        "internal_traceability_coverage_pct": _pct(traced, len(internal)) if internal else 100.0,
        "analytical_judgments": len(opinions),
        "opinion_premises_required": len(premise_required),
        "opinion_premises_cited": len(premise_cited),
        "opinion_premise_citation_coverage_pct": (
            _pct(len(premise_cited), len(premise_required)) if premise_required else 100.0
        ),
    }


def reconcile_extraction_counts(
    *,
    total_extracted_items: int,
    body_not_claimed_markers: int,
    external_factual_denominator: int,
    catalog_only_not_claimed: int = 0,
    total_not_claimed_rows: int | None = None,
) -> dict[str, Any]:
    denominator_closed = total_extracted_items - body_not_claimed_markers == external_factual_denominator
    catalog_closed = (
        total_not_claimed_rows is None
        or body_not_claimed_markers + catalog_only_not_claimed == total_not_claimed_rows
    )
    return {
        "total_extracted_items": total_extracted_items,
        "body_not_claimed_markers": body_not_claimed_markers,
        "external_factual_denominator": external_factual_denominator,
        "catalog_only_not_claimed": catalog_only_not_claimed,
        "total_not_claimed_rows": total_not_claimed_rows,
        "unexplained_difference": (
            total_extracted_items - body_not_claimed_markers - external_factual_denominator
        ),
        "closed": denominator_closed and catalog_closed,
    }


def _sections(report: str) -> list[tuple[str, str]]:
    body = split_report_body(report)
    lines = body.splitlines()
    headings = []
    for index, line in enumerate(lines):
        match = _HEADING.match(line.strip())
        if match:
            headings.append(
                (
                    index,
                    len(match.group(1)),
                    re.sub(r"[*_`]", "", match.group(2)).strip(),
                )
            )
    sections = []
    for heading_index, (line_index, level, title) in enumerate(headings):
        end = len(lines)
        for next_line, next_level, _ in headings[heading_index + 1 :]:
            if next_level <= level:
                end = next_line
                break
        sections.append((title, "\n".join(lines[line_index + 1 : end]).strip()))
    if headings and headings[0][0] > 0:
        sections.insert(0, ("preamble", "\n".join(lines[: headings[0][0]]).strip()))
    return sections


def _meaningful(text: str) -> bool:
    cleaned = re.sub(r"[\s*_`#>|\-]", "", text)
    return len(cleaned) >= 30


def _complete_missing_disclosure(text: str) -> bool:
    lowered = text.lower()
    has_missing = any(
        token in lowered
        for token in ("缺失", "未找到", "不可用", "数据不足", "unavailable", "missing")
    )
    has_impact = any(
        token in lowered
        for token in ("影响", "限制", "不能", "无法", "置信度", "impact", "limit")
    )
    has_degrade = any(
        token in lowered
        for token in ("partial", "failed", "降级", "不完整", "unknown")
    )
    return has_missing and has_impact and has_degrade and _meaningful(text)


def evaluate_completeness(report: str) -> dict[str, Any]:
    parsed = _sections(report)

    def selected(patterns: Iterable[str]) -> list[tuple[str, str]]:
        compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        return [
            (title, content)
            for title, content in parsed
            if any(pattern.search(title) for pattern in compiled)
        ]

    rules = {
        "technical": [r"技术面", r"technical"],
        "fundamental": [r"基本面", r"fundamental"],
        "news_events": [r"消息面", r"新闻", r"news", r"events?"],
        "risks": [r"风险", r"risks?", r"数据限制"],
        "catalysts": [r"催化剂", r"后续.*(?:指标|事件)", r"catalysts?", r"watch"],
        "bull_case": [r"看多因素", r"正面因素", r"bull"],
        "bear_case": [r"看空", r"bear", r"downside"],
        "final_synthesis": [r"综合判断", r"综合结论", r"final synthesis", r"conclusion"],
    }
    result: dict[str, Any] = {}
    for module, patterns in rules.items():
        matches = selected(patterns)
        content = "\n".join(item[1] for item in matches)
        completed = _meaningful(content) or _complete_missing_disclosure(content)
        if module == "bear_case" and completed:
            completed = bool(
                re.search(
                    r"看空|下行|偏谨慎|bear|downside|若|如果",
                    "\n".join(item[0] for item in matches) + "\n" + content,
                    re.IGNORECASE,
                )
            )
        result[module] = {
            "completed": completed,
            "matched_headings": [item[0] for item in matches],
            "reason": (
                "substantive_content"
                if completed and not _complete_missing_disclosure(content)
                else "complete_missing_data_disclosure"
                if completed
                else "missing_or_non_substantive"
            ),
        }
    completed_count = sum(item["completed"] for item in result.values())
    result["completed_modules"] = completed_count
    result["completeness_score"] = round(completed_count / 8 * 100, 2)
    return result


def extract_confidence_score(report: str) -> int | None:
    cleaned = str(report or "").replace("*", "")
    for pattern in (
        r"最终分数\s*[:：]\s*(\d{1,3})",
        r"最终分数[^0-9]{0,20}(\d{1,3})\s*/\s*100",
        r"综合置信度评分[^0-9]{0,30}(\d{1,3})",
    ):
        match = re.search(pattern, cleaned)
        if match and 0 <= int(match.group(1)) <= 100:
            return int(match.group(1))
    return None


def _round2(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def _change(values: list[float], periods: int) -> float | None:
    if len(values) <= periods or values[-periods - 1] == 0:
        return None
    return (values[-1] / values[-periods - 1] - 1) * 100


def _mean(values: list[float], window: int) -> float | None:
    if len(values) < window:
        return None
    return sum(values[-window:]) / window


def _rsi14(values: list[float]) -> float | None:
    if len(values) <= 14:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for previous, current in zip(values, values[1:]):
        delta = current - previous
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    average_gain = gains[0]
    average_loss = losses[0]
    alpha = 1 / 14
    for gain, loss in zip(gains[1:], losses[1:]):
        average_gain = (1 - alpha) * average_gain + alpha * gain
        average_loss = (1 - alpha) * average_loss + alpha * loss
    if average_loss == 0:
        return 100.0
    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def independent_technical_ground_truth(evidence: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_bars = evidence.get("raw_bars")
    if not isinstance(raw_bars, list):
        raise ValueError("technical evidence has no raw_bars list")
    by_date: dict[str, dict[str, Any]] = {}
    for raw in raw_bars:
        if not isinstance(raw, dict) or not raw.get("date") or raw.get("close") is None:
            continue
        by_date[str(raw["date"])] = raw
    bars = [by_date[key] for key in sorted(by_date)]
    if len(bars) < 2:
        raise ValueError("technical evidence has fewer than two usable bars")
    closes = [float(item["close"]) for item in bars]
    volumes = [float(item.get("volume") or 0) for item in bars]

    def range_min(field: str, window: int) -> float | None:
        values = [float(item[field]) for item in bars[-window:] if item.get(field) is not None]
        return min(values) if values else None

    def range_max(field: str, window: int) -> float | None:
        values = [float(item[field]) for item in bars[-window:] if item.get(field) is not None]
        return max(values) if values else None

    current = closes[-1]
    support20 = range_min("low", 20)
    resistance20 = range_max("high", 20)
    support60 = range_min("low", 60)
    resistance60 = range_max("high", 60)
    high252 = range_max("high", 252)
    low252 = range_min("low", 252)

    def distance(level: float | None) -> float | None:
        return None if level in {None, 0} else (current / float(level) - 1) * 100

    raw_values = {
        "current_price_usd": current,
        "previous_close_usd": closes[-2],
        "daily_change_percent": _change(closes, 1),
        "return_5d_percent": _change(closes, 5),
        "return_20d_percent": _change(closes, 20),
        "return_60d_percent": _change(closes, 60),
        "sma20_usd": _mean(closes, 20),
        "sma50_usd": _mean(closes, 50),
        "sma200_usd": _mean(closes, 200),
        "rsi14": _rsi14(closes),
        "latest_volume": volumes[-1],
        "average_volume_20d": _mean(volumes, 20),
        "average_volume_60d": _mean(volumes, 60),
        "volume_ratio_to_20d": (
            volumes[-1] / _mean(volumes, 20)
            if _mean(volumes, 20) not in {None, 0}
            else None
        ),
        "support_20d_usd": support20,
        "resistance_20d_usd": resistance20,
        "support_60d_usd": support60,
        "resistance_60d_usd": resistance60,
        "distance_from_support_20d_percent": distance(support20),
        "distance_from_resistance_20d_percent": distance(resistance20),
        "high_52_week_usd": high252,
        "low_52_week_usd": low252,
        "distance_from_52_week_high_percent": distance(high252),
        "distance_from_52_week_low_percent": distance(low252),
    }
    result = {}
    last_date = str(bars[-1]["date"])
    for metric, (key, unit, tolerance_group) in TECHNICAL_KEYS.items():
        raw_value = raw_values[key]
        if tolerance_group == "volume":
            value = int(raw_value) if raw_value is not None else None
        else:
            value = _round2(raw_value)
        result[metric] = {
            "value": value,
            "unit": unit,
            "period": last_date,
            "indicator_key": key,
            "tolerance_group": tolerance_group,
            "observation_count": len(bars),
        }
    return result


def _numeric_equal(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _technical_verification(
    claim: dict[str, Any],
    ground_truth: dict[str, dict[str, Any]],
    evidence: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    metric = str(claim.get("metric_name") or "")
    expected = ground_truth.get(metric)
    if expected is None or expected.get("value") is None:
        return _unverifiable("technical metric cannot be recomputed from saved bars")
    tolerance_group = expected["tolerance_group"]
    tolerances = rubric["rounding_tolerances"]
    tolerance = {
        "price": tolerances["technical_price_usd"],
        "percent": tolerances["technical_percent"],
        "rsi": tolerances["technical_rsi"],
        "ratio": tolerances["technical_ratio"],
        "volume": tolerances["technical_volume_shares"],
    }[tolerance_group]
    value_ok = _numeric_equal(claim.get("value"), expected["value"], tolerance)
    unit_ok = str(claim.get("unit") or "") == expected["unit"]
    period_ok = str(claim.get("period_or_as_of_date") or "") == expected["period"]
    anchor_ok = str(claim.get("source_anchor") or "").endswith(expected["indicator_key"])
    source_ok = str(claim.get("source_url") or "") == str(evidence.get("source_url") or "")
    correct = value_ok and unit_ok and period_ok and anchor_ok and source_ok
    notes = []
    for name, passed in (
        ("value", value_ok),
        ("unit", unit_ok),
        ("period", period_ok),
        ("anchor", anchor_ok),
        ("source_url", source_ok),
    ):
        if not passed:
            notes.append(f"{name}_mismatch")
    return {
        "accuracy": "correct" if correct else "incorrect",
        "numerical": "correct" if correct else "incorrect",
        "ground_truth": expected,
        "evidence_source": "saved_raw_bars_independent_recalculation",
        "evidence_url": evidence.get("source_url"),
        "notes": notes or [f"absolute_tolerance={tolerance}"],
    }


def _fact_records(
    company_facts: dict[str, Any], taxonomy: str, concept: str, unit: str
) -> list[dict[str, Any]]:
    values = (
        company_facts.get("facts", {})
        .get(taxonomy, {})
        .get(concept, {})
        .get("units", {})
        .get(unit, [])
    )
    return [item for item in values if isinstance(item, dict)]


def _anchor_records(
    company_facts: dict[str, Any], anchor: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    records = []
    descriptors = []
    for match in _ANCHOR.finditer(anchor):
        taxonomy, concept, unit, start, end, accession = match.groups()
        candidates = _fact_records(company_facts, taxonomy, concept, unit)
        selected = [
            item
            for item in candidates
            if str(item.get("end") or "") == end
            and (
                (start == "instant" and not item.get("start"))
                or str(item.get("start") or "") == start
            )
            and (not accession or str(item.get("accn") or "") == accession)
        ]
        if not selected:
            continue
        record = max(selected, key=lambda item: str(item.get("filed") or ""))
        records.append(record)
        descriptors.append(
            {
                "taxonomy": taxonomy,
                "concept": concept,
                "unit": unit,
                "start": start,
                "end": end,
                "accession": accession or str(record.get("accn") or ""),
            }
        )
    return records, descriptors


def _comparison_record(
    company_facts: dict[str, Any], anchor: str, current: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    match = _COMPARE_ANCHOR.search(anchor)
    if not match:
        return None, None
    taxonomy, concept, unit, end = match.groups()
    candidates = [
        item
        for item in _fact_records(company_facts, taxonomy, concept, unit)
        if str(item.get("end") or "") == end and item.get("val") is not None
    ]
    if current and current.get("start") and current.get("end"):
        current_days = (date.fromisoformat(current["end"]) - date.fromisoformat(current["start"])).days
        comparable = []
        for item in candidates:
            try:
                days = (date.fromisoformat(item["end"]) - date.fromisoformat(item["start"])).days
            except (KeyError, TypeError, ValueError):
                continue
            if abs(days - current_days) <= 3:
                comparable.append(item)
        candidates = comparable or candidates
    record = max(candidates, key=lambda item: str(item.get("filed") or ""), default=None)
    descriptor = (
        {"taxonomy": taxonomy, "concept": concept, "unit": unit, "end": end}
        if record
        else None
    )
    return record, descriptor


def _allowed_concepts(metric: str) -> set[str]:
    if metric in {"reporting_period", "filing_date", "revenue_yoy"}:
        return INDEPENDENT_CONCEPTS["revenue"]
    if metric == "net_income_yoy":
        return INDEPENDENT_CONCEPTS["net_income"]
    if metric == "free_cash_flow":
        return INDEPENDENT_CONCEPTS["operating_cash_flow"] | INDEPENDENT_CONCEPTS["capital_expenditure"]
    if metric == "net_margin":
        return INDEPENDENT_CONCEPTS["net_income"] | INDEPENDENT_CONCEPTS["revenue"]
    if metric == "equity_ratio":
        return INDEPENDENT_CONCEPTS["equity"] | INDEPENDENT_CONCEPTS["total_assets"]
    if metric == "operating_cash_flow_to_net_income":
        return INDEPENDENT_CONCEPTS["operating_cash_flow"] | INDEPENDENT_CONCEPTS["net_income"]
    return INDEPENDENT_CONCEPTS.get(metric, set())


def _scale_financial(value: float, unit: str) -> float:
    lowered = unit.lower()
    if "billion" in lowered:
        return value / 1_000_000_000
    if "million" in lowered:
        return value / 1_000_000
    if "thousand" in lowered:
        return value / 1_000
    return value


def _fundamental_unit_matches(
    metric: str,
    claim_unit: str,
    raw_units: set[str],
) -> bool:
    if len(raw_units) != 1:
        return False
    if metric in {"reporting_period", "filing_date"}:
        return claim_unit == "date"
    if metric in {"revenue_yoy", "net_income_yoy"}:
        return claim_unit == "percent"
    if metric in {
        "net_margin",
        "equity_ratio",
        "operating_cash_flow_to_net_income",
    }:
        return claim_unit == "ratio"
    raw_unit = next(iter(raw_units))
    if metric == "EPS":
        normalized_claim = claim_unit.replace("/shares", "/share")
        normalized_raw = raw_unit.replace("/shares", "/share")
        return normalized_claim == normalized_raw
    return claim_unit.startswith(raw_unit)


def _submissions_record(submissions: dict[str, Any], accession: str) -> dict[str, Any] | None:
    recent = submissions.get("filings", {}).get("recent", {})
    accessions = recent.get("accessionNumber") or []
    try:
        index = accessions.index(accession)
    except ValueError:
        return None

    def field(name: str) -> Any:
        values = recent.get(name) or []
        return values[index] if index < len(values) else None

    return {
        "form": field("form"),
        "period_end": field("reportDate"),
        "filing_date": field("filingDate"),
        "accession_number": accession,
        "primary_document": field("primaryDocument"),
    }


def _fundamental_verification(
    claim: dict[str, Any],
    company_facts: dict[str, Any],
    submissions: dict[str, Any],
    rubric: dict[str, Any],
) -> dict[str, Any]:
    metric = str(claim.get("metric_name") or "")
    if metric == "latest_filing_period":
        value = claim.get("value") if isinstance(claim.get("value"), dict) else {}
        accession = str(value.get("accession_number") or "")
        record = _submissions_record(submissions, accession)
        if record is None:
            return _unverifiable("filing accession not found in saved SEC Submissions")
        expected_url = f"https://data.sec.gov/submissions/CIK{str(company_facts.get('cik') or '').zfill(10)}.json"
        correct = (
            all(value.get(key) == record.get(key) for key in ("form", "period_end", "filing_date", "accession_number"))
            and claim.get("period_or_as_of_date") == record["period_end"]
            and claim.get("source_date") == record["filing_date"]
            and claim.get("source_url") == expected_url
        )
        return {
            "accuracy": "correct" if correct else "incorrect",
            "numerical": "not_applicable",
            "ground_truth": record,
            "evidence_source": "saved_sec_submissions",
            "evidence_url": expected_url,
            "notes": [] if correct else ["filing_metadata_mismatch"],
        }

    anchor = str(claim.get("source_anchor") or "")
    records, descriptors = _anchor_records(company_facts, anchor)
    if not records:
        return _unverifiable("source anchor did not resolve in saved SEC Company Facts")
    concepts = {item["concept"] for item in descriptors}
    if not concepts <= _allowed_concepts(metric):
        return {
            "accuracy": "incorrect",
            "numerical": "incorrect" if isinstance(claim.get("value"), (int, float)) else "not_applicable",
            "ground_truth": {"resolved_records": records, "descriptors": descriptors},
            "evidence_source": "saved_sec_companyfacts",
            "evidence_url": claim.get("source_url"),
            "notes": ["concept_not_allowed_by_independent_metric_mapping"],
        }
    raw_values = [float(item["val"]) for item in records]
    claim_unit = str(claim.get("unit") or "")
    raw_units = {item["unit"] for item in descriptors}
    unit_ok = _fundamental_unit_matches(metric, claim_unit, raw_units)
    period = str(records[0].get("end") or "")
    period_ok = str(claim.get("period_or_as_of_date") or "") == period
    filed_dates = {str(item.get("filed") or "") for item in records}
    source_date_ok = str(claim.get("source_date") or "") in filed_dates
    cik = str(company_facts.get("cik") or "").zfill(10)
    expected_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    source_ok = str(claim.get("source_url") or "") == expected_url

    expected: Any
    if metric == "reporting_period":
        expected = period
    elif metric == "filing_date":
        expected = str(records[0].get("filed") or "")
    elif metric in {"revenue_yoy", "net_income_yoy"}:
        previous, previous_descriptor = _comparison_record(company_facts, anchor, records[0])
        if previous is None or float(previous.get("val") or 0) == 0:
            return _unverifiable("comparison period did not resolve in saved Company Facts")
        if previous_descriptor and previous_descriptor["unit"] not in raw_units:
            unit_ok = False
        expected = round((raw_values[0] / float(previous["val"]) - 1) * 100, 2)
    elif metric == "free_cash_flow":
        expected = _scale_financial(raw_values[0] - raw_values[1], claim_unit) if len(raw_values) == 2 else None
    elif metric == "total_debt":
        expected = _scale_financial(sum(raw_values), claim_unit)
    elif metric == "net_margin":
        expected = round(raw_values[0] / raw_values[1], 4) if len(raw_values) == 2 and raw_values[1] else None
    elif metric == "equity_ratio":
        expected = round(raw_values[0] / raw_values[1], 4) if len(raw_values) == 2 and raw_values[1] else None
    elif metric == "operating_cash_flow_to_net_income":
        expected = round(raw_values[0] / raw_values[1], 4) if len(raw_values) == 2 and raw_values[1] else None
    else:
        expected = _scale_financial(raw_values[0], claim_unit)
    if expected is None:
        return _unverifiable("derived financial metric lacks compatible component facts")

    is_numeric = isinstance(claim.get("value"), (int, float)) and not isinstance(claim.get("value"), bool)
    if is_numeric:
        if metric in {"net_margin", "equity_ratio", "operating_cash_flow_to_net_income"}:
            tolerance = rubric["rounding_tolerances"]["financial_ratio"]
        elif metric in {"revenue_yoy", "net_income_yoy"}:
            tolerance = rubric["rounding_tolerances"]["financial_percent"]
        elif metric == "EPS":
            tolerance = rubric["rounding_tolerances"]["financial_eps"]
        else:
            tolerance = rubric["rounding_tolerances"]["financial_display_amount"]
        value_ok = _numeric_equal(claim.get("value"), expected, tolerance)
        numerical = "correct" if value_ok and unit_ok and period_ok else "incorrect"
    else:
        value_ok = claim.get("value") == expected
        numerical = "not_applicable"
    correct = value_ok and unit_ok and period_ok and source_date_ok and source_ok
    notes = []
    for name, passed in (
        ("value", value_ok),
        ("unit_or_currency", unit_ok),
        ("period", period_ok),
        ("source_date", source_date_ok),
        ("source_url", source_ok),
    ):
        if not passed:
            notes.append(f"{name}_mismatch")
    return {
        "accuracy": "correct" if correct else "incorrect",
        "numerical": numerical,
        "ground_truth": {
            "value": expected,
            "unit": claim_unit,
            "period": period,
            "records": records,
            "descriptors": descriptors,
        },
        "evidence_source": "saved_sec_companyfacts",
        "evidence_url": expected_url,
        "notes": notes,
    }


def _news_verification(claim: dict[str, Any], news_snapshot: dict[str, Any]) -> dict[str, Any]:
    metric_type = str(claim.get("claim_type") or "")
    url = str(claim.get("source_url") or "")
    value = claim.get("value") if isinstance(claim.get("value"), dict) else {}
    if metric_type == "news_event":
        news = news_snapshot.get("news") if isinstance(news_snapshot.get("news"), dict) else {}
        if news.get("error"):
            return _unverifiable("structured news snapshot could not be retrieved")
        article = next(
            (item for item in news.get("articles") or [] if str(item.get("url") or "") == url),
            None,
        )
        if article is None:
            return {
                "accuracy": "incorrect",
                "numerical": "not_applicable",
                "ground_truth": None,
                "evidence_source": "saved_structured_news_response",
                "evidence_url": url or None,
                "notes": ["article_url_not_found_in_snapshot"],
            }
        ticker = str(claim.get("company_or_ticker") or "")
        expected = {
            "title": article.get("headline"),
            "description": article.get("summary"),
            "company_match": ticker in (article.get("symbols") or []),
            "publisher": article.get("source"),
            "date": article.get("created_at"),
            "url": article.get("url"),
        }
        expected_boundary = classify_article_evidence(
            article,
            company_match=expected["company_match"],
        )
        expected["evidence_depth"] = expected_boundary["evidence_depth"]
        expected["full_text_available"] = expected_boundary["full_text_available"]
        claim_depth = claim.get("evidence_depth", expected["evidence_depth"])
        claim_full_text = claim.get(
            "full_text_available", expected["full_text_available"]
        )
        correct = (
            value.get("title") == expected["title"]
            and value.get("description") == expected["description"]
            and value.get("company_match") == expected["company_match"]
            and claim.get("source_name") == expected["publisher"]
            and claim.get("source_date") == expected["date"]
            and claim.get("period_or_as_of_date") == expected["date"]
            and claim_depth == expected["evidence_depth"]
            and claim_full_text == expected["full_text_available"]
        )
        return {
            "accuracy": "correct" if correct else "incorrect",
            "numerical": "not_applicable",
            "ground_truth": expected,
            "evidence_source": "saved_structured_news_response",
            "evidence_url": url,
            "notes": [] if correct else ["news_metadata_or_description_mismatch"],
        }

    filings = news_snapshot.get("filings") if isinstance(news_snapshot.get("filings"), dict) else {}
    if filings.get("error"):
        return _unverifiable("structured SEC filing-text snapshot could not be retrieved")
    accession = str(value.get("accession_number") or "")
    filing = next(
        (item for item in filings.get("filings") or [] if str(item.get("accession_number") or "") == accession),
        None,
    )
    if filing is None:
        return _unverifiable("filing accession not present in saved filing-text snapshot")
    primary = filing.get("primary_document") if isinstance(filing.get("primary_document"), dict) else {}
    expected_boundary = classify_filing_evidence(filing)
    possible_urls = {str(filing.get("filing_index_url") or ""), str(primary.get("url") or "")}
    description = str(value.get("description") or "").strip()
    formatted_items: list[str] = []
    for item in filing.get("item_descriptions") or []:
        if isinstance(item, dict):
            item_number = str(item.get("item") or "").strip()
            item_text = str(item.get("description") or "").strip()
            formatted = ": ".join(part for part in (item_number, item_text) if part)
        else:
            formatted = str(item).strip()
        if formatted:
            formatted_items.append(formatted)
    joined_items = "、".join(formatted_items)
    raw_text = " ".join(str(primary.get("text") or "").split())
    legacy_joined_items = "; ".join(
        str(item).strip()
        for item in filing.get("item_descriptions") or []
        if str(item).strip()
    )
    description_ok = (
        not description
        or description in {joined_items, legacy_joined_items}
        or raw_text.startswith(description)
    )
    readable = not primary.get("error") and bool(raw_text or joined_items)
    if not readable:
        return _unverifiable("filing document text was unavailable in saved snapshot")
    claim_depth = claim.get("evidence_depth", expected_boundary["evidence_depth"])
    claim_full_text = claim.get(
        "full_text_available", expected_boundary["full_text_available"]
    )
    correct = (
        value.get("form") == filing.get("form")
        and claim.get("source_date") == filing.get("filing_date")
        and url in possible_urls
        and description_ok
        and claim_depth == expected_boundary["evidence_depth"]
        and claim_full_text == expected_boundary["full_text_available"]
    )
    return {
        "accuracy": "correct" if correct else "incorrect",
        "numerical": "not_applicable",
        "ground_truth": {
            "form": filing.get("form"),
            "filing_date": filing.get("filing_date"),
            "report_date": filing.get("report_date"),
            "accession_number": accession,
            "urls": sorted(item for item in possible_urls if item),
            "description_verified": description_ok,
        },
        "evidence_source": "saved_structured_sec_filing_text_response",
        "evidence_url": url,
        "notes": [] if correct else ["filing_metadata_or_description_mismatch"],
    }


def _unverifiable(reason: str) -> dict[str, Any]:
    return {
        "accuracy": "unverifiable",
        "numerical": "unverifiable",
        "ground_truth": None,
        "evidence_source": "unavailable",
        "evidence_url": None,
        "notes": [reason],
    }


def _has_claim_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(_has_claim_value(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_has_claim_value(item) for item in value)
    return str(value).strip() != ""


def citation_integrity_details(
    citations: list[dict[str, Any]], report: str, registry: list[dict[str, Any]]
) -> dict[str, Any]:
    body = split_report_body(report)
    ids = [str(item.get("claim_id") or "") for item in citations]
    counts = Counter(ids)
    duplicate_ids = sorted(key for key, count in counts.items() if key and count > 1)
    known_ids = set(ids)
    body_ids = set(_ANY_CLAIM_MARKER.findall(body))
    unknown_ids = sorted(body_ids - known_ids)
    registered_urls = {str(item.get("source_url") or "") for item in registry if item.get("source_url")}
    body_urls = {match.rstrip(".,;") for match in _URL.findall(body)}
    unregistered_urls = sorted(body_urls - registered_urls)
    unused_source_ids = [
        str(item.get("source_id") or "")
        for item in registry
        if not (set(str(value) for value in item.get("claim_ids") or []) & body_ids)
    ]
    duplicate_source_ids = sorted(
        key
        for key, count in Counter(str(item.get("source_id") or "") for item in registry).items()
        if key and count > 1
    )
    return {
        "body_claim_ids": sorted(body_ids),
        "duplicate_claim_ids": duplicate_ids,
        "unknown_claim_ids": unknown_ids,
        "unregistered_urls": unregistered_urls,
        "unused_source_ids": unused_source_ids,
        "duplicate_source_ids": duplicate_source_ids,
        "citation_integrity": not any(
            (duplicate_ids, unknown_ids, unregistered_urls, unused_source_ids, duplicate_source_ids)
        ),
    }


def _citation_link_ok(
    claim: dict[str, Any],
    integrity: dict[str, Any],
    registry: list[dict[str, Any]],
    id_count: Counter[str],
) -> tuple[bool, list[str]]:
    claim_id = str(claim.get("claim_id") or "")
    notes = []
    if id_count[claim_id] != 1:
        notes.append("claim_id_not_unique")
    if claim_id not in set(integrity["body_claim_ids"]):
        notes.append("claim_marker_missing_from_report_body")
    matching = [item for item in registry if claim_id in [str(value) for value in item.get("claim_ids") or []]]
    if not matching:
        notes.append("claim_missing_from_source_registry")
    elif not any(str(item.get("source_url") or "") == str(claim.get("source_url") or "") for item in matching):
        notes.append("registry_url_mismatch")
    return not notes, notes


def evaluate_fact_claims(
    *,
    ticker: str,
    response: dict[str, Any],
    technical_evidence: dict[str, Any] | None,
    company_facts: dict[str, Any] | None,
    submissions: dict[str, Any] | None,
    news_snapshot: dict[str, Any] | None,
    rubric: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    citations = [item for item in response.get("citations") or [] if isinstance(item, dict)]
    registry = [item for item in response.get("source_registry") or [] if isinstance(item, dict)]
    report = str(response.get("report") or "")
    integrity = citation_integrity_details(citations, report, registry)
    id_count = Counter(str(item.get("claim_id") or "") for item in citations)
    technical_truth = None
    technical_error = None
    if technical_evidence:
        try:
            technical_truth = independent_technical_ground_truth(technical_evidence)
        except Exception as error:
            technical_error = f"{type(error).__name__}: {error}"
    rows = []
    seen_catalog: dict[str, set[str]] = {"technical": set(), "fundamental": set(), "news_events": set()}

    for claim in citations:
        agent = str(claim.get("agent") or "unknown")
        category = "news_events" if agent == "news" else agent
        metric = str(claim.get("metric_name") or "")
        if category in seen_catalog:
            seen_catalog[category].add(metric)
        has_value = _has_claim_value(claim.get("value"))
        if not has_value:
            verification = {
                "accuracy": "not_claimed",
                "numerical": "not_claimed",
                "ground_truth": None,
                "evidence_source": str(claim.get("source_name") or "unavailable"),
                "evidence_url": claim.get("source_url"),
                "notes": ["no factual value asserted; evaluated under completeness"],
            }
        elif agent == "technical":
            verification = (
                _technical_verification(claim, technical_truth, technical_evidence, rubric)
                if technical_truth is not None and technical_evidence is not None
                else _unverifiable(technical_error or "technical evidence unavailable")
            )
        elif agent == "fundamental":
            verification = (
                _fundamental_verification(claim, company_facts, submissions, rubric)
                if company_facts is not None and submissions is not None
                else _unverifiable("SEC snapshots unavailable")
            )
        elif agent == "news":
            verification = (
                _news_verification(claim, news_snapshot)
                if news_snapshot is not None
                else _unverifiable("structured news snapshot unavailable")
            )
        else:
            verification = _unverifiable("unknown claim agent")

        declared_status = str(claim.get("citation_status") or "unsupported")
        if not has_value:
            support_label = "not_claimed"
        elif declared_status == "unsupported":
            support_label = "unsupported"
        elif declared_status == "partial":
            support_label = "unverifiable"
        elif verification["accuracy"] == "unverifiable":
            support_label = "unverifiable"
        elif verification["accuracy"] == "correct":
            support_label = "supported"
        else:
            support_label = "unsupported"

        link_ok, link_notes = _citation_link_ok(claim, integrity, registry, id_count)
        marker_present = str(claim.get("claim_id") or "") in set(
            integrity["body_claim_ids"]
        )
        if not has_value or not claim.get("source_url"):
            citation_label = "not_evaluated"
        elif not marker_present:
            citation_label = "not_cited"
        elif verification["accuracy"] == "unverifiable" or declared_status == "partial":
            citation_label = "unverifiable"
        elif link_ok and support_label == "supported":
            citation_label = "correct"
        else:
            citation_label = "incorrect"

        numerical = verification["numerical"]
        if support_label == "unsupported" and numerical not in {"not_applicable", "not_claimed"}:
            numerical = "unsupported"

        rows.append(
            {
                "ticker": ticker,
                "claim_id": str(claim.get("claim_id") or ""),
                "category": category,
                "claim_type": str(claim.get("claim_type") or ""),
                "metric_name": metric,
                "agent_claim": str(claim.get("claim_text") or ""),
                "ground_truth": _json_cell(verification["ground_truth"]),
                "accuracy_label": verification["accuracy"],
                "support_label": support_label,
                "numerical_label": numerical,
                "citation_label": citation_label,
                "evidence_source": verification["evidence_source"],
                "evidence_url": str(verification.get("evidence_url") or ""),
                "notes": "; ".join(verification["notes"] + link_notes),
                "evidence_depth": claim.get("evidence_depth") or "",
                "full_text_available": claim.get("full_text_available", ""),
                "claim_scope_limited": claim.get("claim_scope_limited", ""),
                "limitation_disclosed": claim.get("limitation_disclosed", ""),
            }
        )

    catalog = rubric["required_claim_catalog"]
    for category in ("technical", "fundamental"):
        for metric in catalog[category]:
            if metric in seen_catalog[category]:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "claim_id": f"NOT-{category[:1].upper()}-{metric}",
                    "category": category,
                    "claim_type": "expected_claim_gap",
                    "metric_name": metric,
                    "agent_claim": "",
                    "ground_truth": "null",
                    "accuracy_label": "not_claimed",
                    "support_label": "not_claimed",
                    "numerical_label": "not_claimed",
                    "citation_label": "not_evaluated",
                    "evidence_source": "rubric_required_claim_catalog",
                    "evidence_url": "",
                    "notes": "expected claim was not asserted; completeness gap, not unsupported",
                }
            )
    news_claimed = any(
        row["category"] == "news_events" and row["accuracy_label"] != "not_claimed"
        for row in rows
    )
    if not news_claimed:
        rows.append(
            {
                "ticker": ticker,
                "claim_id": "NOT-N-event_or_filing",
                "category": "news_events",
                "claim_type": "expected_claim_gap",
                "metric_name": "event_or_filing",
                "agent_claim": "",
                "ground_truth": "null",
                "accuracy_label": "not_claimed",
                "support_label": "not_claimed",
                "numerical_label": "not_claimed",
                "citation_label": "not_evaluated",
                "evidence_source": "rubric_required_claim_catalog",
                "evidence_url": "",
                "notes": "no news or filing event asserted; evaluate disclosure under completeness",
            }
        )
    return rows, integrity


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rank = {"incorrect": 4, "unverifiable": 3, "correct": 2, "not_claimed": 1}
    unique: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("unique_claim_key") or row.get("claim_id") or f"row-{index}")
        current = unique.get(key)
        if current is None or rank.get(str(row.get("accuracy_label")), 0) > rank.get(str(current.get("accuracy_label")), 0):
            unique[key] = row
    unique_rows = list(unique.values())
    factual_correct = sum(row["accuracy_label"] == "correct" for row in unique_rows)
    factual_incorrect = sum(row["accuracy_label"] == "incorrect" for row in unique_rows)
    factual_unverifiable = sum(row["accuracy_label"] == "unverifiable" for row in unique_rows)
    factual_not_claimed = sum(row["accuracy_label"] == "not_claimed" for row in unique_rows)
    factual_denominator = factual_correct + factual_incorrect
    total_factual = len(unique_rows) - factual_not_claimed
    unsupported = sum(row["support_label"] == "unsupported" for row in unique_rows)
    numerical_correct = sum(row["numerical_label"] == "correct" for row in unique_rows)
    numerical_incorrect = sum(row["numerical_label"] == "incorrect" for row in unique_rows)
    numerical_unsupported = sum(row["numerical_label"] == "unsupported" for row in unique_rows)
    numerical_unverifiable = sum(row["numerical_label"] == "unverifiable" for row in unique_rows)
    numerical_denominator = numerical_correct + numerical_incorrect
    occurrence_rows = [
        row
        for row in rows
        if row.get("occurrence_id") and row.get("accuracy_label") != "not_claimed"
    ]
    if not occurrence_rows:
        occurrence_rows = [row for row in rows if row.get("accuracy_label") != "not_claimed"]
    # Citation coverage is occurrence-based: each body restatement needs its own marker.
    citation_correct = sum(row["citation_label"] == "correct" for row in occurrence_rows)
    citation_incorrect = sum(row["citation_label"] == "incorrect" for row in occurrence_rows)
    citation_unverifiable = sum(row["citation_label"] == "unverifiable" for row in occurrence_rows)
    citation_not_cited = sum(row["citation_label"] == "not_cited" for row in occurrence_rows)
    citation_denominator = citation_correct + citation_incorrect
    required_claims = len(occurrence_rows)
    return {
        "factual_correct": factual_correct,
        "factual_incorrect": factual_incorrect,
        "factual_unverifiable": factual_unverifiable,
        "factual_not_claimed": factual_not_claimed,
        "factual_accuracy_denominator": factual_denominator,
        "factual_accuracy_pct": _pct(factual_correct, factual_denominator),
        "unsupported_claims": unsupported,
        "total_factual_claims": total_factual,
        "unsupported_claim_rate_pct": _pct(unsupported, total_factual),
        "numerically_correct": numerical_correct,
        "numerically_incorrect": numerical_incorrect,
        "unsupported_numerical_claims": numerical_unsupported,
        "unverifiable_numerical_claims": numerical_unverifiable,
        "numerical_accuracy_denominator": numerical_denominator,
        "numerical_accuracy_pct": _pct(numerical_correct, numerical_denominator),
        "citation_correct": citation_correct,
        "citation_incorrect": citation_incorrect,
        "citation_unverifiable": citation_unverifiable,
        "citation_not_cited": citation_not_cited,
        "citation_accuracy_denominator": citation_denominator,
        "citation_accuracy_pct": _pct(citation_correct, citation_denominator),
        "citation_required_claims": required_claims,
        "validly_cited_claims": citation_correct,
        "citation_coverage_pct": _pct(citation_correct, required_claims),
        "unique_external_claims": total_factual,
        "external_claim_occurrences": len(occurrence_rows),
    }


def _agent_status(response: dict[str, Any], role: str) -> str:
    detail = (response.get("agent_statuses") or {}).get(role)
    return str(detail.get("status") or "failed") if isinstance(detail, dict) else "failed"


def build_report_row(
    *,
    run_id: str,
    ticker: str,
    response: dict[str, Any],
    fact_rows: list[dict[str, Any]],
    integrity: dict[str, Any],
    report_path: Path,
    technical_path: Path | None,
    registry_path: Path,
    inventory: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    completeness = evaluate_completeness(str(response.get("report") or ""))
    metrics = metric_summary(fact_rows)
    inventory_metrics = claim_inventory_summary(inventory, citations)
    classification = evidence_classification_summary(inventory)
    modules = {
        key: completeness[key]["completed"]
        for key in (
            "technical",
            "fundamental",
            "news_events",
            "risks",
            "catalysts",
            "bull_case",
            "bear_case",
            "final_synthesis",
        )
    }
    row = {
        "run_id": run_id,
        "ticker": ticker,
        **modules,
        "completed_modules": completeness["completed_modules"],
        "completeness_score": completeness["completeness_score"],
        "confidence_score": extract_confidence_score(str(response.get("report") or "")) or "",
        "technical_agent_status": _agent_status(response, "technical"),
        "fundamental_agent_status": _agent_status(response, "fundamental"),
        "news_agent_status": _agent_status(response, "news"),
        "strategy_agent_status": _agent_status(response, "strategy"),
        "freshness": _json_cell(response.get("freshness_summary") or {}),
        **metrics,
        "unique_external_claims": classification["unique_external_claims"],
        "external_claim_occurrences": classification["external_claim_occurrences"],
        "system_derived_claims": classification["system_derived"],
        "operational_disclosures": classification["operational_disclosure"],
        "analytical_judgments": classification["analytical_judgment"],
        "not_claimed_markers": classification["not_claimed_marker"],
        "legitimately_excluded_non_claims": classification["legitimately_excluded_non_claim"],
        "total_extracted_items": classification["total_extracted_items"],
        "reconciliation_closed": classification["reconciliation_closed"],
        "internal_system_claims": classification["internal_system_claims"],
        "traceable_internal_claims": classification["traceable_internal_claims"],
        "internal_traceability_coverage_pct": classification["internal_traceability_coverage_pct"],
        "opinion_premises_required": classification["opinion_premises_required"],
        "opinion_premises_cited": classification["opinion_premises_cited"],
        "opinion_premise_citation_coverage_pct": classification["opinion_premise_citation_coverage_pct"],
        "citation_integrity": integrity["citation_integrity"],
        "duplicate_claim_ids": _json_cell(integrity["duplicate_claim_ids"]),
        "unknown_claim_ids": _json_cell(integrity["unknown_claim_ids"]),
        "unregistered_urls": _json_cell(integrity["unregistered_urls"]),
        "unused_source_ids": _json_cell(integrity["unused_source_ids"]),
        **{
            key: _json_cell(value) if key == "registry_unused_claims" else value
            for key, value in inventory_metrics.items()
        },
        "report_path": _portable(report_path),
        "technical_evidence_path": _portable(technical_path) if technical_path else "",
        "source_registry_path": _portable(registry_path),
    }
    detail = {
        "modules": completeness,
        "metrics": metrics,
        "citation_integrity": integrity,
        "claim_inventory": inventory_metrics,
        "classification_reconciliation": classification,
    }
    return row, detail


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_ticker_list(tickers: Iterable[str]) -> list[str]:
    normalized = [str(ticker).strip().upper() for ticker in tickers]
    duplicates = sorted(key for key, count in Counter(normalized).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate tickers: {duplicates}")
    if not normalized or any(not ticker for ticker in normalized):
        raise ValueError("ticker list contains an empty value")
    return normalized


def _safe_error(error: Exception) -> dict[str, Any]:
    status = getattr(getattr(error, "response", None), "status_code", None)
    return {"error": type(error).__name__, "http_status": status}


def _sec_company(companies: dict[str, Any], ticker: str) -> dict[str, Any]:
    aliases = {ticker.upper(), ticker.upper().replace(".", "-")}
    for company in companies.values():
        if str(company.get("ticker") or "").upper() in aliases:
            return company
    raise ValueError(f"SEC ticker mapping not found for {ticker}")


def _get_json(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def _aggregate(report_rows: list[dict[str, Any]]) -> dict[str, Any]:
    additive = (
        "factual_correct",
        "factual_incorrect",
        "factual_unverifiable",
        "factual_not_claimed",
        "unsupported_claims",
        "total_factual_claims",
        "numerically_correct",
        "numerically_incorrect",
        "unsupported_numerical_claims",
        "unverifiable_numerical_claims",
        "citation_correct",
        "citation_incorrect",
        "citation_unverifiable",
        "citation_not_cited",
        "citation_required_claims",
        "validly_cited_claims",
        "completed_modules",
        "total_body_factual_claims",
        "registry_matched_claims",
        "unmatched_body_claims",
        "cited_body_claims",
        "uncited_body_claims",
        "invalid_citation_claims",
        "unique_external_claims",
        "external_claim_occurrences",
        "system_derived_claims",
        "operational_disclosures",
        "analytical_judgments",
        "not_claimed_markers",
        "legitimately_excluded_non_claims",
        "total_extracted_items",
        "internal_system_claims",
        "traceable_internal_claims",
        "opinion_premises_required",
        "opinion_premises_cited",
    )
    result = {key: sum(int(row[key]) for row in report_rows) for key in additive}
    factual_denominator = result["factual_correct"] + result["factual_incorrect"]
    numerical_denominator = result["numerically_correct"] + result["numerically_incorrect"]
    citation_denominator = result["citation_correct"] + result["citation_incorrect"]
    result.update(
        {
            "factual_accuracy_denominator": factual_denominator,
            "factual_accuracy_pct": _pct(result["factual_correct"], factual_denominator),
            "unsupported_claim_rate_pct": _pct(result["unsupported_claims"], result["total_factual_claims"]),
            "numerical_accuracy_denominator": numerical_denominator,
            "numerical_accuracy_pct": _pct(result["numerically_correct"], numerical_denominator),
            "citation_accuracy_denominator": citation_denominator,
            "citation_accuracy_pct": _pct(result["citation_correct"], citation_denominator),
            "citation_coverage_pct": _pct(result["validly_cited_claims"], result["citation_required_claims"]),
            "completeness_denominator": len(report_rows) * 8,
            "completeness_score": _pct(result["completed_modules"], len(report_rows) * 8),
            "claim_inventory_coverage_pct": _pct(
                result["registry_matched_claims"], result["total_body_factual_claims"]
            ),
            "internal_traceability_coverage_pct": _pct(
                result["traceable_internal_claims"], result["internal_system_claims"]
            ) if result["internal_system_claims"] else 100.0,
            "opinion_premise_citation_coverage_pct": _pct(
                result["opinion_premises_cited"], result["opinion_premises_required"]
            ) if result["opinion_premises_required"] else 100.0,
            "reconciliation_closed": all(bool(row.get("reconciliation_closed")) for row in report_rows),
        }
    )
    return result


async def run_preflight(
    output_directory: Path,
    *,
    run_id: str,
    tickers: Iterable[str] = PREFLIGHT_TICKERS,
) -> dict[str, Any]:
    tickers = validate_ticker_list(tickers)
    output_directory.mkdir(parents=True, exist_ok=False)
    directories = {
        name: output_directory / name
        for name in ("reports", "responses", "technical_evidence", "source_registry", "ground_truth")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["TECHNICAL_EVIDENCE_DIR"] = str(directories["technical_evidence"])

    from app import AnalysisRequest, analyze_stock
    from tools.filing_text_tool import get_recent_sec_filing_texts
    from tools.news_tool import get_stock_news

    sec_user_agent = os.getenv("SEC_USER_AGENT")
    if not sec_user_agent:
        raise RuntimeError("SEC_USER_AGENT is unavailable to the running program")
    sec_headers = {
        "User-Agent": sec_user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }
    rubric = load_rubric()
    fact_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    validation = {
        "duplicate_tickers": [],
        "missing_reports": [],
        "missing_technical_evidence": [],
        "missing_source_registries": [],
        "parse_failures": [],
        "forbidden_financial_displays": [],
    }

    with httpx.Client(timeout=45.0, headers=sec_headers, follow_redirects=True) as sec_client:
        companies = _get_json(sec_client, "https://www.sec.gov/files/company_tickers.json")
        _write_json(directories["ground_truth"] / "sec_company_tickers.json", companies)

        for ticker in tickers:
            safe = _safe_ticker(ticker)
            response_path = directories["responses"] / f"{safe}.json"
            report_path = directories["reports"] / f"{safe}.md"
            registry_path = directories["source_registry"] / f"{safe}.json"
            technical_path = directories["technical_evidence"] / f"{safe}.json"
            sec_directory = directories["ground_truth"] / "sec"
            news_directory = directories["ground_truth"] / "news"
            sec_directory.mkdir(parents=True, exist_ok=True)
            news_directory.mkdir(parents=True, exist_ok=True)

            try:
                response = await analyze_stock(AnalysisRequest(symbol=ticker))
                response_data = response.model_dump()
            except Exception as error:
                response_data = {"symbol": ticker, "report": "", "error": _safe_error(error)}
                validation["parse_failures"].append({"ticker": ticker, "stage": "analyze", **_safe_error(error)})
            _write_json(response_path, response_data)
            report_path.write_text(str(response_data.get("report") or ""), encoding="utf-8")
            forbidden = forbidden_financial_displays(str(response_data.get("report") or ""))
            if forbidden:
                validation["forbidden_financial_displays"].append(
                    {"ticker": ticker, "matches": forbidden}
                )

            source_value = (response_data.get("technical_evidence") or {}).get("evidence_path")
            technical_evidence = None
            if source_value and Path(str(source_value)).is_file():
                source_path = Path(str(source_value))
                technical_evidence = json.loads(source_path.read_text(encoding="utf-8"))
                if source_path.resolve() != technical_path.resolve():
                    shutil.copy2(source_path, technical_path)
            else:
                validation["missing_technical_evidence"].append(ticker)

            registry_payload = {
                "citations": response_data.get("citations") or [],
                "source_registry": response_data.get("source_registry") or [],
                "production_citation_summary": response_data.get("citation_summary") or {},
            }
            _write_json(registry_path, registry_payload)

            try:
                company = _sec_company(companies, ticker)
                cik = str(company["cik_str"]).zfill(10)
                company_facts = _get_json(
                    sec_client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                )
                submissions = _get_json(sec_client, f"https://data.sec.gov/submissions/CIK{cik}.json")
            except Exception as error:
                company_facts = None
                submissions = None
                validation["parse_failures"].append({"ticker": ticker, "stage": "sec", **_safe_error(error)})
            _write_json(sec_directory / f"{safe}_companyfacts.json", company_facts or {"error": "unavailable"})
            _write_json(sec_directory / f"{safe}_submissions.json", submissions or {"error": "unavailable"})

            news_snapshot: dict[str, Any] = {}
            try:
                news_snapshot["news"] = get_stock_news(ticker, limit=5)
            except Exception as error:
                news_snapshot["news"] = _safe_error(error)
            try:
                news_snapshot["filings"] = get_recent_sec_filing_texts(
                    ticker, lookback_days=90, filing_limit=3, max_characters_per_document=7000
                )
            except Exception as error:
                news_snapshot["filings"] = _safe_error(error)
            _write_json(news_directory / f"{safe}.json", news_snapshot)

            try:
                registry_fact_rows, integrity = evaluate_fact_claims(
                    ticker=ticker,
                    response=response_data,
                    technical_evidence=technical_evidence,
                    company_facts=company_facts,
                    submissions=submissions,
                    news_snapshot=news_snapshot,
                    rubric=rubric,
                )
                inventory = build_claim_inventory(
                    ticker,
                    str(response_data.get("report") or ""),
                    [item for item in response_data.get("citations") or [] if isinstance(item, dict)],
                    [item for item in response_data.get("internal_provenance") or [] if isinstance(item, dict)],
                )
                response_citations = [
                    item
                    for item in response_data.get("citations") or []
                    if isinstance(item, dict)
                ]
                ticker_fact_rows, inventory = evaluate_claim_inventory(
                    inventory, registry_fact_rows, response_citations
                )
                row, detail = build_report_row(
                    run_id=run_id,
                    ticker=ticker,
                    response=response_data,
                    fact_rows=ticker_fact_rows,
                    integrity=integrity,
                    report_path=report_path,
                    technical_path=technical_path if technical_path.is_file() else None,
                    registry_path=registry_path,
                    inventory=inventory,
                    citations=response_citations,
                )
                fact_rows.extend(ticker_fact_rows)
                inventory_rows.extend(inventory)
                report_rows.append(row)
                details[ticker] = detail
            except Exception as error:
                validation["parse_failures"].append({"ticker": ticker, "stage": "evaluate", **_safe_error(error)})

    for ticker in tickers:
        safe = _safe_ticker(ticker)
        if not (directories["reports"] / f"{safe}.md").is_file():
            validation["missing_reports"].append(ticker)
        if not (directories["source_registry"] / f"{safe}.json").is_file():
            validation["missing_source_registries"].append(ticker)

    fact_path = output_directory / "fact_evaluation.csv"
    report_evaluation_path = output_directory / "report_evaluation.csv"
    inventory_path = output_directory / "claim_inventory.csv"
    _write_csv(fact_path, fact_rows, FACT_FIELDS)
    _write_csv(report_evaluation_path, report_rows, REPORT_FIELDS)
    _write_csv(inventory_path, inventory_rows, CLAIM_INVENTORY_FIELDS)
    exceptions = [
        row
        for row in fact_rows
        if row["accuracy_label"] in {"incorrect", "unverifiable", "not_claimed"}
        or row["support_label"] == "unsupported"
        or row["citation_label"] in {"incorrect", "not_cited", "unverifiable"}
    ]
    exceptions_path = output_directory / "evaluation_exceptions.csv"
    unmatched_path = output_directory / "unmatched_body_claims.csv"
    _write_csv(exceptions_path, exceptions, FACT_FIELDS)
    _write_csv(
        unmatched_path,
        [item for item in inventory_rows if not item.get("registry_claim_id")],
        CLAIM_INVENTORY_FIELDS,
    )
    summary = {
        "run_id": run_id,
        "rubric_version": rubric["rubric_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "aggregate": _aggregate(report_rows),
        "reports": {row["ticker"]: row for row in report_rows},
        "details": details,
        "exceptions": exceptions,
        "validation": validation,
        "ground_truth_independence": rubric["principles"]["ground_truth_independence"],
        "files": {
            "fact_evaluation": _portable(fact_path),
            "report_evaluation": _portable(report_evaluation_path),
            "claim_inventory": _portable(inventory_path),
            "exceptions": _portable(exceptions_path),
            "unmatched_body_claims": _portable(unmatched_path),
        },
    }
    summary_path = output_directory / "evaluation_summary.json"
    _write_json(summary_path, summary)
    summary["files"]["summary"] = _portable(summary_path)
    return summary


def reevaluate_frozen_run(
    source_directory: Path,
    output_directory: Path,
    *,
    run_id: str,
    tickers: Iterable[str] = PREFLIGHT_TICKERS,
    apply_current_postprocessors: bool = False,
) -> dict[str, Any]:
    tickers = validate_ticker_list(tickers)
    if not source_directory.is_dir():
        raise FileNotFoundError(f"frozen input run not found: {source_directory}")
    output_directory.mkdir(parents=True, exist_ok=False)
    for name in ("reports", "responses", "technical_evidence", "source_registry", "ground_truth"):
        source = source_directory / name
        if source.is_dir():
            shutil.copytree(source, output_directory / name)

    rubric = load_rubric()
    fact_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    validation = {
        "duplicate_tickers": [],
        "missing_reports": [],
        "missing_technical_evidence": [],
        "missing_source_registries": [],
        "parse_failures": [],
        "forbidden_financial_displays": [],
    }
    for ticker in tickers:
        safe = _safe_ticker(ticker)
        response_path = output_directory / "responses" / f"{safe}.json"
        report_path = output_directory / "reports" / f"{safe}.md"
        technical_path = output_directory / "technical_evidence" / f"{safe}.json"
        registry_path = output_directory / "source_registry" / f"{safe}.json"
        companyfacts_path = output_directory / "ground_truth" / "sec" / f"{safe}_companyfacts.json"
        submissions_path = output_directory / "ground_truth" / "sec" / f"{safe}_submissions.json"
        news_path = output_directory / "ground_truth" / "news" / f"{safe}.json"
        if not report_path.is_file():
            validation["missing_reports"].append(ticker)
        if not technical_path.is_file():
            validation["missing_technical_evidence"].append(ticker)
        if not registry_path.is_file():
            validation["missing_source_registries"].append(ticker)
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if apply_current_postprocessors:
                from tools.citation_registry import finalize_cited_report
                from tools.financial_amount import normalize_financial_amount_display

                processed_report, processed_registry, processed_summary = finalize_cited_report(
                    str(response.get("report") or ""),
                    [item for item in response.get("citations") or [] if isinstance(item, dict)],
                )
                processed_report = normalize_financial_amount_display(processed_report)
                response["report"] = processed_report
                response["source_registry"] = processed_registry
                response["citation_summary"] = processed_summary
                _write_json(response_path, response)
                report_path.write_text(processed_report, encoding="utf-8")
                _write_json(
                    registry_path,
                    {
                        "citations": response.get("citations") or [],
                        "source_registry": processed_registry,
                        "production_citation_summary": processed_summary,
                    },
                )
            forbidden = forbidden_financial_displays(str(response.get("report") or ""))
            if forbidden:
                validation["forbidden_financial_displays"].append(
                    {"ticker": ticker, "matches": forbidden}
                )
            technical = (
                json.loads(technical_path.read_text(encoding="utf-8"))
                if technical_path.is_file()
                else None
            )
            company_facts = json.loads(companyfacts_path.read_text(encoding="utf-8"))
            submissions = json.loads(submissions_path.read_text(encoding="utf-8"))
            news = json.loads(news_path.read_text(encoding="utf-8"))
            registry_rows, integrity = evaluate_fact_claims(
                ticker=ticker,
                response=response,
                technical_evidence=technical,
                company_facts=company_facts,
                submissions=submissions,
                news_snapshot=news,
                rubric=rubric,
            )
            inventory = build_claim_inventory(
                ticker,
                str(response.get("report") or ""),
                [item for item in response.get("citations") or [] if isinstance(item, dict)],
                [item for item in response.get("internal_provenance") or [] if isinstance(item, dict)],
            )
            response_citations = [
                item
                for item in response.get("citations") or []
                if isinstance(item, dict)
            ]
            ticker_rows, inventory = evaluate_claim_inventory(
                inventory, registry_rows, response_citations
            )
            report_row, detail = build_report_row(
                run_id=run_id,
                ticker=ticker,
                response=response,
                fact_rows=ticker_rows,
                integrity=integrity,
                report_path=report_path,
                technical_path=technical_path if technical_path.is_file() else None,
                registry_path=registry_path,
                inventory=inventory,
                citations=response_citations,
            )
            fact_rows.extend(ticker_rows)
            inventory_rows.extend(inventory)
            report_rows.append(report_row)
            details[ticker] = detail
        except Exception as error:
            validation["parse_failures"].append(
                {"ticker": ticker, "stage": "reevaluate", **_safe_error(error)}
            )

    fact_path = output_directory / "fact_evaluation.csv"
    report_path = output_directory / "report_evaluation.csv"
    inventory_path = output_directory / "claim_inventory.csv"
    _write_csv(fact_path, fact_rows, FACT_FIELDS)
    _write_csv(report_path, report_rows, REPORT_FIELDS)
    _write_csv(inventory_path, inventory_rows, CLAIM_INVENTORY_FIELDS)
    exceptions = [
        row
        for row in fact_rows
        if row["accuracy_label"] in {"incorrect", "unverifiable", "not_claimed"}
        or row["support_label"] == "unsupported"
        or row["citation_label"] in {"incorrect", "not_cited", "unverifiable"}
    ]
    exceptions_path = output_directory / "evaluation_exceptions.csv"
    unmatched_path = output_directory / "unmatched_body_claims.csv"
    _write_csv(exceptions_path, exceptions, FACT_FIELDS)
    _write_csv(
        unmatched_path,
        [item for item in inventory_rows if not item.get("registry_claim_id")],
        CLAIM_INVENTORY_FIELDS,
    )
    summary = {
        "run_id": run_id,
        "source_run_id": source_directory.name,
        "frozen_snapshot_reevaluation": True,
        "current_postprocessors_applied": apply_current_postprocessors,
        "rubric_version": rubric["rubric_version"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": tickers,
        "aggregate": _aggregate(report_rows),
        "reports": {row["ticker"]: row for row in report_rows},
        "details": details,
        "exceptions": exceptions,
        "validation": validation,
        "ground_truth_independence": rubric["principles"]["ground_truth_independence"],
        "files": {
            "fact_evaluation": _portable(fact_path),
            "report_evaluation": _portable(report_path),
            "claim_inventory": _portable(inventory_path),
            "exceptions": _portable(exceptions_path),
            "unmatched_body_claims": _portable(unmatched_path),
        },
    }
    summary_path = output_directory / "evaluation_summary.json"
    _write_json(summary_path, summary)
    summary["files"]["summary"] = _portable(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 6A final-evaluation preflight")
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("phase6a-preflight-%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument(
        "--apply-current-postprocessors",
        action="store_true",
        help="Apply deterministic citation/value/display postprocessors to the frozen response before evaluation.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--input-run",
        type=Path,
        help="Re-evaluate a frozen run into a new run directory without calling production agents.",
    )
    args = parser.parse_args()
    output = args.output_root / args.run_id
    summary = (
        reevaluate_frozen_run(
            args.input_run,
            output,
            run_id=args.run_id,
            apply_current_postprocessors=args.apply_current_postprocessors,
        )
        if args.input_run
        else asyncio.run(run_preflight(output, run_id=args.run_id))
    )
    print(
        json.dumps(
            {
                "run_id": summary["run_id"],
                "output_directory": _portable(output),
                "aggregate": summary["aggregate"],
                "validation": summary["validation"],
                "files": summary["files"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
