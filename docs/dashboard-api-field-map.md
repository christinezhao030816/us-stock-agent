# Dashboard API Field Map

The Interview Release consumes the existing `POST /analyze` response without changing the API contract. The dashboard requests `language: "en"` and does not use a browser-side LLM or translation service.

| Dashboard area | Primary API field | Deterministic fallback | NVDA verification value |
|---|---|---|---|
| Ticker | `symbol` | Requested ticker | `NVDA` |
| Company name | `company_name` | Company name parsed from the exact `Ticker and Data Dates` line | `NVIDIA Corporation` |
| Exchange | `exchange` | None | Not available |
| Current price | `technical_evidence.indicator_evidence.current_price_usd.rounded_result` | `current_price` indicator | `178.67 USD` |
| Price date | `technical_evidence.last_trading_date` | `technical_price_date` | `2026-11-09` |
| 20-day return | `technical_evidence.indicator_evidence.return_20d_percent.rounded_result` | `20_day_return` indicator | `-21.05%` |
| Technical trend | Current price compared with saved SMA20 and SMA50 inputs | Not available unless all three values exist | Bearish |
| Agent status | `agent_statuses.<agent>` | No data returned | Technical success; Fundamental, News, and Strategy partial |
| Agent narrative | `agent_details.<agent>` when supplied | Exact Markdown section parser | Response did not include `agent_details`; exact sections were used |
| Strategy citations | T/F/N markers present in the extracted Strategy narrative | None | 32 unique inherited claims, deduplicated and validated against `source_registry[].claim_ids` |
| News retrieval depth | News citations and `agent_statuses.news.missing_items` | None | 10 news articles, including headline-only and truncated evidence |
| Fundamental freshness | `freshness_summary.fundamental_freshness_status` | `fundamental_freshness.summary` | `caution` |
| Period alignment | `freshness_summary.latest_period_alignment_status` | `fundamental_freshness.summary` | `aligned` |
| Citation coverage | `citation_summary.citation_coverage_pct` | None | `95.92%` |
| Research Usability | `research_usability` if supplied | Deterministic label from freshness, alignment, and Agent status | Usable with limitations; no accuracy score is inferred |
| Evidence drawer | `citations`, `source_registry`, `internal_provenance` | Empty-state disclosure | 56 claims, 16 registered sources, 7 provenance records |

## Status presentation

- `success` with core data → **Success**
- `partial` with core data → **Completed with limitations**
- `partial` without core data → **Partial** or **No data returned**, depending on `core_data_present`
- `failed` → **Failed**
- Missing response object → **No data returned**

The presentation label never changes the backend status. It clarifies whether data was retrieved but constrained by missing fields or evidence depth.

## Citation counting

Each Agent count is based on citation markers that actually appear in that Agent's displayed narrative. IDs are deduplicated and counted only when they exist in `source_registry[].claim_ids`. Strategy therefore counts inherited `[Tn]`, `[Fn]`, and `[Nn]` markers and never fabricates `[Sn]` claims.
