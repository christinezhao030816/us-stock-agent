import csv
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parent

FACTS = [
    ("fundamental", "revenue"),
    ("fundamental", "net_income"),
    ("fundamental", "eps"),
    ("fundamental", "operating_cash_flow"),
    ("fundamental", "total_assets"),
    ("fundamental", "total_liabilities"),
    ("technical", "current_price"),
    ("technical", "twenty_day_return"),
    ("technical", "moving_average"),
    ("technical", "rsi"),
    ("news", "event_exists"),
    ("news", "event_date"),
    ("news", "company_match"),
    ("news", "description_accuracy"),
    ("news", "source_exists"),
]


def read_csv(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return list(csv.DictReader(file))


def main() -> None:
    holdings = read_csv(
        EVALUATION_DIR / "deep_evaluation_holdings.csv"
    )
    results = read_csv(
        EVALUATION_DIR / "full_portfolio_results.csv"
    )

    results_by_ticker = {
        row["ticker"].strip().upper(): row
        for row in results
    }

    output_path = (
        EVALUATION_DIR / "deep_evaluation_template.csv"
    )

    fieldnames = [
        "ticker",
        "sector",
        "category",
        "fact_name",
        "report_path",
        "claim_present",
        "agent_claim",
        "ground_truth",
        "accuracy_label",
        "support_label",
        "evidence_source",
        "evidence_url",
        "evidence_date",
        "notes",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for holding in holdings:
            ticker = holding["ticker"].strip().upper()
            result = results_by_ticker.get(ticker, {})

            for category, fact_name in FACTS:
                writer.writerow(
                    {
                        "ticker": ticker,
                        "sector": holding["sector"],
                        "category": category,
                        "fact_name": fact_name,
                        "report_path": result.get(
                            "raw_output_path",
                            "",
                        ),
                        "claim_present": "",
                        "agent_claim": "",
                        "ground_truth": "",
                        "accuracy_label": "",
                        "support_label": "",
                        "evidence_source": "",
                        "evidence_url": "",
                        "evidence_date": "",
                        "notes": "",
                    }
                )

    print(f"已生成: {output_path}")
    print(f"股票数: {len(holdings)}")
    print(f"核验项数: {len(holdings) * len(FACTS)}")


if __name__ == "__main__":
    main()
