import sys
from pathlib import Path

from agents import Runner


project_directory = Path(__file__).resolve().parent
agent_directory = project_directory / "agents"

sys.path.insert(0, str(agent_directory))

from strategist_agent import strategist_agent


def get_stock_symbol() -> str:
    while True:
        symbol = input(
            "请输入美股代码，例如 AAPL、MSFT 或 NVDA："
        ).strip().upper()

        is_valid = (
            symbol
            and len(symbol) <= 10
            and all(
                character.isalnum() or character in ".-"
                for character in symbol
            )
        )

        if is_valid:
            return symbol

        print("股票代码格式不正确，请重新输入。")


def main() -> None:
    print("=" * 50)
    print("美股多 Agent 分析系统")
    print("=" * 50)

    symbol = get_stock_symbol()

    print(f"\n正在分析 {symbol}，请等待……\n")

    request = f"""
请综合分析美股 {symbol}。

必须分别调用：
1. 技术面分析师
2. 基本面分析师
3. 消息面分析师

所有专业分析师都必须分析 {symbol}，
不得分析其他股票。
最后给出统一的综合结论。
"""

    result = Runner.run_sync(
        strategist_agent,
        request,
    )

    print("\n" + "=" * 50)
    print(f"{symbol} 综合分析报告")
    print("=" * 50)
    print(result.final_output)


if __name__ == "__main__":
    main()
