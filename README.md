# US Stock Multi-Agent Analysis System

一个基于 Python、FastAPI 和 OpenAI Agents SDK 构建的美股多 Agent 分析系统。

用户输入股票代码后，系统会分别调用技术面、基本面和消息面 Agent，最后由策略 Agent 汇总结果，并通过网页生成结构化中文分析报告。

## 项目功能

- 技术面分析
  - 5日、20日和60日收益率
  - 20日、50日和200日均线
  - RSI
  - 成交量变化
  - 支撑位与阻力位

- 基本面分析
  - 营业收入和净利润
  - 收入及净利润同比增长
  - 经营现金流、资本支出和自由现金流
  - 现金、债务、资产和股东权益
  - 支持从 SEC 申报及附件中补充外国发行人的数据

- 消息面分析
  - 获取近期相关新闻
  - 股票代码相关性筛选
  - 新闻去重
  - 来源质量提示
  - 结合 SEC 8-K、6-K 和相关附件进行核验

- 综合分析
  - 对比技术面、基本面和消息面信号
  - 识别不同分析之间的支持与矛盾
  - 输出正面因素、风险因素和后续关注事项
  - 根据数据完整性、时效性和来源质量计算置信度

## 数据来源

- Yahoo Finance via `yfinance`
- SEC Company Facts API
- SEC EDGAR Submissions and Filing Documents
- Alpaca News API
- Benzinga news distributed through Alpaca

不同数据源可能存在更新时间差异。系统会在报告中说明数据日期、读取限制和可能的延迟。

## 技术栈

- Python
- FastAPI
- OpenAI Agents SDK
- Pydantic
- HTTPX
- yfinance
- HTML、CSS 和 JavaScript

## 项目结构

```text
us-stock-agent/
├── agents/
│   ├── strategist_agent.py
│   ├── technical_agent.py
│   ├── fundamental_agent.py
│   └── news_agent.py
├── tools/
│   ├── technical_tool.py
│   ├── fundamental_tool.py
│   ├── news_tool.py
│   ├── filing_tool.py
│   ├── filing_text_tool.py
│   └── confidence_tool.py
├── static/
│   └── index.html
├── app.py
├── requirements.txt
├── .env.example
└── README.md
