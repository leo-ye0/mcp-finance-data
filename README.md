# Yahoo Finance MCP Server

A comprehensive financial data analysis server built with the Model Context Protocol (MCP), providing extensive Yahoo Finance data through a clean and efficient interface. This server offers real-time market data, portfolio management, and economic indicators.

## 🚀 Features

### Stock & Crypto Information
- Real-time price data
- Historical prices with customizable intervals
- Market capitalization
- Trading volume
- 52-week highs and lows
- Sector and industry classification (stocks)
- Cryptocurrency support (BTC-USD, ETH-USD, etc.)

### Market Data
- Stock news and updates
- Analyst recommendations
- Options data and chains
- Stock actions (dividends, splits)

### Economic Indicators
- **Major Indices**
  - S&P 500
  - Dow Jones
  - NASDAQ
  - Russell 2000

- **Market Metrics**
  - Treasury Yields (2Y, 5Y, 10Y, 30Y)
  - Volatility Indices (VIX, VXN)
  - Commodities (Gold, Oil, Silver)
  - Forex Rates (EUR/USD, GBP/USD, USD/JPY)

## 🛠️ Installation

1. Ensure you have Python 3.11.5 or later installed
2. Clone the repository:
```bash
git clone https://github.com/leo-ye0/yahoo-finance-mcp.git
cd yahoo-finance-mcp
```

3. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

4. Install the package and dependencies:
```bash
pip install -e .
```

## 🚀 Usage

### Starting the Server

```bash
python src/yahoo_finance_mcp/server.py
```

### Example Usage

```python
# Get stock/crypto information
await get_stock_info("AAPL")  # For stocks
await get_stock_info("BTC-USD")  # For cryptocurrencies

# Portfolio Analysis
await analyze_portfolio(
    symbols=["AAPL", "MSFT", "GOOGL"],
    weights=[0.4, 0.3, 0.3]
)

# Get economic indicators
await get_economic_indicators()  # Get comprehensive market data

# Get market summary
await get_market_summary()  # Quick overview of major indices
```

## 📊 Available Tools

| Tool Name | Description |
|-----------|-------------|
| `get_stock_info` | Get comprehensive stock/crypto information |
| `get_historical_prices` | Get historical price data with customizable periods |
| `get_financial_statements` | Get income statement, balance sheet, and cash flow data |
| `get_stock_news` | Get latest news articles for a stock |
| `get_analyst_recommendations` | Get analyst recommendations and recent changes |
| `get_options_data` | Get options data including calls and puts |
| `compare_stocks` | Compare multiple stocks across various metrics |
| `get_market_summary` | Get quick overview of major market indices |
| `get_holder_info` | Get information about major holders, institutional investors, and insider trades |
| `get_earnings_info` | Get earnings dates, estimates, history, and trends |
| `get_dividend_info` | Get dividend rates, yield, history, and next payment date |
| `get_balance_sheet_analysis` | Get detailed balance sheet analysis with key ratios and trends |
| `get_cash_flow_analysis` | Get comprehensive cash flow analysis and metrics |
| `get_financial_ratios` | Get key financial ratios across multiple categories |

## 🔧 Dependencies

- Python >=3.11.5
- yfinance >=0.2.18
- pandas >=1.5.0
- numpy >=1.21.0
- pydantic >=2.0.0
- fastmcp >=2.10.5
- mcp >=0.1.0

## 🤖 Claude Integration

### Setup with Claude Desktop

1. Install the server:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Open Claude's config:
```bash
# On MacOS
code ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

3. Add this to your config:
```json
{
  "mcpServers": {
    "yfinance": {
      "command": "uv path like /Users/xxxxx/anaconda3/bin/uv",
      "args": [
        "--directory",
        "PATH/TO/FOLDER/yahoo-finance-mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

### Example Prompts

Once connected, you can ask Claude things like:

- "What's the current price and info for AAPL?"
- "Get historical prices for MSFT over the last 6 months"
- "Show me the latest financial statements for GOOGL"
- "What's the latest news about NVDA?"
- "Get analyst recommendations for AMZN"
- "Show me options data for TSLA"
- "Compare the market cap and PE ratios of AAPL, MSFT, and GOOGL"
- "What's the current market summary for major indices?"
- "Who are the major institutional holders of AAPL?"
- "Show me recent insider transactions for MSFT"
- "When is the next earnings date for AAPL?"
- "What's the dividend history and yield for MSFT?"
- "Analyze AAPL's balance sheet metrics and trends"
- "Show me GOOGL's cash flow analysis"
- "What are the key financial ratios for NVDA?"
