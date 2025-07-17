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

### Portfolio Management
- Portfolio composition analysis
- Performance metrics
  - Total return
  - Annual return
  - Volatility
  - Sharpe ratio
- Risk metrics
  - Maximum drawdown
  - Value at Risk (VaR)
  - Portfolio beta
- Custom portfolio weights

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
| `analyze_portfolio` | Analyze a portfolio of stocks |
| `get_economic_indicators` | Get comprehensive market indicators |
| `get_market_summary` | Get quick market overview |

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
      "command": "uv",
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

- "What's the current price of AAPL?"
- "Compare MSFT, GOOGL, and AMZN"
- "Get the latest news about NVDA"
- "Analyze a portfolio of AAPL, MSFT, and GOOGL with equal weights"
- "Show me the price of Bitcoin"
- "What are the current treasury yields?"
