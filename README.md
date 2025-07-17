# Yahoo Finance MCP Server

A comprehensive financial data analysis server built with the Model Context Protocol (MCP), providing extensive Yahoo Finance data through a clean and efficient interface. This server offers real-time stock data, technical analysis, portfolio management, economic indicators, cryptocurrency data, and ESG scores.

## 🚀 Features

### Stock Information
- Comprehensive stock details
- Real-time and historical prices
- Company information and financials
- Market capitalization and trading metrics
- Dividend information
- Sector and industry classification

### Technical Analysis
- Simple Moving Averages (50 and 200-day)
- Exponential Moving Averages (12 and 26-day)
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Bollinger Bands
- Stochastic Oscillator

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

### Financial Statements
- Income statements
- Balance sheets
- Cash flow statements
- Quarterly and annual data

### Market Data
- Stock news and updates
- Analyst recommendations
- Options data and chains
- Stock actions (dividends, splits)

### Economic Indicators
- Major market indices
- Treasury yields
- Volatility indices
- Price changes and trends

### Cryptocurrency
- Real-time crypto prices
- Market capitalization
- Trading volume
- Supply information
- Price changes

### ESG Analysis
- Overall ESG scores
- Environmental scores
- Social scores
- Governance scores
- Controversy levels
- Peer comparison

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
# Get stock information
await get_stock_info("AAPL")

# Get historical prices
await get_historical_prices("MSFT", period="1y", interval="1d")

# Technical Analysis
await get_technical_indicators("GOOGL", interval="1d")

# Portfolio Analysis
await analyze_portfolio(
    symbols=["AAPL", "MSFT", "GOOGL"],
    weights=[0.4, 0.3, 0.3]
)

# Financial Statements
await get_financial_statements("TSLA", statement_type="income_statement", period="annual")

# Market News
await get_stock_news("NVDA", max_articles=5)

# Economic Indicators
await get_economic_indicators(["^TNX", "^VIX", "^DJI"])

# Cryptocurrency Data
await get_crypto_data(["BTC-USD", "ETH-USD", "DOGE-USD"])

# ESG Scores
await get_esg_scores("MSFT")
```

## 📊 Available Tools

| Tool Name | Description |
|-----------|-------------|
| `get_stock_info` | Get comprehensive stock information |
| `get_historical_prices` | Get historical price data with customizable periods |
| `get_technical_indicators` | Get technical analysis indicators |
| `analyze_portfolio` | Analyze a portfolio of stocks |
| `get_financial_statements` | Get financial statements |
| `get_stock_news` | Get latest news articles |
| `get_analyst_recommendations` | Get analyst recommendations and changes |
| `get_options_data` | Get options chain data |
| `compare_stocks` | Compare multiple stocks |
| `get_market_summary` | Get market indices summary |
| `get_economic_indicators` | Get major economic indicators |
| `get_crypto_data` | Get cryptocurrency data |
| `get_esg_scores` | Get ESG scores and rankings |

## 📈 Data Sources

All data is sourced from Yahoo Finance through the `yfinance` library, providing:
- Real-time and historical market data
- Company financial information
- News and analysis
- Cryptocurrency markets
- ESG ratings

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
# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Open Claude's config:
```bash
# On MacOS
code ~/Library/Application\ Support/Claude/claude_desktop_config.json


3. Add this to your config:
```json
{
  "mcpServers": {
    "yfinance": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/yahoo-finance-mcp",
        "run",
        "server.py"
      ]
    }
  }
}
```

Replace `/path/to/yahoo-finance-mcp` with your actual installation path.

### Example Prompts

Once connected, you can ask Claude things like:

- "What's the current price of AAPL?"
- "Show me technical indicators for TSLA"
- "Compare MSFT, GOOGL, and AMZN"
- "Get the latest news about NVDA"
- "Analyze a portfolio of AAPL, MSFT, and GOOGL with equal weights"
- "What are the ESG scores for MSFT?"
- "Show me the price of Bitcoin and Ethereum"
