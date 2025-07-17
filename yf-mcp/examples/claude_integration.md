# Claude Integration Examples

This document shows how to use the Yahoo Finance MCP server with Claude for various financial analysis tasks.

## Setup

1. Start the Yahoo Finance MCP server:
```bash
yahoo-finance-mcp
```

2. Configure Claude to connect to the MCP server (specific steps depend on your Claude client)

## Example Conversations with Claude

### Stock Analysis

**You:** "Show me the historical stock prices for AAPL over the last 6 months with daily intervals."

**Claude will use:** `get_historical_prices(symbol="AAPL", period="6mo", interval="1d")`

**You:** "What are the key financial metrics for Tesla from the stock info?"

**Claude will use:** `get_stock_info(symbol="TSLA")`

### Financial Health Analysis

**You:** "Get the quarterly balance sheet for Microsoft."

**Claude will use:** `get_financial_statements(symbol="MSFT", statement_type="balance_sheet", period="quarterly")`

**You:** "Compare the quarterly income statements of Amazon and Google."

**Claude will use:** 
- `get_financial_statements(symbol="AMZN", statement_type="income_statement", period="quarterly")`
- `get_financial_statements(symbol="GOOGL", statement_type="income_statement", period="quarterly")`

**You:** "Show me the annual cash flow statement for NVIDIA."

**Claude will use:** `get_financial_statements(symbol="NVDA", statement_type="cash_flow", period="annual")`

### Market Research

**You:** "Get the latest news articles about Meta Platforms."

**Claude will use:** `get_stock_news(symbol="META", limit=10)`

**You:** "Show me the institutional holders of Apple stock."

**Claude will use:** `get_institutional_holders(symbol="AAPL")`

**You:** "What are the recent insider transactions for Tesla?"

**Claude will use:** `get_insider_transactions(symbol="TSLA")`

**You:** "What are the analyst recommendations for Amazon over the last 3 months?"

**Claude will use:** `get_analyst_recommendations(symbol="AMZN")`

### Options Analysis

**You:** "Get the options chain for SPY with expiration date 2024-06-21 for calls."

**Claude will use:** `get_options_chain(symbol="SPY", expiration="2024-06-21", option_type="calls")`

First, Claude might check available expirations:
`get_options_expirations(symbol="SPY")`

### Complex Analysis Requests

**You:** "Create a comprehensive analysis of Microsoft's financial health using their latest quarterly financial statements."

**Claude will use multiple tools:**
1. `get_stock_info(symbol="MSFT")`
2. `get_financial_statements(symbol="MSFT", statement_type="balance_sheet", period="quarterly")`
3. `get_financial_statements(symbol="MSFT", statement_type="income_statement", period="quarterly")`
4. `get_financial_statements(symbol="MSFT", statement_type="cash_flow", period="quarterly")`
5. `get_analyst_recommendations(symbol="MSFT")`

**You:** "Compare the dividend history and stock splits of Coca-Cola and PepsiCo."

**Claude will use:**
1. `get_dividend_history(symbol="KO")`
2. `get_dividend_history(symbol="PEP")`
3. `get_stock_splits(symbol="KO")`
4. `get_stock_splits(symbol="PEP")`
5. `compare_stocks(symbols=["KO", "PEP"])`

**You:** "Analyze the institutional ownership changes in Tesla over the past year."

**Claude will use:**
1. `get_institutional_holders(symbol="TSLA")`
2. `get_stock_info(symbol="TSLA")`
3. Additional analysis of the institutional data

**You:** "Generate a report on the options market activity for Apple stock with expiration in 30 days."

**Claude will use:**
1. `get_options_expirations(symbol="AAPL")`
2. `get_options_chain(symbol="AAPL", expiration="[nearest 30-day expiration]", option_type="both")`
3. `get_stock_info(symbol="AAPL")`

**You:** "Summarize the latest analyst upgrades and downgrades in the tech sector over the last 6 months."

**Claude will use:**
1. `get_analyst_recommendations(symbol="AAPL")`
2. `get_analyst_recommendations(symbol="MSFT")`
3. `get_analyst_recommendations(symbol="GOOGL")`
4. `get_analyst_recommendations(symbol="AMZN")`
5. `get_analyst_recommendations(symbol="META")`
6. And potentially other tech stocks

## Tips for Effective Use

### Be Specific with Stock Symbols
- Use standard ticker symbols (AAPL, MSFT, GOOGL, etc.)
- For companies with multiple share classes, specify (e.g., GOOGL vs GOOG)

### Time Periods
- For historical data, you can request: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- For intervals: 1m, 2m, 5m, 15m, 30m, 1h, 1d, 5d, 1wk, 1mo, 3mo

### Financial Statements
- Available types: balance_sheet, income_statement, cash_flow
- Available periods: quarterly, annual

### Options Data
- Always check expiration dates first with `get_options_expirations`
- Use YYYY-MM-DD format for expiration dates
- Specify calls, puts, or both

### Comparative Analysis
- Use `compare_stocks` for side-by-side metric comparison
- Limit to 10 stocks maximum for readability

## Error Handling

The server includes comprehensive error handling for:
- Invalid stock symbols
- Market closures and data availability
- Network timeouts
- Rate limiting

If you encounter errors, Claude will provide helpful feedback about what went wrong and suggest alternatives. 