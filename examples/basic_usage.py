"""
Basic usage examples for Yahoo Finance MCP Server.

This file demonstrates how to use the various tools provided by the server.
Note: This is for illustration - actual usage would be through MCP client integration.
"""

import asyncio
from yahoo_finance_mcp.tools import YahooFinanceTools


async def example_stock_analysis():
    """Example: Comprehensive stock analysis."""
    print("=== Stock Analysis Example ===")
    tools = YahooFinanceTools()
    
    # Get basic stock information
    print("\n1. Getting stock information for AAPL:")
    result = await tools.handle_tool_call("get_stock_info", {"symbol": "AAPL"})
    print(result[0].text)
    
    # Get historical prices
    print("\n2. Getting 6 months of historical prices:")
    result = await tools.handle_tool_call("get_historical_prices", {
        "symbol": "AAPL",
        "period": "6mo",
        "interval": "1d"
    })
    print(result[0].text[:500] + "...")  # Truncate for display


async def example_financial_statements():
    """Example: Financial statement analysis."""
    print("\n=== Financial Statements Example ===")
    tools = YahooFinanceTools()
    
    # Get quarterly balance sheet
    print("\n1. Getting quarterly balance sheet for MSFT:")
    result = await tools.handle_tool_call("get_financial_statements", {
        "symbol": "MSFT",
        "statement_type": "balance_sheet",
        "period": "quarterly"
    })
    print(result[0].text[:800] + "...")  # Truncate for display
    
    # Get annual cash flow
    print("\n2. Getting annual cash flow for NVDA:")
    result = await tools.handle_tool_call("get_financial_statements", {
        "symbol": "NVDA",
        "statement_type": "cash_flow",
        "period": "annual"
    })
    print(result[0].text[:800] + "...")  # Truncate for display


async def example_market_research():
    """Example: Market research and news analysis."""
    print("\n=== Market Research Example ===")
    tools = YahooFinanceTools()
    
    # Get latest news
    print("\n1. Getting latest news for META:")
    result = await tools.handle_tool_call("get_stock_news", {
        "symbol": "META",
        "limit": 3
    })
    print(result[0].text)
    
    # Get analyst recommendations
    print("\n2. Getting analyst recommendations for AMZN:")
    result = await tools.handle_tool_call("get_analyst_recommendations", {
        "symbol": "AMZN"
    })
    print(result[0].text)


async def example_options_analysis():
    """Example: Options market analysis."""
    print("\n=== Options Analysis Example ===")
    tools = YahooFinanceTools()
    
    # Get options expiration dates
    print("\n1. Getting options expirations for SPY:")
    result = await tools.handle_tool_call("get_options_expirations", {
        "symbol": "SPY"
    })
    print(result[0].text)
    
    # Note: For options chain, you would use a specific expiration date
    # print("\n2. Getting options chain for SPY:")
    # result = await tools.handle_tool_call("get_options_chain", {
    #     "symbol": "SPY",
    #     "expiration": "2024-06-21",  # Use actual expiration date
    #     "option_type": "calls"
    # })
    # print(result[0].text)


async def example_comparative_analysis():
    """Example: Comparative stock analysis."""
    print("\n=== Comparative Analysis Example ===")
    tools = YahooFinanceTools()
    
    # Compare multiple tech stocks
    print("\n1. Comparing tech giants:")
    result = await tools.handle_tool_call("compare_stocks", {
        "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"]
    })
    print(result[0].text)
    
    # Get dividend history
    print("\n2. Getting dividend history for KO:")
    result = await tools.handle_tool_call("get_dividend_history", {
        "symbol": "KO"
    })
    print(result[0].text)


async def example_institutional_data():
    """Example: Institutional and insider data."""
    print("\n=== Institutional Data Example ===")
    tools = YahooFinanceTools()
    
    # Get institutional holders
    print("\n1. Getting institutional holders for TSLA:")
    result = await tools.handle_tool_call("get_institutional_holders", {
        "symbol": "TSLA"
    })
    print(result[0].text)
    
    # Get insider transactions
    print("\n2. Getting insider transactions for TSLA:")
    result = await tools.handle_tool_call("get_insider_transactions", {
        "symbol": "TSLA"
    })
    print(result[0].text)


async def run_all_examples():
    """Run all examples."""
    try:
        await example_stock_analysis()
        await example_financial_statements()
        await example_market_research()
        await example_options_analysis()
        await example_comparative_analysis()
        await example_institutional_data()
        
        print("\n=== All examples completed successfully! ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Note: Some examples may fail if market is closed or data is unavailable.")


if __name__ == "__main__":
    print("Yahoo Finance MCP Server - Basic Usage Examples")
    print("=" * 50)
    print("Note: These examples demonstrate the tool capabilities.")
    print("In actual usage, you would interact through an MCP client.\n")
    
    asyncio.run(run_all_examples()) 