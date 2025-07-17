"""MCP tools for Yahoo Finance server."""

from typing import Any, Dict, List, Optional
from mcp.types import Tool, TextContent
from .client import YahooFinanceClient
from .types import StatementType, Period, Interval, TimePeriod, OptionType
import json
from datetime import datetime


class YahooFinanceTools:
    """MCP tools for Yahoo Finance data."""
    
    def __init__(self):
        """Initialize the tools with a Yahoo Finance client."""
        self.client = YahooFinanceClient()
    
    def get_tools(self) -> List[Tool]:
        """Get all available MCP tools."""
        return [
            Tool(
                name="get_stock_info",
                description="Get comprehensive information about a stock including financial metrics, valuation ratios, and company details",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_historical_prices",
                description="Get historical stock price data with customizable time periods and intervals",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        },
                        "period": {
                            "type": "string",
                            "enum": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                            "description": "Time period for historical data",
                            "default": "1y"
                        },
                        "interval": {
                            "type": "string",
                            "enum": ["1m", "2m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
                            "description": "Data interval",
                            "default": "1d"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_financial_statements",
                description="Get financial statements (balance sheet, income statement, or cash flow) for a company",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        },
                        "statement_type": {
                            "type": "string",
                            "enum": ["balance_sheet", "income_statement", "cash_flow"],
                            "description": "Type of financial statement to retrieve"
                        },
                        "period": {
                            "type": "string",
                            "enum": ["quarterly", "annual"],
                            "description": "Reporting period",
                            "default": "quarterly"
                        }
                    },
                    "required": ["symbol", "statement_type"]
                }
            ),
            Tool(
                name="get_stock_news",
                description="Get the latest news articles about a specific stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of news articles to retrieve",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_analyst_recommendations",
                description="Get analyst recommendations and ratings for a stock over time",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_institutional_holders",
                description="Get institutional holders and their positions in a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_insider_transactions",
                description="Get recent insider trading transactions for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_options_expirations",
                description="Get available options expiration dates for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_options_chain",
                description="Get the complete options chain for a specific expiration date",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        },
                        "expiration": {
                            "type": "string",
                            "description": "Options expiration date in YYYY-MM-DD format"
                        },
                        "option_type": {
                            "type": "string",
                            "enum": ["calls", "puts", "both"],
                            "description": "Type of options to retrieve",
                            "default": "both"
                        }
                    },
                    "required": ["symbol", "expiration"]
                }
            ),
            Tool(
                name="get_dividend_history",
                description="Get historical dividend payments for a stock",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_stock_splits",
                description="Get historical stock splits for a company",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT, GOOGL)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="compare_stocks",
                description="Compare key financial metrics between multiple stocks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of stock symbols to compare (e.g., ['AAPL', 'MSFT', 'GOOGL'])",
                            "minItems": 2,
                            "maxItems": 10
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific metrics to compare (optional). If not provided, will compare all key metrics",
                            "default": []
                        }
                    },
                    "required": ["symbols"]
                }
            )
        ]
    
    async def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle a tool call and return the results."""
        try:
            if name == "get_stock_info":
                result = self.client.get_stock_info(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Stock Information for {result.symbol}:\n\n{self._format_stock_info(result)}"
                )]
            
            elif name == "get_historical_prices":
                period = TimePeriod(arguments.get("period", "1y"))
                interval = Interval(arguments.get("interval", "1d"))
                result = self.client.get_historical_prices(
                    arguments["symbol"], period, interval
                )
                return [TextContent(
                    type="text",
                    text=f"Historical Prices for {arguments['symbol']}:\n\n{self._format_price_data(result)}"
                )]
            
            elif name == "get_financial_statements":
                statement_type = StatementType(arguments["statement_type"])
                period = Period(arguments.get("period", "quarterly"))
                result = self.client.get_financial_statements(
                    arguments["symbol"], statement_type, period
                )
                return [TextContent(
                    type="text",
                    text=f"Financial Statements for {arguments['symbol']}:\n\n{self._format_financial_statements(result)}"
                )]
            
            elif name == "get_stock_news":
                limit = arguments.get("limit", 10)
                result = self.client.get_stock_news(arguments["symbol"], limit)
                return [TextContent(
                    type="text",
                    text=f"Latest News for {arguments['symbol']}:\n\n{self._format_news(result)}"
                )]
            
            elif name == "get_analyst_recommendations":
                result = self.client.get_analyst_recommendations(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Analyst Recommendations for {arguments['symbol']}:\n\n{self._format_analyst_recommendations(result)}"
                )]
            
            elif name == "get_institutional_holders":
                result = self.client.get_institutional_holders(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Institutional Holders for {arguments['symbol']}:\n\n{self._format_institutional_holders(result)}"
                )]
            
            elif name == "get_insider_transactions":
                result = self.client.get_insider_transactions(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Insider Transactions for {arguments['symbol']}:\n\n{self._format_insider_transactions(result)}"
                )]
            
            elif name == "get_options_expirations":
                result = self.client.get_options_expirations(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Options Expiration Dates for {arguments['symbol']}:\n\n{', '.join(result)}"
                )]
            
            elif name == "get_options_chain":
                option_type = OptionType(arguments.get("option_type", "both"))
                result = self.client.get_options_chain(
                    arguments["symbol"], arguments["expiration"], option_type
                )
                return [TextContent(
                    type="text",
                    text=f"Options Chain for {arguments['symbol']} ({arguments['expiration']}):\n\n{self._format_options_chain(result)}"
                )]
            
            elif name == "get_dividend_history":
                result = self.client.get_dividend_history(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Dividend History for {arguments['symbol']}:\n\n{self._format_dividend_history(result)}"
                )]
            
            elif name == "get_stock_splits":
                result = self.client.get_stock_splits(arguments["symbol"])
                return [TextContent(
                    type="text",
                    text=f"Stock Splits for {arguments['symbol']}:\n\n{self._format_stock_splits(result)}"
                )]
            
            elif name == "compare_stocks":
                result = self._compare_stocks(arguments["symbols"], arguments.get("metrics", []))
                return [TextContent(
                    type="text",
                    text=f"Stock Comparison:\n\n{result}"
                )]
            
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]
        
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error executing {name}: {str(e)}"
            )]
    
    def _format_stock_info(self, info) -> str:
        """Format stock information for display."""
        lines = []
        lines.append(f"Company: {info.name}")
        lines.append(f"Symbol: {info.symbol}")
        if info.sector:
            lines.append(f"Sector: {info.sector}")
        if info.industry:
            lines.append(f"Industry: {info.industry}")
        
        lines.append("\nValuation Metrics:")
        if info.current_price:
            lines.append(f"  Current Price: ${info.current_price:.2f}")
        if info.market_cap:
            lines.append(f"  Market Cap: ${info.market_cap:,.0f}")
        if info.enterprise_value:
            lines.append(f"  Enterprise Value: ${info.enterprise_value:,.0f}")
        if info.trailing_pe:
            lines.append(f"  Trailing P/E: {info.trailing_pe:.2f}")
        if info.forward_pe:
            lines.append(f"  Forward P/E: {info.forward_pe:.2f}")
        if info.price_to_book:
            lines.append(f"  Price to Book: {info.price_to_book:.2f}")
        if info.dividend_yield:
            lines.append(f"  Dividend Yield: {info.dividend_yield:.2%}")
        
        lines.append("\nTrading Metrics:")
        if info.volume:
            lines.append(f"  Volume: {info.volume:,}")
        if info.avg_volume:
            lines.append(f"  Average Volume: {info.avg_volume:,}")
        if info.beta:
            lines.append(f"  Beta: {info.beta:.2f}")
        
        if info.business_summary:
            lines.append(f"\nBusiness Summary:\n{info.business_summary[:500]}...")
        
        return "\n".join(lines)
    
    def _format_price_data(self, prices) -> str:
        """Format price data for display."""
        if not prices:
            return "No price data available."
        
        lines = []
        lines.append("Date\t\tOpen\tHigh\tLow\tClose\tVolume")
        lines.append("-" * 60)
        
        for price in prices[-20:]:  # Show last 20 data points
            date_str = price.date.strftime("%Y-%m-%d")
            volume_str = f"{price.volume:,}" if price.volume else "N/A"
            lines.append(
                f"{date_str}\t{price.open:.2f}\t{price.high:.2f}\t{price.low:.2f}\t{price.close:.2f}\t{volume_str}"
            )
        
        if len(prices) > 20:
            lines.append(f"\n... and {len(prices) - 20} more records")
        
        return "\n".join(lines)
    
    def _format_financial_statements(self, statements) -> str:
        """Format financial statements for display."""
        if not statements:
            return "No financial statement data available."
        
        lines = []
        for statement in statements[:4]:  # Show last 4 periods
            lines.append(f"\n{statement.statement_type.value.title()} - {statement.date.strftime('%Y-%m-%d')}:")
            lines.append("-" * 50)
            
            # Show top financial items
            for key, value in list(statement.data.items())[:15]:
                if value is not None:
                    if isinstance(value, (int, float)) and abs(value) > 1000:
                        lines.append(f"  {key}: ${value:,.0f}")
                    else:
                        lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)
    
    def _format_news(self, news_items) -> str:
        """Format news items for display."""
        if not news_items:
            return "No news available."
        
        lines = []
        for item in news_items:
            lines.append(f"Title: {item.title}")
            lines.append(f"Publisher: {item.publisher}")
            lines.append(f"Published: {item.publish_time.strftime('%Y-%m-%d %H:%M')}")
            if item.summary:
                lines.append(f"Summary: {item.summary[:200]}...")
            lines.append(f"Link: {item.link}")
            lines.append("-" * 80)
        
        return "\n".join(lines)
    
    def _format_analyst_recommendations(self, recommendations) -> str:
        """Format analyst recommendations for display."""
        if not recommendations:
            return "No analyst recommendations available."
        
        lines = []
        lines.append("Date\t\tStrong Buy\tBuy\tHold\tSell\tStrong Sell\tMean Rating")
        lines.append("-" * 70)
        
        for rec in recommendations[-10:]:  # Show last 10 periods
            lines.append(
                f"{rec.period}\t{rec.strong_buy}\t\t{rec.buy}\t{rec.hold}\t{rec.sell}\t{rec.strong_sell}\t\t{rec.recommendation_mean:.2f}"
            )
        
        return "\n".join(lines)
    
    def _format_institutional_holders(self, holders) -> str:
        """Format institutional holders for display."""
        if not holders:
            return "No institutional holder data available."
        
        lines = []
        lines.append("Holder\t\t\t\tShares\t\t% Out\tValue")
        lines.append("-" * 60)
        
        for holder in holders[:20]:  # Show top 20 holders
            holder_name = holder.holder[:30] + "..." if len(holder.holder) > 30 else holder.holder
            lines.append(
                f"{holder_name:<30}\t{holder.shares:,}\t{holder.percent_out:.2f}%\t${holder.value:,}"
            )
        
        return "\n".join(lines)
    
    def _format_insider_transactions(self, transactions) -> str:
        """Format insider transactions for display."""
        if not transactions:
            return "No insider transaction data available."
        
        lines = []
        lines.append("Insider\t\t\tRelation\tDate\t\tTransaction\tShares")
        lines.append("-" * 70)
        
        for txn in transactions[:15]:  # Show last 15 transactions
            insider_name = txn.insider[:20] + "..." if len(txn.insider) > 20 else txn.insider
            shares = f"{txn.shares_transacted:,}" if txn.shares_transacted else "N/A"
            lines.append(
                f"{insider_name:<20}\t{txn.relation[:10]}\t{txn.last_date.strftime('%Y-%m-%d')}\t{txn.transaction[:10]}\t{shares}"
            )
        
        return "\n".join(lines)
    
    def _format_options_chain(self, options_chain) -> str:
        """Format options chain for display."""
        lines = []
        
        if options_chain.calls:
            lines.append("CALLS:")
            lines.append("Strike\tLast\tBid\tAsk\tVolume\tOI\tIV")
            lines.append("-" * 50)
            for call in options_chain.calls[:10]:  # Show first 10
                vol = call.volume if call.volume else 0
                oi = call.open_interest if call.open_interest else 0
                lines.append(
                    f"{call.strike:.0f}\t{call.last_price:.2f}\t{call.bid:.2f}\t{call.ask:.2f}\t{vol}\t{oi}\t{call.implied_volatility:.3f}"
                )
        
        if options_chain.puts:
            lines.append("\nPUTS:")
            lines.append("Strike\tLast\tBid\tAsk\tVolume\tOI\tIV")
            lines.append("-" * 50)
            for put in options_chain.puts[:10]:  # Show first 10
                vol = put.volume if put.volume else 0
                oi = put.open_interest if put.open_interest else 0
                lines.append(
                    f"{put.strike:.0f}\t{put.last_price:.2f}\t{put.bid:.2f}\t{put.ask:.2f}\t{vol}\t{oi}\t{put.implied_volatility:.3f}"
                )
        
        return "\n".join(lines)
    
    def _format_dividend_history(self, dividends) -> str:
        """Format dividend history for display."""
        if not dividends:
            return "No dividend history available."
        
        lines = []
        lines.append("Date\t\tDividend")
        lines.append("-" * 30)
        
        for div in dividends[-20:]:  # Show last 20 dividends
            lines.append(f"{div.date.strftime('%Y-%m-%d')}\t${div.dividend:.4f}")
        
        return "\n".join(lines)
    
    def _format_stock_splits(self, splits) -> str:
        """Format stock splits for display."""
        if not splits:
            return "No stock splits available."
        
        lines = []
        lines.append("Date\t\tSplit Ratio")
        lines.append("-" * 30)
        
        for split in splits:
            lines.append(f"{split.date.strftime('%Y-%m-%d')}\t{split.split_ratio}")
        
        return "\n".join(lines)
    
    def _compare_stocks(self, symbols: List[str], metrics: List[str]) -> str:
        """Compare multiple stocks."""
        comparison_data = {}
        
        for symbol in symbols:
            try:
                info = self.client.get_stock_info(symbol)
                comparison_data[symbol] = info
            except Exception as e:
                comparison_data[symbol] = f"Error: {e}"
        
        lines = []
        lines.append("Stock Comparison:")
        lines.append("=" * 50)
        
        # Create comparison table
        metric_fields = [
            ("Symbol", "symbol"),
            ("Current Price", "current_price"),
            ("Market Cap", "market_cap"),
            ("P/E Ratio", "trailing_pe"),
            ("P/B Ratio", "price_to_book"),
            ("Dividend Yield", "dividend_yield"),
            ("Beta", "beta"),
            ("Revenue Growth", "revenue_growth"),
        ]
        
        if metrics:
            # Filter to requested metrics
            metric_fields = [(name, field) for name, field in metric_fields if field in metrics]
        
        for metric_name, field_name in metric_fields:
            line = f"{metric_name:<15}"
            for symbol in symbols:
                if isinstance(comparison_data[symbol], str):
                    line += f"{comparison_data[symbol]:<15}"
                else:
                    value = getattr(comparison_data[symbol], field_name, None)
                    if value is None:
                        line += f"{'N/A':<15}"
                    elif field_name in ["current_price", "dividend_yield"]:
                        line += f"${value:.2f}    " if field_name == "current_price" else f"{value:.2%}     "
                    elif field_name == "market_cap":
                        line += f"${value/1e9:.1f}B    "
                    else:
                        line += f"{value:.2f}       "
            lines.append(line)
        
        return "\n".join(lines) 