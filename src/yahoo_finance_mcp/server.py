"""Yahoo Finance Server

A financial data analysis server inspired by the Model Context Protocol,
providing comprehensive Yahoo Finance data through a clean API.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import yfinance as yf
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESGRiskLevel(str, Enum):
    """ESG risk level enumeration."""
    NEGLIGIBLE = "Negligible risk"
    LOW = "Low risk"
    MEDIUM = "Medium risk"
    HIGH = "High risk"
    SEVERE = "Severe risk"


class ESGData(BaseModel):
    """ESG (Environmental, Social, Governance) data model."""
    total_risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    environmental_risk_score: Optional[float] = None
    environmental_risk_level: Optional[str] = None
    social_risk_score: Optional[float] = None
    social_risk_level: Optional[str] = None
    governance_risk_score: Optional[float] = None
    governance_risk_level: Optional[str] = None
    controversy_level: Optional[int] = None
    controversy_description: Optional[str] = None
    peer_rank: Optional[str] = None
    peer_percentile: Optional[float] = None


class StockInfo(BaseModel):
    """Stock information data model."""
    """Yahoo Finance Server

A financial data analysis server inspired by the Model Context Protocol,
providing comprehensive Yahoo Finance data through a clean API.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import yfinance as yf
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
yfinance_server = FastMCP(
    "yahoo-finance",
    instructions="""
# Yahoo Finance MCP Server

This server provides comprehensive financial data from Yahoo Finance.

Available tools:
- get_stock_info: Get comprehensive stock information including price, market cap, financial metrics
- get_historical_prices: Get historical price data with customizable periods and intervals
- get_financial_statements: Get financial statements (income, balance sheet, cash flow)
- get_stock_news: Get latest news articles for a stock
- get_analyst_recommendations: Get analyst recommendations and recent changes
- get_options_data: Get options data including calls and puts
- compare_stocks: Compare multiple stocks across various metrics
- get_market_summary: Get summary of major market indices
- get_holder_info: Get information about major holders, institutional investors, and insider trades
- get_earnings_info: Get earnings dates, estimates, history, and trends
- get_dividend_info: Get dividend rates, yield, history, and next payment date
- get_balance_sheet_analysis: Get detailed balance sheet analysis with key ratios and trends
- get_cash_flow_analysis: Get comprehensive cash flow analysis and metrics
- get_financial_ratios: Get key financial ratios across multiple categories
"""
)


class StockInfo(BaseModel):
    """Stock information data model."""
    symbol: str
    company_name: str = ""
    current_price: float = 0.0
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    volume: Optional[int] = None
    sector: Optional[str] = None
    industry: Optional[str] = None


class HistoricalPrice(BaseModel):
    """Historical price data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int


class HistoricalData(BaseModel):
    """Historical price data collection."""
    symbol: str
    period: str
    interval: str
    data_points: int
    start_date: str
    end_date: str
    prices: List[HistoricalPrice]


class NewsArticle(BaseModel):
    """News article data model."""
    title: str
    publisher: str
    link: str
    published_at: str
    summary: str = ""


class FinancialStatement(BaseModel):
    """Financial statement data."""
    symbol: str
    period: str
    statement_type: str
    data: Dict[str, Dict[str, Any]]


class OptionsContract(BaseModel):
    """Options contract data."""
    strike: float
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float


class OptionsData(BaseModel):
    """Options data collection."""
    symbol: str
    expiration: str
    available_expirations: List[str]
    calls: List[OptionsContract]
    puts: List[OptionsContract]


class MarketIndex(BaseModel):
    """Market index data."""
    name: str
    current_price: float
    change: float
    change_percent: float
    volume: int


class MarketSummary(BaseModel):
    """Market summary data."""
    summary_date: str
    indices: Dict[str, MarketIndex]


class AnalystRecommendation(BaseModel):
    """Analyst recommendation data."""
    symbol: str
    current_recommendations: Dict[str, Any]
    recent_changes: List[Dict[str, Any]]


class Portfolio(BaseModel):
    """Portfolio data."""
    total_value: float
    cash: float
    stocks: Dict[str, Dict[str, Any]]
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]


class EconomicIndicator(BaseModel):
    """Economic indicator data model."""
    symbol: str
    name: str
    current_value: float
    change: float
    change_percent: float
    previous_close: float
    open: float
    day_high: float
    day_low: float
    volume: int
    description: Optional[str] = None
    category: Optional[str] = None


class MarketIndicators(BaseModel):
    """Market indicators data collection."""
    timestamp: str
    major_indices: Dict[str, EconomicIndicator]
    treasury_yields: Dict[str, EconomicIndicator]
    commodities: Dict[str, EconomicIndicator]
    forex: Dict[str, EconomicIndicator]
    volatility: Dict[str, EconomicIndicator]


class EarningsInfo(BaseModel):
    """Earnings information data model."""
    symbol: str
    next_earnings_date: Optional[str] = None
    eps_estimate: Optional[float] = None
    eps_actual: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_actual: Optional[float] = None
    earnings_history: List[Dict[str, Any]] = []
    earnings_trend: List[Dict[str, Any]] = []


class DividendInfo(BaseModel):
    """Dividend information data model."""
    symbol: str
    dividend_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    five_year_avg_dividend_yield: Optional[float] = None
    dividend_history: List[Dict[str, Any]] = []
    next_dividend_date: Optional[str] = None


class BalanceSheetAnalysis(BaseModel):
    """Balance sheet analysis data model."""
    symbol: str
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    working_capital: Optional[float] = None
    asset_turnover: Optional[float] = None
    quarterly_trends: List[Dict[str, Any]] = []


class CashFlowAnalysis(BaseModel):
    """Cash flow analysis data model."""
    symbol: str
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None
    financing_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    cash_flow_coverage: Optional[float] = None
    capital_expenditure: Optional[float] = None
    quarterly_trends: List[Dict[str, Any]] = []


class FinancialRatios(BaseModel):
    """Financial ratios data model."""
    symbol: str
    profitability: Dict[str, Optional[float]] = {}  # ROE, ROA, Profit Margin
    liquidity: Dict[str, Optional[float]] = {}      # Current, Quick, Cash Ratio
    solvency: Dict[str, Optional[float]] = {}       # Debt/Equity, Interest Coverage
    efficiency: Dict[str, Optional[float]] = {}     # Asset Turnover, Inventory Turnover
    valuation: Dict[str, Optional[float]] = {}      # P/E, P/B, EV/EBITDA
    growth: Dict[str, Optional[float]] = {}         # Revenue, EPS Growth


class HolderType(str, Enum):
    """Holder type enumeration."""
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


@yfinance_server.tool(
    name="get_holder_info",
    description="""Get holder information for a given stock.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    holder_type: str
        Type of holder information to retrieve:
        - major_holders: Major shareholders
        - institutional_holders: Institutional investors
        - mutualfund_holders: Mutual fund holders
        - insider_transactions: Recent insider trades
        - insider_purchases: Recent insider purchases
        - insider_roster_holders: Insider roster information
"""
)
async def get_holder_info(symbol: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol."""
    try:
        symbol = validate_symbol(symbol)
        company = yf.Ticker(symbol)
        
        # Verify company exists
        try:
            if company.isin is None:
                return f"Company ticker {symbol} not found."
        except Exception as e:
            logger.error(f"Error getting holder info for {symbol}: {e}")
            return f"Error getting holder info for {symbol}: {e}"

        # Get appropriate holder data based on type
        if holder_type == HolderType.major_holders:
            holders = company.major_holders
            if holders is not None and not holders.empty:
                return holders.reset_index(names="metric").to_json(orient="records")
        elif holder_type == HolderType.institutional_holders:
            holders = company.institutional_holders
            if holders is not None and not holders.empty:
                return holders.to_json(orient="records")
        elif holder_type == HolderType.mutualfund_holders:
            holders = company.mutualfund_holders
            if holders is not None and not holders.empty:
                return holders.to_json(orient="records", date_format="iso")
        elif holder_type == HolderType.insider_transactions:
            holders = company.insider_transactions
            if holders is not None and not holders.empty:
                return holders.to_json(orient="records", date_format="iso")
        elif holder_type == HolderType.insider_purchases:
            holders = company.insider_purchases
            if holders is not None and not holders.empty:
                return holders.to_json(orient="records", date_format="iso")
        elif holder_type == HolderType.insider_roster_holders:
            holders = company.insider_roster_holders
            if holders is not None and not holders.empty:
                return holders.to_json(orient="records", date_format="iso")
        else:
            return f"Invalid holder type {holder_type}. Please use one of: {', '.join(HolderType.__members__.keys())}"
            
        return f"No {holder_type} data available for {symbol}"
        
    except Exception as e:
        logger.error(f"Error getting holder info for {symbol}: {e}")
        return f"Failed to retrieve holder information: {str(e)}"


def validate_symbol(symbol: str) -> str:
    """Validate and clean stock symbol."""
    if not symbol or not symbol.strip():
        raise ValueError("Stock symbol cannot be empty")
    return symbol.strip().upper()


def format_currency(amount: Optional[float]) -> str:
    """Format currency amount."""
    if amount is None:
        return "N/A"
    if abs(amount) >= 1e12:
        return f"${amount/1e12:.2f}T"
    elif abs(amount) >= 1e9:
        return f"${amount/1e9:.2f}B"
    elif abs(amount) >= 1e6:
        return f"${amount/1e6:.2f}M"
    else:
        return f"${amount:.2f}"


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get comprehensive information about a stock or cryptocurrency.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT") or crypto symbol (e.g., "BTC-USD", "ETH-USD")
        For cryptocurrencies, use the format: "BTC-USD", "ETH-USD", "DOGE-USD", etc.
"""
)
async def get_stock_info(symbol: str) -> str:
    """Get comprehensive information about a stock or cryptocurrency."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # First get historical data as it's more reliable for current price
        hist = stock.history(period="1d")
        if hist.empty:
            logger.warning(f"No historical data found for {symbol}")
        else:
            logger.info(f"Historical data for {symbol}: Open={hist['Open'].iloc[-1]}, High={hist['High'].iloc[-1]}, Low={hist['Low'].iloc[-1]}, Close={hist['Close'].iloc[-1]}")
        
        # Get basic info
        info = stock.info
        if not info or len(info) <= 1:
            return f"No data found for symbol: {symbol}"
        
        # Log available price fields
        price_fields = {k: v for k, v in info.items() if isinstance(v, (int, float)) and 'price' in k.lower()}
        logger.info(f"Available price fields for {symbol}: {json.dumps(price_fields, indent=2)}")
        
        # Check if it's a cryptocurrency
        is_crypto = "-USD" in symbol
        
        # Get current price from historical data first
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        
        # If no historical price, try info fields
        if current_price == 0.0:
            price_candidates = [
                ("regularMarketPrice", info.get("regularMarketPrice")),
                ("currentPrice", info.get("currentPrice")),
                ("previousClose", info.get("previousClose")),
                ("open", info.get("open")),
                ("ask", info.get("ask")),
                ("bid", info.get("bid"))
            ]
            
            for field_name, value in price_candidates:
                if value and value > 0:
                    current_price = float(value)
                    logger.info(f"Using {field_name} for {symbol} price: {current_price}")
                    break
        
        # Get market cap and volume
        market_cap = info.get("marketCap", 0)
        volume = info.get("volume24Hr" if is_crypto else "volume", 0)
        
        # If market cap is 0 for crypto, try to calculate it
        if is_crypto and market_cap == 0:
            circulating_supply = info.get("circulatingSupply", 0)
            if circulating_supply > 0 and current_price > 0:
                market_cap = circulating_supply * current_price
                logger.info(f"Calculated market cap for {symbol}: {market_cap}")
        
        # Get 52-week data
        week_52_high = info.get("fiftyTwoWeekHigh", 0)
        week_52_low = info.get("fiftyTwoWeekLow", 0)
        
        # If we don't have 52-week data, try to get it from historical data
        if (week_52_high == 0 or week_52_low == 0) and not hist.empty:
            yearly_hist = stock.history(period="1y")
            if not yearly_hist.empty:
                week_52_high = float(yearly_hist["High"].max())
                week_52_low = float(yearly_hist["Low"].min())
                logger.info(f"Got 52-week data from history for {symbol}: High={week_52_high}, Low={week_52_low}")
        
        stock_info = StockInfo(
            symbol=symbol,
            company_name=info.get("longName", ""),
            current_price=current_price,
            market_cap=market_cap,
            pe_ratio=None if is_crypto else info.get("trailingPE"),
            dividend_yield=None if is_crypto else info.get("dividendYield"),
            fifty_two_week_high=week_52_high,
            fifty_two_week_low=week_52_low,
            volume=volume,
            sector=None if is_crypto else info.get("sector"),
            industry=None if is_crypto else info.get("industry")
        )
        
        return stock_info.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting stock info for {symbol}: {e}")
        return f"Failed to retrieve stock information: {str(e)}"


@yfinance_server.tool(
    name="get_historical_prices",
    description="""Get historical price data for a stock with customizable time periods and intervals.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    period: str
        Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        Default: "1mo"
    interval: str
        Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        Default: "1d"
"""
)
async def get_historical_prices(symbol: str, period: str = "1mo", interval: str = "1d") -> str:
    """Get historical price data for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get historical data
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return f"No historical data found for {symbol}"
        
        # Convert to response format
        prices = []
        for date, row in hist.iterrows():
            prices.append(HistoricalPrice(
                date=date.strftime("%Y-%m-%d"),
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"]) if pd.notna(row["Volume"]) else 0
            ))
        
        historical_data = HistoricalData(
            symbol=symbol,
            period=period,
            interval=interval,
            data_points=len(prices),
            start_date=hist.index[0].strftime("%Y-%m-%d"),
            end_date=hist.index[-1].strftime("%Y-%m-%d"),
            prices=prices
        )
        
        return historical_data.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting historical prices for {symbol}: {e}")
        return f"Failed to retrieve historical data: {str(e)}"


@yfinance_server.tool(
    name="get_financial_statements",
    description="""Get financial statements for a company including income statement, balance sheet, and cash flow.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    statement_type: str
        Type of statement: "income_statement", "balance_sheet", or "cash_flow"
        Default: "income_statement"
    period: str
        Period type: "annual" or "quarterly"
        Default: "annual"
"""
)
async def get_financial_statements(symbol: str, statement_type: str = "income_statement", period: str = "annual") -> str:
    """Get financial statements for a company."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get the appropriate financial statement
        if statement_type == "income_statement":
            if period == "annual":
                stmt = stock.financials
            else:
                stmt = stock.quarterly_financials
        elif statement_type == "balance_sheet":
            if period == "annual":
                stmt = stock.balance_sheet
            else:
                stmt = stock.quarterly_balance_sheet
        elif statement_type == "cash_flow":
            if period == "annual":
                stmt = stock.cashflow
            else:
                stmt = stock.quarterly_cashflow
        else:
            return f"Invalid statement type: {statement_type}. Use: income_statement, balance_sheet, or cash_flow"
        
        if stmt.empty:
            return f"No {statement_type} data found for {symbol}"
        
        # Convert to dictionary
        data = {}
        for col in stmt.columns:
            year = col.strftime("%Y") if hasattr(col, 'strftime') else str(col)
            data[year] = {}
            for idx in stmt.index:
                value = stmt.loc[idx, col]
                if pd.notna(value):
                    data[year][str(idx)] = float(value) if isinstance(value, (int, float)) else str(value)
        
        financial_statement = FinancialStatement(
            symbol=symbol,
            period=period,
            statement_type=statement_type,
            data=data
        )
        
        return financial_statement.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting financial statements for {symbol}: {e}")
        return f"Failed to retrieve financial data: {str(e)}"


@yfinance_server.tool(
    name="get_stock_news",
    description="""Get latest news articles for a stock.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    max_articles: int
        Maximum number of articles to return
        Default: 10
"""
)
async def get_stock_news(symbol: str, max_articles: int = 10) -> str:
    """Get latest news articles for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        news = stock.news
        
        if not news:
            return f"No news found for {symbol}"
        
        articles = []
        for article in news[:max_articles]:
            try:
                published_time = datetime.fromtimestamp(article.get("providerPublishTime", 0))
                articles.append(NewsArticle(
                    title=article.get("title", ""),
                    publisher=article.get("publisher", ""),
                    link=article.get("link", ""),
                    published_at=published_time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary=article.get("summary", "")
                ))
            except Exception as article_error:
                logger.warning(f"Error processing article: {article_error}")
                continue
        
        return json.dumps([article.model_dump() for article in articles], indent=2)
    except Exception as e:
        logger.error(f"Error getting news for {symbol}: {e}")
        return f"Failed to retrieve news: {str(e)}"


@yfinance_server.tool(
    name="get_analyst_recommendations",
    description="""Get analyst recommendations and recent changes for a stock.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
"""
)
async def get_analyst_recommendations(symbol: str) -> str:
    """Get analyst recommendations for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get recommendations
        recommendations = stock.recommendations
        upgrades_downgrades = stock.upgrades_downgrades
        
        current_recommendations = {}
        recent_changes = []
        
        # Process current recommendations
        if recommendations is not None and not recommendations.empty:
            latest = recommendations.iloc[-1]
            current_recommendations = {
                "period": latest.name.strftime("%Y-%m") if hasattr(latest.name, 'strftime') else str(latest.name),
                "strong_buy": int(latest.get("strongBuy", 0)),
                "buy": int(latest.get("buy", 0)),
                "hold": int(latest.get("hold", 0)),
                "sell": int(latest.get("sell", 0)),
                "strong_sell": int(latest.get("strongSell", 0))
            }
        
        # Process upgrades/downgrades
        if upgrades_downgrades is not None and not upgrades_downgrades.empty:
            for date, row in upgrades_downgrades.head(10).iterrows():
                recent_changes.append({
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "firm": str(row.get("Firm", "")),
                    "to_grade": str(row.get("ToGrade", "")),
                    "from_grade": str(row.get("FromGrade", "")),
                    "action": str(row.get("Action", ""))
                })
        
        analyst_rec = AnalystRecommendation(
            symbol=symbol,
            current_recommendations=current_recommendations,
            recent_changes=recent_changes
        )
        
        return analyst_rec.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting analyst recommendations for {symbol}: {e}")
        return f"Failed to retrieve analyst data: {str(e)}"


@yfinance_server.tool(
    name="get_options_data",
    description="""Get options data including calls and puts for a stock.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    expiration_date: str
        Specific expiration date (YYYY-MM-DD format, optional)
        If not provided, uses the nearest expiration date
"""
)
async def get_options_data(symbol: str, expiration_date: Optional[str] = None) -> str:
    """Get options data for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            return f"No options data available for {symbol}"
        
        # Use provided date or nearest expiration
        if expiration_date:
            if expiration_date not in expirations:
                return f"Expiration date {expiration_date} not available. Available: {list(expirations)}"
            exp_date = expiration_date
        else:
            exp_date = expirations[0]
        
        # Get options chain
        options_chain = stock.option_chain(exp_date)
        
        calls = []
        puts = []
        
        # Process calls
        if not options_chain.calls.empty:
            for _, row in options_chain.calls.iterrows():
                calls.append(OptionsContract(
                    strike=float(row["strike"]),
                    last_price=float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0.0,
                    bid=float(row["bid"]) if pd.notna(row["bid"]) else 0.0,
                    ask=float(row["ask"]) if pd.notna(row["ask"]) else 0.0,
                    volume=int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    open_interest=int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0,
                    implied_volatility=float(row["impliedVolatility"]) if pd.notna(row["impliedVolatility"]) else 0.0
                ))
        
        # Process puts
        if not options_chain.puts.empty:
            for _, row in options_chain.puts.iterrows():
                puts.append(OptionsContract(
                    strike=float(row["strike"]),
                    last_price=float(row["lastPrice"]) if pd.notna(row["lastPrice"]) else 0.0,
                    bid=float(row["bid"]) if pd.notna(row["bid"]) else 0.0,
                    ask=float(row["ask"]) if pd.notna(row["ask"]) else 0.0,
                    volume=int(row["volume"]) if pd.notna(row["volume"]) else 0,
                    open_interest=int(row["openInterest"]) if pd.notna(row["openInterest"]) else 0,
                    implied_volatility=float(row["impliedVolatility"]) if pd.notna(row["impliedVolatility"]) else 0.0
                ))
        
        options_data = OptionsData(
            symbol=symbol,
            expiration=exp_date,
            available_expirations=list(expirations),
            calls=calls,
            puts=puts
        )
        
        return options_data.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting options data for {symbol}: {e}")
        return f"Failed to retrieve options data: {str(e)}"


@yfinance_server.tool(
    name="compare_stocks",
    description="""Compare multiple stocks across various financial metrics.
    
Args:
    symbols: List[str]
        List of stock ticker symbols to compare (e.g., ["AAPL", "MSFT", "GOOGL"])
    metrics: List[str]
        Specific metrics to compare (optional)
        Default metrics: currentPrice, marketCap, trailingPE, dividendYield, 
                        fiftyTwoWeekHigh, fiftyTwoWeekLow, volume
"""
)
async def compare_stocks(symbols: List[str], metrics: Optional[List[str]] = None) -> str:
    """Compare multiple stocks across various metrics."""
    try:
        if not symbols:
            return "At least one symbol must be provided"
        
        if metrics is None:
            metrics = ["currentPrice", "marketCap", "trailingPE", "dividendYield", 
                      "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "volume"]
        
        comparison = {
            "symbols": [],
            "comparison_date": datetime.now().strftime("%Y-%m-%d"),
            "metrics": {}
        }
        
        stock_data = {}
        for symbol in symbols:
            try:
                symbol = validate_symbol(symbol)
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if info and len(info) > 1:
                    stock_data[symbol] = info
                    comparison["symbols"].append(symbol)
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue
        
        if not stock_data:
            return "No valid stock data found for any symbols"
        
        # Build comparison metrics
        for metric in metrics:
            comparison["metrics"][metric] = {}
            for symbol in comparison["symbols"]:
                value = stock_data[symbol].get(metric)
                if value is not None:
                    comparison["metrics"][metric][symbol] = value
                else:
                    comparison["metrics"][metric][symbol] = "N/A"
        
        return json.dumps(comparison, indent=2)
    except Exception as e:
        logger.error(f"Error comparing stocks: {e}")
        return f"Failed to compare stocks: {str(e)}"


@yfinance_server.tool(
    name="get_market_summary",
    description="""Get a quick summary of major market indices and indicators.
    
Returns current values and changes for:
- S&P 500
- Dow Jones
- NASDAQ
- VIX
- 10Y Treasury Yield
"""
)
async def get_market_summary() -> str:
    """Get a summary of major market indices."""
    try:
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^VIX": "VIX",
            "^TNX": "10Y Treasury"
        }
        
        summary_data = {}
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if info:
                    current = info.get("regularMarketPrice", 0.0)
                    previous = info.get("regularMarketPreviousClose", 0.0)
                    change = current - previous
                    change_percent = (change / previous * 100) if previous != 0 else 0
                    
                    summary_data[symbol] = MarketIndex(
                        name=name,
                        current_price=round(float(current), 2),
                        change=round(float(change), 2),
                        change_percent=round(float(change_percent), 2),
                        volume=info.get("regularMarketVolume", 0)
                    )
                    logger.info(f"Got market data for {name}: {current} ({change_percent:+.2f}%)")
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue
        
        market_summary = MarketSummary(
            summary_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            indices=summary_data
        )
        
        return market_summary.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        return f"Failed to retrieve market summary: {str(e)}"


@yfinance_server.tool(
    name="get_earnings_info",
    description="""Get earnings information for a stock including dates, estimates, and history.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
"""
)
async def get_earnings_info(symbol: str) -> str:
    """Get earnings information for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get earnings data
        calendar = stock.calendar
        earnings = stock.earnings
        earnings_dates = stock.earnings_dates
        earnings_trend = stock.earnings_trend
        
        # Initialize earnings info
        earnings_info = {
            "symbol": symbol,
            "next_earnings_date": None,
            "eps_estimate": None,
            "eps_actual": None,
            "revenue_estimate": None,
            "revenue_actual": None,
            "earnings_history": [],
            "earnings_trend": []
        }
        
        # Process calendar data
        if calendar is not None and not calendar.empty:
            try:
                earnings_info["next_earnings_date"] = calendar.index[0].strftime("%Y-%m-%d") if hasattr(calendar.index[0], 'strftime') else str(calendar.index[0])
                earnings_info["eps_estimate"] = float(calendar.iloc[0].get("EPS Estimate", 0)) if pd.notna(calendar.iloc[0].get("EPS Estimate")) else None
                earnings_info["revenue_estimate"] = float(calendar.iloc[0].get("Revenue Estimate", 0)) if pd.notna(calendar.iloc[0].get("Revenue Estimate")) else None
            except Exception as e:
                logger.warning(f"Error processing calendar data for {symbol}: {e}")
        
        # Process earnings history
        if earnings_dates is not None and not earnings_dates.empty:
            for date, row in earnings_dates.iterrows():
                try:
                    history_entry = {
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                        "eps_estimate": float(row.get("EPS Estimate", 0)) if pd.notna(row.get("EPS Estimate")) else None,
                        "eps_actual": float(row.get("Reported EPS", 0)) if pd.notna(row.get("Reported EPS")) else None,
                        "surprise": float(row.get("Surprise(%)", 0)) if pd.notna(row.get("Surprise(%)")) else None
                    }
                    earnings_info["earnings_history"].append(history_entry)
                except Exception as e:
                    logger.warning(f"Error processing earnings history entry for {symbol}: {e}")
        
        # Process earnings trend
        if earnings_trend is not None and not earnings_trend.empty:
            for date, row in earnings_trend.iterrows():
                try:
                    trend_entry = {
                        "period": str(date),
                        "eps_estimate": float(row.get("EPS Estimate", 0)) if pd.notna(row.get("EPS Estimate")) else None,
                        "eps_trend": float(row.get("EPS Trend", 0)) if pd.notna(row.get("EPS Trend")) else None,
                        "eps_revisions": float(row.get("EPS Revisions", 0)) if pd.notna(row.get("EPS Revisions")) else None
                    }
                    earnings_info["earnings_trend"].append(trend_entry)
                except Exception as e:
                    logger.warning(f"Error processing earnings trend entry for {symbol}: {e}")
        
        return json.dumps(earnings_info, indent=2)
    except Exception as e:
        logger.error(f"Error getting earnings info for {symbol}: {e}")
        return f"Failed to retrieve earnings information: {str(e)}"


@yfinance_server.tool(
    name="get_dividend_info",
    description="""Get dividend information for a stock including history and yield.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
"""
)
async def get_dividend_info(symbol: str) -> str:
    """Get dividend information for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        dividends = stock.dividends
        
        # Initialize dividend info
        dividend_info = {
            "symbol": symbol,
            "dividend_rate": info.get("dividendRate"),
            "dividend_yield": info.get("dividendYield"),
            "payout_ratio": info.get("payoutRatio"),
            "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
            "dividend_history": [],
            "next_dividend_date": None
        }
        
        # Get next dividend date if available
        if info.get("dividendDate"):
            try:
                div_date = datetime.fromtimestamp(info["dividendDate"])
                dividend_info["next_dividend_date"] = div_date.strftime("%Y-%m-%d")
            except Exception as e:
                logger.warning(f"Error processing dividend date for {symbol}: {e}")
        
        # Process dividend history
        if dividends is not None and not dividends.empty:
            for date, amount in dividends.items():
                try:
                    history_entry = {
                        "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                        "amount": float(amount) if pd.notna(amount) else None
                    }
                    dividend_info["dividend_history"].append(history_entry)
                except Exception as e:
                    logger.warning(f"Error processing dividend history entry for {symbol}: {e}")
        
        return json.dumps(dividend_info, indent=2)
    except Exception as e:
        logger.error(f"Error getting dividend info for {symbol}: {e}")
        return f"Failed to retrieve dividend information: {str(e)}"


@yfinance_server.tool(
    name="get_balance_sheet_analysis",
    description="""Get detailed balance sheet analysis including key ratios and trends.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    period: str
        Period type: "annual" or "quarterly"
        Default: "annual"
"""
)
async def get_balance_sheet_analysis(symbol: str, period: str = "annual") -> str:
    """Get balance sheet analysis for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get balance sheet data
        balance_sheet = stock.balance_sheet if period == "annual" else stock.quarterly_balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            return f"No balance sheet data available for {symbol}"
            
        # Initialize analysis
        analysis = {
            "symbol": symbol,
            "total_assets": None,
            "total_liabilities": None,
            "total_equity": None,
            "current_ratio": None,
            "quick_ratio": None,
            "debt_to_equity": None,
            "working_capital": None,
            "asset_turnover": None,
            "quarterly_trends": []
        }
        
        # Get latest values
        latest = balance_sheet.iloc[:, 0]
        
        # Calculate key metrics
        try:
            total_assets = float(latest.get("Total Assets", 0))
            total_liabilities = float(latest.get("Total Liabilities Net Minority Interest", 0))
            current_assets = float(latest.get("Current Assets", 0))
            current_liabilities = float(latest.get("Current Liabilities", 0))
            inventory = float(latest.get("Inventory", 0))
            
            analysis["total_assets"] = total_assets
            analysis["total_liabilities"] = total_liabilities
            analysis["total_equity"] = total_assets - total_liabilities
            
            if current_liabilities != 0:
                analysis["current_ratio"] = current_assets / current_liabilities
                analysis["quick_ratio"] = (current_assets - inventory) / current_liabilities
            
            if total_liabilities != 0 and (total_assets - total_liabilities) != 0:
                analysis["debt_to_equity"] = total_liabilities / (total_assets - total_liabilities)
            
            analysis["working_capital"] = current_assets - current_liabilities
            
            if total_assets != 0:
                revenue = float(stock.financials.loc["Total Revenue", :].iloc[0]) if not stock.financials.empty else 0
                analysis["asset_turnover"] = revenue / total_assets
        except Exception as e:
            logger.warning(f"Error calculating balance sheet metrics for {symbol}: {e}")
        
        # Calculate quarterly trends
        for date, values in balance_sheet.items():
            try:
                trend = {
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "total_assets": float(values.get("Total Assets", 0)) if pd.notna(values.get("Total Assets")) else None,
                    "total_liabilities": float(values.get("Total Liabilities Net Minority Interest", 0)) if pd.notna(values.get("Total Liabilities Net Minority Interest")) else None,
                    "current_ratio": float(values.get("Current Assets", 0)) / float(values.get("Current Liabilities", 1)) if pd.notna(values.get("Current Assets")) and pd.notna(values.get("Current Liabilities")) and float(values.get("Current Liabilities", 0)) != 0 else None
                }
                analysis["quarterly_trends"].append(trend)
            except Exception as e:
                logger.warning(f"Error processing quarterly trend for {symbol}: {e}")
        
        return json.dumps(analysis, indent=2)
    except Exception as e:
        logger.error(f"Error getting balance sheet analysis for {symbol}: {e}")
        return f"Failed to retrieve balance sheet analysis: {str(e)}"


@yfinance_server.tool(
    name="get_cash_flow_analysis",
    description="""Get detailed cash flow analysis including key metrics and trends.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    period: str
        Period type: "annual" or "quarterly"
        Default: "annual"
"""
)
async def get_cash_flow_analysis(symbol: str, period: str = "annual") -> str:
    """Get cash flow analysis for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get cash flow data
        cash_flow = stock.cashflow if period == "annual" else stock.quarterly_cashflow
        if cash_flow is None or cash_flow.empty:
            return f"No cash flow data available for {symbol}"
            
        # Initialize analysis
        analysis = {
            "symbol": symbol,
            "operating_cash_flow": None,
            "investing_cash_flow": None,
            "financing_cash_flow": None,
            "free_cash_flow": None,
            "cash_flow_coverage": None,
            "capital_expenditure": None,
            "quarterly_trends": []
        }
        
        # Get latest values
        latest = cash_flow.iloc[:, 0]
        
        # Calculate key metrics
        try:
            operating_cf = float(latest.get("Operating Cash Flow", 0))
            investing_cf = float(latest.get("Investing Cash Flow", 0))
            financing_cf = float(latest.get("Financing Cash Flow", 0))
            capex = float(latest.get("Capital Expenditure", 0))
            
            analysis["operating_cash_flow"] = operating_cf
            analysis["investing_cash_flow"] = investing_cf
            analysis["financing_cash_flow"] = financing_cf
            analysis["capital_expenditure"] = capex
            analysis["free_cash_flow"] = operating_cf - abs(capex)
            
            if operating_cf != 0:
                total_debt = float(stock.balance_sheet.iloc[stock.balance_sheet.index.get_loc("Total Debt"), 0]) if "Total Debt" in stock.balance_sheet.index else 0
                analysis["cash_flow_coverage"] = operating_cf / total_debt if total_debt != 0 else None
        except Exception as e:
            logger.warning(f"Error calculating cash flow metrics for {symbol}: {e}")
        
        # Calculate quarterly trends
        for date, values in cash_flow.items():
            try:
                trend = {
                    "date": date.strftime("%Y-%m-%d") if hasattr(date, 'strftime') else str(date),
                    "operating_cash_flow": float(values.get("Operating Cash Flow", 0)) if pd.notna(values.get("Operating Cash Flow")) else None,
                    "free_cash_flow": float(values.get("Operating Cash Flow", 0)) - abs(float(values.get("Capital Expenditure", 0))) if pd.notna(values.get("Operating Cash Flow")) and pd.notna(values.get("Capital Expenditure")) else None,
                    "cash_flow_margin": float(values.get("Operating Cash Flow", 0)) / float(values.get("Total Revenue", 1)) if pd.notna(values.get("Operating Cash Flow")) and pd.notna(values.get("Total Revenue")) and float(values.get("Total Revenue", 0)) != 0 else None
                }
                analysis["quarterly_trends"].append(trend)
            except Exception as e:
                logger.warning(f"Error processing quarterly trend for {symbol}: {e}")
        
        return json.dumps(analysis, indent=2)
    except Exception as e:
        logger.error(f"Error getting cash flow analysis for {symbol}: {e}")
        return f"Failed to retrieve cash flow analysis: {str(e)}"


@yfinance_server.tool(
    name="get_financial_ratios",
    description="""Get comprehensive financial ratios including profitability, liquidity, solvency, and valuation metrics.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
"""
)
async def get_financial_ratios(symbol: str) -> str:
    """Get financial ratios for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Initialize ratios
        ratios = {
            "symbol": symbol,
            "profitability": {},
            "liquidity": {},
            "solvency": {},
            "efficiency": {},
            "valuation": {},
            "growth": {}
        }
        
        # Profitability ratios
        try:
            ratios["profitability"] = {
                "return_on_equity": info.get("returnOnEquity"),
                "return_on_assets": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins")
            }
        except Exception as e:
            logger.warning(f"Error calculating profitability ratios for {symbol}: {e}")
        
        # Liquidity ratios
        try:
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                current_assets = float(balance_sheet.iloc[balance_sheet.index.get_loc("Current Assets"), 0])
                current_liabilities = float(balance_sheet.iloc[balance_sheet.index.get_loc("Current Liabilities"), 0])
                inventory = float(balance_sheet.iloc[balance_sheet.index.get_loc("Inventory"), 0])
                cash = float(balance_sheet.iloc[balance_sheet.index.get_loc("Cash And Cash Equivalents"), 0])
                
                ratios["liquidity"] = {
                    "current_ratio": current_assets / current_liabilities if current_liabilities != 0 else None,
                    "quick_ratio": (current_assets - inventory) / current_liabilities if current_liabilities != 0 else None,
                    "cash_ratio": cash / current_liabilities if current_liabilities != 0 else None
                }
        except Exception as e:
            logger.warning(f"Error calculating liquidity ratios for {symbol}: {e}")
        
        # Solvency ratios
        try:
            ratios["solvency"] = {
                "debt_to_equity": info.get("debtToEquity"),
                "interest_coverage": info.get("interestCoverage")
            }
        except Exception as e:
            logger.warning(f"Error calculating solvency ratios for {symbol}: {e}")
        
        # Efficiency ratios
        try:
            ratios["efficiency"] = {
                "asset_turnover": info.get("assetTurnover"),
                "inventory_turnover": info.get("inventoryTurnover"),
                "receivables_turnover": info.get("receivablesTurnover")
            }
        except Exception as e:
            logger.warning(f"Error calculating efficiency ratios for {symbol}: {e}")
        
        # Valuation ratios
        try:
            ratios["valuation"] = {
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                "ev_to_ebitda": info.get("enterpriseToEbitda")
            }
        except Exception as e:
            logger.warning(f"Error calculating valuation ratios for {symbol}: {e}")
        
        # Growth ratios
        try:
            ratios["growth"] = {
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "eps_growth": info.get("earningsQuarterlyGrowth")
            }
        except Exception as e:
            logger.warning(f"Error calculating growth ratios for {symbol}: {e}")
        
        return json.dumps(ratios, indent=2)
    except Exception as e:
        logger.error(f"Error getting financial ratios for {symbol}: {e}")
        return f"Failed to retrieve financial ratios: {str(e)}"


def main():
    """Main entry point for the Yahoo Finance MCP server."""
    logger.info("Starting Yahoo Finance MCP Server...")
    yfinance_server.run(transport="stdio")


if __name__ == "__main__":
    main() 
