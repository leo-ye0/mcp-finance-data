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
    esg: Optional[ESGData] = None


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


class TechnicalIndicator(BaseModel):
    """Technical indicator data."""
    symbol: str
    date: str
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    rsi_14: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_middle: Optional[float] = None
    bollinger_lower: Optional[float] = None
    stoch_k: Optional[float] = None
    stoch_d: Optional[float] = None


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


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators data model."""
    timestamp: str
    symbol: str
    # Trend Indicators
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    # Momentum Indicators
    macd_line: float
    macd_signal: float
    macd_histogram: float
    rsi_14: float
    stoch_k: float
    stoch_d: float
    # Volatility Indicators
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    atr_14: float  # Average True Range
    # Volume Indicators
    obv: float  # On Balance Volume
    mfi_14: float  # Money Flow Index
    # Additional Indicators
    williams_r: float  # Williams %R
    cci_20: float  # Commodity Channel Index


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
"""
)


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


def determine_risk_level(score: Optional[float]) -> Optional[str]:
    """Determine ESG risk level based on score."""
    if score is None:
        return None
    if score < 10:
        return ESGRiskLevel.NEGLIGIBLE
    elif score < 20:
        return ESGRiskLevel.LOW
    elif score < 30:
        return ESGRiskLevel.MEDIUM
    elif score < 40:
        return ESGRiskLevel.HIGH
    else:
        return ESGRiskLevel.SEVERE


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get comprehensive information about a stock or cryptocurrency.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT") or crypto symbol (e.g., "BTC-USD", "ETH-USD")
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
        
        # Log available ESG fields
        esg_fields = {k: v for k, v in info.items() if any(esg_term in k.lower() for esg_term in ['esg', 'environmental', 'social', 'governance'])}
        logger.info(f"Available ESG fields for {symbol}: {json.dumps(esg_fields, indent=2)}")
        
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
        
        # Get ESG data
        esg_data = None
        if not is_crypto:  # ESG data is only relevant for stocks
            total_risk_score = info.get("esgScore")
            env_risk_score = info.get("environmentScore")
            social_risk_score = info.get("socialScore")
            gov_risk_score = info.get("governanceScore")
            
            if any([total_risk_score, env_risk_score, social_risk_score, gov_risk_score]):
                esg_data = ESGData(
                    total_risk_score=total_risk_score,
                    risk_level=determine_risk_level(total_risk_score),
                    environmental_risk_score=env_risk_score,
                    environmental_risk_level=determine_risk_level(env_risk_score),
                    social_risk_score=social_risk_score,
                    social_risk_level=determine_risk_level(social_risk_score),
                    governance_risk_score=gov_risk_score,
                    governance_risk_level=determine_risk_level(gov_risk_score),
                    controversy_level=info.get("controversyLevel"),
                    controversy_description=info.get("controversyDescription"),
                    peer_rank=info.get("esgPerformance"),
                    peer_percentile=info.get("percentile")
                )
                
                logger.info(
                    f"ESG data for {symbol}:\n"
                    f"Total Risk Score: {total_risk_score} ({determine_risk_level(total_risk_score)})\n"
                    f"Environmental Risk: {env_risk_score} ({determine_risk_level(env_risk_score)})\n"
                    f"Social Risk: {social_risk_score} ({determine_risk_level(social_risk_score)})\n"
                    f"Governance Risk: {gov_risk_score} ({determine_risk_level(gov_risk_score)})"
                )
        
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
            industry=None if is_crypto else info.get("industry"),
            esg=esg_data
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
    description="""Get latest news articles for a stock from Yahoo Finance.
    
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


def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate technical indicators from price data."""
    if data.empty:
        return {}
    
    # Calculate SMAs
    sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
    sma_200 = data['Close'].rolling(window=200).mean().iloc[-1]
    
    # Calculate EMAs for MACD
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    middle_band = data['Close'].rolling(window=20).mean()
    std_dev = data['Close'].rolling(window=20).std()
    upper_band = middle_band + (std_dev * 2)
    lower_band = middle_band - (std_dev * 2)
    
    # Calculate Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    k = 100 * ((data['Close'] - low_14) / (high_14 - low_14))
    d = k.rolling(window=3).mean()
    
    return {
        'sma_50': sma_50,
        'sma_200': sma_200,
        'ema_12': ema_12.iloc[-1],
        'ema_26': ema_26.iloc[-1],
        'macd': macd.iloc[-1],
        'macd_signal': macd_signal.iloc[-1],
        'macd_hist': macd_hist.iloc[-1],
        'rsi_14': rsi.iloc[-1],
        'bollinger_upper': upper_band.iloc[-1],
        'bollinger_middle': middle_band.iloc[-1],
        'bollinger_lower': lower_band.iloc[-1],
        'stoch_k': k.iloc[-1],
        'stoch_d': d.iloc[-1]
    }


@yfinance_server.tool(
    name="get_technical_indicators",
    description="""Get technical analysis indicators for a stock.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    interval: str
        Data interval (1d, 5d, 1wk, 1mo, 3mo)
        Default: "1d"
"""
)
async def get_technical_indicators(symbol: str, interval: str = "1d") -> str:
    """Get technical analysis indicators for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get historical data for calculations
        hist = stock.history(period="1y", interval=interval)
        
        if hist.empty:
            return f"No data found for {symbol}"
        
        # Calculate indicators
        indicators = calculate_technical_indicators(hist)
        
        technical_data = TechnicalIndicator(
            symbol=symbol,
            date=hist.index[-1].strftime("%Y-%m-%d"),
            **indicators
        )
        
        return technical_data.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting technical indicators for {symbol}: {e}")
        return f"Failed to retrieve technical indicators: {str(e)}"


@yfinance_server.tool(
    name="analyze_portfolio",
    description="""Analyze a portfolio of stocks.
    
Args:
    symbols: List[str]
        List of stock ticker symbols in the portfolio
    weights: Optional[List[float]]
        Portfolio weights for each stock (must sum to 1)
        If not provided, equal weights will be used
"""
)
async def analyze_portfolio(symbols: List[str], weights: Optional[List[float]] = None) -> str:
    """Analyze a portfolio of stocks."""
    try:
        if not symbols:
            return "At least one symbol must be provided"
        
        # Use equal weights if not provided
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        if len(weights) != len(symbols):
            return "Number of weights must match number of symbols"
        
        if abs(sum(weights) - 1.0) > 0.0001:
            return "Weights must sum to 1"
        
        portfolio_data = {
            'total_value': 0.0,
            'cash': 0.0,
            'stocks': {},
            'performance': {},
            'risk_metrics': {}
        }
        
        returns_data = []
        
        # Get data for each stock
        for symbol, weight in zip(symbols, weights):
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                returns_data.append(returns * weight)
                
                portfolio_data['stocks'][symbol] = {
                    'weight': weight,
                    'current_price': current_price,
                    'shares': weight * 100000 / current_price,  # Assuming $100,000 portfolio
                    'value': weight * 100000
                }
        
        # Calculate portfolio metrics
        if returns_data:
            portfolio_returns = pd.concat(returns_data, axis=1).sum(axis=1)
            portfolio_data['performance'] = {
                'total_return': float(((1 + portfolio_returns).prod() - 1) * 100),
                'annual_return': float(((1 + portfolio_returns).prod() ** (252/len(portfolio_returns)) - 1) * 100),
                'volatility': float(portfolio_returns.std() * np.sqrt(252) * 100),
                'sharpe_ratio': float(portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252))
            }
            
            portfolio_data['risk_metrics'] = {
                'max_drawdown': float(((1 + portfolio_returns).cumprod().div((1 + portfolio_returns).cumprod().cummax()) - 1).min() * 100),
                'var_95': float(portfolio_returns.quantile(0.05) * 100),
                'var_99': float(portfolio_returns.quantile(0.01) * 100),
                'beta': float(portfolio_returns.cov(portfolio_returns) / portfolio_returns.var())
            }
        
        portfolio_data['total_value'] = sum(stock['value'] for stock in portfolio_data['stocks'].values())
        
        portfolio = Portfolio(**portfolio_data)
        return portfolio.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error analyzing portfolio: {e}")
        return f"Failed to analyze portfolio: {str(e)}"


@yfinance_server.tool(
    name="get_economic_indicators",
    description="""Get comprehensive economic indicators including major indices, treasury yields, commodities, forex, and volatility indices.
    
Returns real-time data for:
- Major Indices (S&P 500, Dow Jones, NASDAQ, Russell 2000)
- Treasury Yields (2Y, 5Y, 10Y, 30Y)
- Commodities (Gold, Oil, Silver)
- Forex (EUR/USD, GBP/USD, USD/JPY)
- Volatility (VIX, VXN)
"""
)
async def get_economic_indicators() -> str:
    """Get comprehensive economic indicators."""
    try:
        indicators = {
            "major_indices": {
                "^GSPC": ("S&P 500", "Major US Stock Market Index"),
                "^DJI": ("Dow Jones Industrial Average", "Blue-chip US Stock Index"),
                "^IXIC": ("NASDAQ Composite", "Tech-heavy US Stock Index"),
                "^RUT": ("Russell 2000", "Small-cap US Stock Index")
            },
            "treasury_yields": {
                "^TNX": ("10-Year Treasury Yield", "US 10 Year Treasury Yield"),
                "^IRX": ("13-Week Treasury Bill", "US 13 Week Treasury Bill Yield"),
                "^TYX": ("30-Year Treasury Yield", "US 30 Year Treasury Yield"),
                "^FVX": ("5-Year Treasury Yield", "US 5 Year Treasury Yield")
            },
            "commodities": {
                "GC=F": ("Gold Futures", "Gold Futures Price"),
                "CL=F": ("Crude Oil Futures", "WTI Crude Oil Futures Price"),
                "SI=F": ("Silver Futures", "Silver Futures Price")
            },
            "forex": {
                "EURUSD=X": ("EUR/USD", "Euro to US Dollar Exchange Rate"),
                "GBPUSD=X": ("GBP/USD", "British Pound to US Dollar Exchange Rate"),
                "JPY=X": ("USD/JPY", "US Dollar to Japanese Yen Exchange Rate")
            },
            "volatility": {
                "^VIX": ("VIX", "CBOE Volatility Index"),
                "^VXN": ("VXN", "NASDAQ 100 Volatility Index")
            }
        }
        
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "major_indices": {},
            "treasury_yields": {},
            "commodities": {},
            "forex": {},
            "volatility": {}
        }
        
        for category, symbols in indicators.items():
            for symbol, (name, description) in symbols.items():
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info:
                        current = info.get("regularMarketPrice", 0.0)
                        previous = info.get("regularMarketPreviousClose", 0.0)
                        change = current - previous
                        change_percent = (change / previous * 100) if previous != 0 else 0
                        
                        indicator = EconomicIndicator(
                            symbol=symbol,
                            name=name,
                            current_value=current,
                            change=round(change, 2),
                            change_percent=round(change_percent, 2),
                            previous_close=previous,
                            open=info.get("regularMarketOpen", 0.0),
                            day_high=info.get("regularMarketDayHigh", 0.0),
                            day_low=info.get("regularMarketDayLow", 0.0),
                            volume=info.get("regularMarketVolume", 0),
                            description=description,
                            category=category
                        )
                        
                        result[category][symbol] = indicator
                        logger.info(f"Got {category} data for {name}: {current} ({change_percent:+.2f}%)")
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue
        
        market_indicators = MarketIndicators(
            timestamp=result["timestamp"],
            major_indices=result["major_indices"],
            treasury_yields=result["treasury_yields"],
            commodities=result["commodities"],
            forex=result["forex"],
            volatility=result["volatility"]
        )
        
        return market_indicators.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting economic indicators: {e}")
        return f"Failed to retrieve economic indicators: {str(e)}"


@yfinance_server.tool(
    name="get_crypto_data",
    description="""Get cryptocurrency data.
    
Args:
    symbols: List[str]
        List of cryptocurrency symbols (e.g., ["BTC-USD", "ETH-USD", "DOGE-USD"])
"""
)
async def get_crypto_data(symbols: List[str]) -> str:
    """Get cryptocurrency data."""
    try:
        results = []
        for symbol in symbols:
            try:
                crypto = yf.Ticker(symbol)
                info = crypto.info
                
                if info:
                    data = CryptoData(
                        symbol=symbol,
                        name=info.get('name', ''),
                        price=info.get('regularMarketPrice', 0.0),
                        market_cap=info.get('marketCap', 0.0),
                        volume_24h=info.get('volume24Hr', 0.0),
                        change_24h=info.get('regularMarketChangePercent', 0.0),
                        circulating_supply=info.get('circulatingSupply', 0.0),
                        max_supply=info.get('maxSupply')
                    )
                    results.append(data)
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue
        
        return json.dumps([crypto.model_dump() for crypto in results], indent=2)
    except Exception as e:
        logger.error(f"Error getting cryptocurrency data: {e}")
        return f"Failed to retrieve cryptocurrency data: {str(e)}"


@yfinance_server.tool(
    name="get_esg_scores",
    description="""Get ESG (Environmental, Social, Governance) scores for a company.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
"""
)
async def get_esg_scores(symbol: str) -> str:
    """Get ESG scores for a company."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info:
            return f"No ESG data found for {symbol}"
        
        # Extract ESG data
        esg_data = ESGScores(
            symbol=symbol,
            total_score=info.get('esgScore', 0.0),
            environmental_score=info.get('environmentScore', 0.0),
            social_score=info.get('socialScore', 0.0),
            governance_score=info.get('governanceScore', 0.0),
            controversy_level=info.get('controversyLevel', 0),
            peer_rank=info.get('esgPerformance', ''),
            peer_percentile=info.get('percentile', 0.0)
        )
        
        return esg_data.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting ESG scores for {symbol}: {e}")
        return f"Failed to retrieve ESG scores: {str(e)}"


@yfinance_server.tool(
    name="debug_symbol_data",
    description="""Debug tool to see all available data for a symbol.
    
Args:
    symbol: str
        Symbol to debug (e.g., "BTC-USD", "AAPL")
"""
)
async def debug_symbol_data(symbol: str) -> str:
    """Debug tool to see all available data for a symbol."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        debug_info = {}
        
        # Get info data
        try:
            info = stock.info
            debug_info["info_keys"] = list(info.keys()) if info else []
            debug_info["info_sample"] = {k: v for k, v in list(info.items())[:20]} if info else {}
            
            # Look for price-related fields
            price_fields = {}
            for key, value in info.items():
                if any(price_word in key.lower() for price_word in ['price', 'close', 'open', 'high', 'low']):
                    price_fields[key] = value
            debug_info["price_fields"] = price_fields
            
        except Exception as e:
            debug_info["info_error"] = str(e)
        
        # Get historical data
        try:
            hist_1d = stock.history(period="1d")
            if not hist_1d.empty:
                debug_info["history_1d_last"] = {
                    "date": hist_1d.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    "open": float(hist_1d["Open"].iloc[-1]),
                    "high": float(hist_1d["High"].iloc[-1]),
                    "low": float(hist_1d["Low"].iloc[-1]),
                    "close": float(hist_1d["Close"].iloc[-1]),
                    "volume": int(hist_1d["Volume"].iloc[-1]) if not pd.isna(hist_1d["Volume"].iloc[-1]) else 0
                }
            else:
                debug_info["history_1d_error"] = "No data"
        except Exception as e:
            debug_info["history_1d_error"] = str(e)
        
        # Get minute data
        try:
            hist_1m = stock.history(period="1d", interval="1m")
            if not hist_1m.empty:
                debug_info["history_1m_last"] = {
                    "date": hist_1m.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    "close": float(hist_1m["Close"].iloc[-1]),
                    "volume": int(hist_1m["Volume"].iloc[-1]) if not pd.isna(hist_1m["Volume"].iloc[-1]) else 0
                }
            else:
                debug_info["history_1m_error"] = "No data"
        except Exception as e:
            debug_info["history_1m_error"] = str(e)
        
        return json.dumps(debug_info, indent=2)
    except Exception as e:
        return f"Debug error: {str(e)}"


def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, float]:
    """Calculate technical analysis indicators."""
    if data.empty:
        return {}

    try:
        # Prepare data
        df = data.copy()
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Calculate SMAs
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Calculate EMAs for MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Calculate MACD
        df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['STOCH_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Calculate ATR
        df['TR'] = pd.DataFrame({
            'HL': df['High'] - df['Low'],
            'HC': abs(df['High'] - df['Close'].shift(1)),
            'LC': abs(df['Low'] - df['Close'].shift(1))
        }).max(axis=1)
        df['ATR_14'] = df['TR'].rolling(window=14).mean()
        
        # Calculate OBV (On Balance Volume)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        # Calculate MFI (Money Flow Index)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        raw_money_flow = typical_price * df['Volume']
        positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), raw_money_flow, 0))
        negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), raw_money_flow, 0))
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        df['MFI_14'] = mfi
        
        # Calculate Williams %R
        highest_high = df['High'].rolling(window=14).max()
        lowest_low = df['Low'].rolling(window=14).min()
        df['Williams_R'] = ((highest_high - df['Close']) / (highest_high - lowest_low)) * -100
        
        # Calculate CCI (Commodity Channel Index)
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        tp_sma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: pd.Series(x).mad())
        df['CCI_20'] = (tp - tp_sma) / (0.015 * mad)
        
        # Get the latest values
        latest = df.iloc[-1]
        return {
            'sma_20': latest['SMA_20'],
            'sma_50': latest['SMA_50'],
            'sma_200': latest['SMA_200'],
            'ema_12': latest['EMA_12'],
            'ema_26': latest['EMA_26'],
            'macd_line': latest['MACD_Line'],
            'macd_signal': latest['MACD_Signal'],
            'macd_histogram': latest['MACD_Hist'],
            'rsi_14': latest['RSI_14'],
            'stoch_k': latest['STOCH_K'],
            'stoch_d': latest['STOCH_D'],
            'bollinger_upper': latest['BB_Upper'],
            'bollinger_middle': latest['BB_Middle'],
            'bollinger_lower': latest['BB_Lower'],
            'atr_14': latest['ATR_14'],
            'obv': latest['OBV'],
            'mfi_14': latest['MFI_14'],
            'williams_r': latest['Williams_R'],
            'cci_20': latest['CCI_20']
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {}


@yfinance_server.tool(
    name="get_technical_analysis",
    description="""Get comprehensive technical analysis indicators for a stock.
    
Args:
    symbol: str
        Stock ticker symbol (e.g., "AAPL", "MSFT", "GOOGL")
    interval: str
        Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
        Default: "1d"
    period: str
        Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        Default: "6mo"
"""
)
async def get_technical_analysis(
    symbol: str,
    interval: str = "1d",
    period: str = "6mo"
) -> str:
    """Get technical analysis indicators for a stock."""
    try:
        symbol = validate_symbol(symbol)
        stock = yf.Ticker(symbol)
        
        # Get historical data for calculations
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            return f"No historical data found for {symbol}"
        
        # Calculate indicators
        indicators = calculate_technical_indicators(hist)
        if not indicators:
            return f"Failed to calculate indicators for {symbol}"
        
        # Create technical analysis response
        technical_data = TechnicalIndicators(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol,
            **indicators
        )
        
        # Log indicator values
        logger.info(
            f"Technical Analysis for {symbol}:\n"
            f"MACD: {indicators['macd_line']:.2f} (Signal: {indicators['macd_signal']:.2f}, Hist: {indicators['macd_histogram']:.2f})\n"
            f"RSI: {indicators['rsi_14']:.2f}\n"
            f"Stochastic: %K={indicators['stoch_k']:.2f}, %D={indicators['stoch_d']:.2f}\n"
            f"Bollinger Bands: Upper={indicators['bollinger_upper']:.2f}, Middle={indicators['bollinger_middle']:.2f}, Lower={indicators['bollinger_lower']:.2f}"
        )
        
        return technical_data.model_dump_json(indent=2)
    except Exception as e:
        logger.error(f"Error getting technical analysis for {symbol}: {e}")
        return f"Failed to retrieve technical analysis: {str(e)}"


def main():
    """Main entry point for the Yahoo Finance MCP server."""
    logger.info("Starting Yahoo Finance MCP Server...")
    yfinance_server.run(transport="stdio")


if __name__ == "__main__":
    main() 
