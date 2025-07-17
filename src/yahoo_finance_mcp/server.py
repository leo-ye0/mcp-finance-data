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
    """Economic indicator data."""
    indicator: str
    value: float
    date: str
    previous: float
    change: float
    change_percent: float


class EconomicEvent(BaseModel):
    """Economic calendar event."""
    event: str
    date: str
    country: str
    actual: Optional[str]
    forecast: Optional[str]
    previous: Optional[str]
    importance: str


class CryptoData(BaseModel):
    """Cryptocurrency data."""
    symbol: str
    name: str
    price: float
    market_cap: float
    volume_24h: float
    change_24h: float
    circulating_supply: float
    max_supply: Optional[float]


class ESGScores(BaseModel):
    """ESG (Environmental, Social, Governance) scores."""
    symbol: str
    total_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    controversy_level: int
    peer_rank: str
    peer_percentile: float


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
    description="""Get a summary of major market indices including S&P 500, Dow Jones, NASDAQ, Russell 2000, and VIX.
    
Args:
    None
"""
)
async def get_market_summary() -> str:
    """Get a summary of major market indices."""
    try:
        indices = {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones",
            "^IXIC": "NASDAQ",
            "^RUT": "Russell 2000",
            "^VIX": "VIX"
        }
        
        summary_data = {}
        
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current = hist["Close"].iloc[-1]
                    previous = hist["Close"].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100 if previous != 0 else 0
                    
                    summary_data[symbol] = MarketIndex(
                        name=name,
                        current_price=round(float(current), 2),
                        change=round(float(change), 2),
                        change_percent=round(float(change_pct), 2),
                        volume=int(hist["Volume"].iloc[-1]) if pd.notna(hist["Volume"].iloc[-1]) else 0
                    )
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
    description="""Get major economic indicators.
    
Args:
    indicators: List[str]
        List of economic indicators to fetch (e.g., ["^TNX", "^VIX", "^DJI"])
"""
)
async def get_economic_indicators(indicators: List[str]) -> str:
    """Get major economic indicators."""
    try:
        results = []
        for indicator in indicators:
            try:
                ticker = yf.Ticker(indicator)
                hist = ticker.history(period="2d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                    change = current - previous
                    change_pct = (change / previous) * 100 if previous != 0 else 0
                    
                    indicator_data = EconomicIndicator(
                        indicator=indicator,
                        value=float(current),
                        date=hist.index[-1].strftime("%Y-%m-%d"),
                        previous=float(previous),
                        change=float(change),
                        change_percent=float(change_pct)
                    )
                    results.append(indicator_data)
            except Exception as e:
                logger.warning(f"Failed to get data for {indicator}: {e}")
                continue
        
        return json.dumps([ind.model_dump() for ind in results], indent=2)
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


def main():
    """Main entry point for the Yahoo Finance MCP server."""
    logger.info("Starting Yahoo Finance MCP Server...")
    yfinance_server.run(transport="stdio")


if __name__ == "__main__":
    main() 