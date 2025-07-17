"""Type definitions for Yahoo Finance MCP server."""

from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class StatementType(str, Enum):
    """Financial statement types."""
    BALANCE_SHEET = "balance_sheet"
    INCOME_STATEMENT = "income_statement"
    CASH_FLOW = "cash_flow"


class Period(str, Enum):
    """Time period options."""
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class Interval(str, Enum):
    """Price data intervals."""
    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"


class TimePeriod(str, Enum):
    """Historical data time periods."""
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    TEN_YEARS = "10y"
    YTD = "ytd"
    MAX = "max"


class OptionType(str, Enum):
    """Option types."""
    CALLS = "calls"
    PUTS = "puts"
    BOTH = "both"


class StockInfo(BaseModel):
    """Stock information model."""
    symbol: str
    name: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    trailing_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_to_revenue: Optional[float] = None
    enterprise_to_ebitda: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    current_price: Optional[float] = None
    target_mean_price: Optional[float] = None
    recommendation_mean: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    website: Optional[str] = None
    business_summary: Optional[str] = None


class PriceData(BaseModel):
    """Historical price data model."""
    date: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    adj_close: Optional[float] = None
    volume: Optional[int] = None


class FinancialStatement(BaseModel):
    """Financial statement model."""
    symbol: str
    statement_type: StatementType
    period: Period
    date: datetime
    data: Dict[str, Union[float, int, str, None]]


class NewsItem(BaseModel):
    """News item model."""
    title: str
    link: str
    publisher: str
    publish_time: datetime
    summary: Optional[str] = None


class AnalystRecommendation(BaseModel):
    """Analyst recommendation model."""
    period: str
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    recommendation_mean: float
    recommendation_key: str


class InstitutionalHolder(BaseModel):
    """Institutional holder model."""
    holder: str
    shares: int
    date_reported: datetime
    percent_out: float
    value: int


class InsiderTransaction(BaseModel):
    """Insider transaction model."""
    insider: str
    relation: str
    last_date: datetime
    transaction: str
    owner_type: str
    shares_owned: Optional[int] = None
    shares_transacted: Optional[int] = None
    value: Optional[float] = None


class OptionContract(BaseModel):
    """Option contract model."""
    contract_symbol: str
    strike: float
    last_price: float
    bid: float
    ask: float
    change: float
    percent_change: float
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: float
    in_the_money: bool


class OptionsChain(BaseModel):
    """Options chain model."""
    expiration: str
    calls: List[OptionContract]
    puts: List[OptionContract]


class DividendData(BaseModel):
    """Dividend data model."""
    date: datetime
    dividend: float


class StockSplit(BaseModel):
    """Stock split model."""
    date: datetime
    split_ratio: str 