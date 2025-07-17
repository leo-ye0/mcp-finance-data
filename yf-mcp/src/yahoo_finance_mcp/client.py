"""Yahoo Finance client for data retrieval."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any
import logging
from .types import (
    StockInfo, PriceData, FinancialStatement, NewsItem, AnalystRecommendation,
    InstitutionalHolder, InsiderTransaction, OptionContract, OptionsChain,
    DividendData, StockSplit, StatementType, Period, Interval, TimePeriod, OptionType
)

logger = logging.getLogger(__name__)


class YahooFinanceClient:
    """Client for Yahoo Finance data retrieval."""
    
    def __init__(self):
        """Initialize the Yahoo Finance client."""
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
    
    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Get a yfinance Ticker object with caching."""
        cache_key = f"ticker_{symbol}"
        now = datetime.now()
        
        if cache_key in self._cache:
            ticker, timestamp = self._cache[cache_key]
            if (now - timestamp).seconds < self._cache_timeout:
                return ticker
        
        ticker = yf.Ticker(symbol.upper())
        self._cache[cache_key] = (ticker, now)
        return ticker
    
    def get_stock_info(self, symbol: str) -> StockInfo:
        """Get comprehensive stock information."""
        try:
            ticker = self._get_ticker(symbol)
            info = ticker.info
            
            return StockInfo(
                symbol=symbol.upper(),
                name=info.get('longName', ''),
                sector=info.get('sector'),
                industry=info.get('industry'),
                market_cap=info.get('marketCap'),
                enterprise_value=info.get('enterpriseValue'),
                trailing_pe=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                price_to_sales=info.get('priceToSalesTrailing12Months'),
                enterprise_to_revenue=info.get('enterpriseToRevenue'),
                enterprise_to_ebitda=info.get('enterpriseToEbitda'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                current_price=info.get('currentPrice'),
                target_mean_price=info.get('targetMeanPrice'),
                recommendation_mean=info.get('recommendationMean'),
                dividend_yield=info.get('dividendYield'),
                beta=info.get('beta'),
                volume=info.get('volume'),
                avg_volume=info.get('averageVolume'),
                website=info.get('website'),
                business_summary=info.get('longBusinessSummary')
            )
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            raise ValueError(f"Could not retrieve stock info for {symbol}: {e}")
    
    def get_historical_prices(
        self, 
        symbol: str, 
        period: TimePeriod = TimePeriod.ONE_YEAR,
        interval: Interval = Interval.ONE_DAY
    ) -> List[PriceData]:
        """Get historical price data."""
        try:
            ticker = self._get_ticker(symbol)
            hist = ticker.history(period=period.value, interval=interval.value)
            
            prices = []
            for date, row in hist.iterrows():
                prices.append(PriceData(
                    date=date.to_pydatetime(),
                    open=row.get('Open'),
                    high=row.get('High'),
                    low=row.get('Low'),
                    close=row.get('Close'),
                    adj_close=row.get('Adj Close'),
                    volume=int(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None
                ))
            
            return prices
        except Exception as e:
            logger.error(f"Error getting historical prices for {symbol}: {e}")
            raise ValueError(f"Could not retrieve historical prices for {symbol}: {e}")
    
    def get_financial_statements(
        self, 
        symbol: str, 
        statement_type: StatementType,
        period: Period = Period.QUARTERLY
    ) -> List[FinancialStatement]:
        """Get financial statements."""
        try:
            ticker = self._get_ticker(symbol)
            
            # Get the appropriate statement
            if statement_type == StatementType.BALANCE_SHEET:
                if period == Period.QUARTERLY:
                    df = ticker.quarterly_balance_sheet
                else:
                    df = ticker.balance_sheet
            elif statement_type == StatementType.INCOME_STATEMENT:
                if period == Period.QUARTERLY:
                    df = ticker.quarterly_financials
                else:
                    df = ticker.financials
            elif statement_type == StatementType.CASH_FLOW:
                if period == Period.QUARTERLY:
                    df = ticker.quarterly_cashflow
                else:
                    df = ticker.cashflow
            
            statements = []
            if not df.empty:
                for date in df.columns:
                    data = {}
                    for index in df.index:
                        value = df.loc[index, date]
                        if pd.notna(value):
                            data[str(index)] = float(value) if isinstance(value, (int, float)) else str(value)
                        else:
                            data[str(index)] = None
                    
                    statements.append(FinancialStatement(
                        symbol=symbol.upper(),
                        statement_type=statement_type,
                        period=period,
                        date=date.to_pydatetime(),
                        data=data
                    ))
            
            return statements
        except Exception as e:
            logger.error(f"Error getting financial statements for {symbol}: {e}")
            raise ValueError(f"Could not retrieve financial statements for {symbol}: {e}")
    
    def get_stock_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Get latest news for a stock."""
        try:
            ticker = self._get_ticker(symbol)
            news = ticker.news
            
            news_items = []
            for item in news[:limit]:
                news_items.append(NewsItem(
                    title=item.get('title', ''),
                    link=item.get('link', ''),
                    publisher=item.get('publisher', ''),
                    publish_time=datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    summary=item.get('summary')
                ))
            
            return news_items
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {e}")
            raise ValueError(f"Could not retrieve news for {symbol}: {e}")
    
    def get_analyst_recommendations(self, symbol: str) -> List[AnalystRecommendation]:
        """Get analyst recommendations."""
        try:
            ticker = self._get_ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is None or recommendations.empty:
                return []
            
            recs = []
            for date, row in recommendations.iterrows():
                recs.append(AnalystRecommendation(
                    period=str(date.date()),
                    strong_buy=int(row.get('strongBuy', 0)),
                    buy=int(row.get('buy', 0)),
                    hold=int(row.get('hold', 0)),
                    sell=int(row.get('sell', 0)),
                    strong_sell=int(row.get('strongSell', 0)),
                    recommendation_mean=float(row.get('recommendationMean', 0)),
                    recommendation_key=row.get('recommendationKey', '')
                ))
            
            return recs
        except Exception as e:
            logger.error(f"Error getting analyst recommendations for {symbol}: {e}")
            raise ValueError(f"Could not retrieve analyst recommendations for {symbol}: {e}")
    
    def get_institutional_holders(self, symbol: str) -> List[InstitutionalHolder]:
        """Get institutional holders."""
        try:
            ticker = self._get_ticker(symbol)
            holders = ticker.institutional_holders
            
            if holders is None or holders.empty:
                return []
            
            holder_list = []
            for _, row in holders.iterrows():
                holder_list.append(InstitutionalHolder(
                    holder=row.get('Holder', ''),
                    shares=int(row.get('Shares', 0)),
                    date_reported=pd.to_datetime(row.get('Date Reported')).to_pydatetime(),
                    percent_out=float(row.get('% Out', 0)),
                    value=int(row.get('Value', 0))
                ))
            
            return holder_list
        except Exception as e:
            logger.error(f"Error getting institutional holders for {symbol}: {e}")
            raise ValueError(f"Could not retrieve institutional holders for {symbol}: {e}")
    
    def get_insider_transactions(self, symbol: str) -> List[InsiderTransaction]:
        """Get insider transactions."""
        try:
            ticker = self._get_ticker(symbol)
            insiders = ticker.insider_transactions
            
            if insiders is None or insiders.empty:
                return []
            
            transactions = []
            for _, row in insiders.iterrows():
                transactions.append(InsiderTransaction(
                    insider=row.get('Insider', ''),
                    relation=row.get('Relation', ''),
                    last_date=pd.to_datetime(row.get('Last Date')).to_pydatetime(),
                    transaction=row.get('Transaction', ''),
                    owner_type=row.get('Owner Type', ''),
                    shares_owned=int(row.get('Shares Owned', 0)) if pd.notna(row.get('Shares Owned')) else None,
                    shares_transacted=int(row.get('Shares Transacted', 0)) if pd.notna(row.get('Shares Transacted')) else None,
                    value=float(row.get('Value', 0)) if pd.notna(row.get('Value')) else None
                ))
            
            return transactions
        except Exception as e:
            logger.error(f"Error getting insider transactions for {symbol}: {e}")
            raise ValueError(f"Could not retrieve insider transactions for {symbol}: {e}")
    
    def get_options_expirations(self, symbol: str) -> List[str]:
        """Get available options expiration dates."""
        try:
            ticker = self._get_ticker(symbol)
            return list(ticker.options)
        except Exception as e:
            logger.error(f"Error getting options expirations for {symbol}: {e}")
            raise ValueError(f"Could not retrieve options expirations for {symbol}: {e}")
    
    def get_options_chain(
        self, 
        symbol: str, 
        expiration: str,
        option_type: OptionType = OptionType.BOTH
    ) -> OptionsChain:
        """Get options chain for a specific expiration."""
        try:
            ticker = self._get_ticker(symbol)
            options = ticker.option_chain(expiration)
            
            def process_options_df(df: pd.DataFrame) -> List[OptionContract]:
                contracts = []
                for _, row in df.iterrows():
                    contracts.append(OptionContract(
                        contract_symbol=row.get('contractSymbol', ''),
                        strike=float(row.get('strike', 0)),
                        last_price=float(row.get('lastPrice', 0)),
                        bid=float(row.get('bid', 0)),
                        ask=float(row.get('ask', 0)),
                        change=float(row.get('change', 0)),
                        percent_change=float(row.get('percentChange', 0)),
                        volume=int(row.get('volume', 0)) if pd.notna(row.get('volume')) else None,
                        open_interest=int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else None,
                        implied_volatility=float(row.get('impliedVolatility', 0)),
                        in_the_money=bool(row.get('inTheMoney', False))
                    ))
                return contracts
            
            calls = process_options_df(options.calls) if option_type in [OptionType.CALLS, OptionType.BOTH] else []
            puts = process_options_df(options.puts) if option_type in [OptionType.PUTS, OptionType.BOTH] else []
            
            return OptionsChain(
                expiration=expiration,
                calls=calls,
                puts=puts
            )
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            raise ValueError(f"Could not retrieve options chain for {symbol}: {e}")
    
    def get_dividend_history(self, symbol: str) -> List[DividendData]:
        """Get dividend history."""
        try:
            ticker = self._get_ticker(symbol)
            dividends = ticker.dividends
            
            dividend_list = []
            for date, dividend in dividends.items():
                dividend_list.append(DividendData(
                    date=date.to_pydatetime(),
                    dividend=float(dividend)
                ))
            
            return dividend_list
        except Exception as e:
            logger.error(f"Error getting dividend history for {symbol}: {e}")
            raise ValueError(f"Could not retrieve dividend history for {symbol}: {e}")
    
    def get_stock_splits(self, symbol: str) -> List[StockSplit]:
        """Get stock split history."""
        try:
            ticker = self._get_ticker(symbol)
            splits = ticker.splits
            
            split_list = []
            for date, split_ratio in splits.items():
                split_list.append(StockSplit(
                    date=date.to_pydatetime(),
                    split_ratio=str(split_ratio)
                ))
            
            return split_list
        except Exception as e:
            logger.error(f"Error getting stock splits for {symbol}: {e}")
            raise ValueError(f"Could not retrieve stock splits for {symbol}: {e}") 