"""
Test cases for Yahoo Finance client.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
from yahoo_finance_mcp.client import YahooFinanceClient
from yahoo_finance_mcp.types import StatementType, Period, TimePeriod, Interval


class TestYahooFinanceClient:
    """Test cases for YahooFinanceClient."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = YahooFinanceClient()
    
    @patch('yahoo_finance_mcp.client.yf.Ticker')
    def test_get_stock_info_success(self, mock_ticker):
        """Test successful stock info retrieval."""
        # Mock the ticker and info response
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = {
            'longName': 'Apple Inc.',
            'sector': 'Technology',
            'marketCap': 3000000000000,
            'currentPrice': 150.0,
            'trailingPE': 25.0
        }
        mock_ticker.return_value = mock_ticker_instance
        
        # Call the method
        result = self.client.get_stock_info("AAPL")
        
        # Verify results
        assert result.symbol == "AAPL"
        assert result.name == "Apple Inc."
        assert result.sector == "Technology"
        assert result.market_cap == 3000000000000
        assert result.current_price == 150.0
        assert result.trailing_pe == 25.0
    
    @patch('yahoo_finance_mcp.client.yf.Ticker')
    def test_get_stock_info_invalid_symbol(self, mock_ticker):
        """Test stock info retrieval with invalid symbol."""
        # Mock ticker to raise an exception
        mock_ticker.side_effect = Exception("Invalid symbol")
        
        # Verify exception is raised
        with pytest.raises(ValueError, match="Could not retrieve stock info"):
            self.client.get_stock_info("INVALID")
    
    @patch('yahoo_finance_mcp.client.yf.Ticker')
    def test_get_historical_prices_success(self, mock_ticker):
        """Test successful historical price retrieval."""
        # Create mock historical data
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [99, 100, 101, 102, 103],
            'Close': [104, 105, 106, 107, 108],
            'Adj Close': [104, 105, 106, 107, 108],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }, index=dates)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Call the method
        result = self.client.get_historical_prices("AAPL", TimePeriod.ONE_MONTH, Interval.ONE_DAY)
        
        # Verify results
        assert len(result) == 5
        assert result[0].open == 100
        assert result[0].close == 104
        assert result[0].volume == 1000000
        assert isinstance(result[0].date, datetime)
    
    @patch('yahoo_finance_mcp.client.yf.Ticker')
    def test_get_financial_statements_success(self, mock_ticker):
        """Test successful financial statement retrieval."""
        # Create mock financial data
        dates = pd.date_range(start='2023-01-01', periods=2, freq='Q')
        mock_data = pd.DataFrame({
            dates[0]: {'Total Revenue': 100000000, 'Net Income': 20000000},
            dates[1]: {'Total Revenue': 110000000, 'Net Income': 25000000}
        }).T
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.quarterly_financials = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Call the method
        result = self.client.get_financial_statements(
            "AAPL", StatementType.INCOME_STATEMENT, Period.QUARTERLY
        )
        
        # Verify results
        assert len(result) == 2
        assert result[0].symbol == "AAPL"
        assert result[0].statement_type == StatementType.INCOME_STATEMENT
        assert result[0].period == Period.QUARTERLY
        assert 'Total Revenue' in result[0].data
    
    @patch('yahoo_finance_mcp.client.yf.Ticker')
    def test_get_stock_news_success(self, mock_ticker):
        """Test successful news retrieval."""
        mock_news = [
            {
                'title': 'Apple Reports Strong Earnings',
                'link': 'https://example.com/news1',
                'publisher': 'Tech News',
                'providerPublishTime': 1640995200,  # 2022-01-01
                'summary': 'Apple exceeded expectations...'
            },
            {
                'title': 'iPhone Sales Surge',
                'link': 'https://example.com/news2',
                'publisher': 'Market Watch',
                'providerPublishTime': 1641081600,  # 2022-01-02
                'summary': 'Strong iPhone demand...'
            }
        ]
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.news = mock_news
        mock_ticker.return_value = mock_ticker_instance
        
        # Call the method
        result = self.client.get_stock_news("AAPL", limit=2)
        
        # Verify results
        assert len(result) == 2
        assert result[0].title == 'Apple Reports Strong Earnings'
        assert result[0].publisher == 'Tech News'
        assert result[1].title == 'iPhone Sales Surge'
    
    @patch('yahoo_finance_mcp.client.yf.Ticker')
    def test_get_options_expirations_success(self, mock_ticker):
        """Test successful options expiration retrieval."""
        mock_expirations = ['2024-01-19', '2024-02-16', '2024-03-15']
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.options = mock_expirations
        mock_ticker.return_value = mock_ticker_instance
        
        # Call the method
        result = self.client.get_options_expirations("AAPL")
        
        # Verify results
        assert result == mock_expirations
    
    def test_ticker_caching(self):
        """Test that ticker objects are cached properly."""
        with patch('yahoo_finance_mcp.client.yf.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            
            # Call twice with same symbol
            ticker1 = self.client._get_ticker("AAPL")
            ticker2 = self.client._get_ticker("AAPL")
            
            # Verify ticker was only created once due to caching
            assert mock_ticker.call_count == 1
            assert ticker1 is ticker2


if __name__ == "__main__":
    pytest.main([__file__]) 