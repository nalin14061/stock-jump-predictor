# data.py
from jugaad_data.nse import stock_df
from datetime import date, timedelta
import pandas as pd
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)

def fetch_stock_data(ticker, period="30y"):
    """
    Fetch historical stock data for a given ticker using jugaad_data.
    Args:
        ticker: Stock ticker (e.g., 'RELIANCE.NS')
        period: Time period ('1y' for 1 year, '30y' for 30 years)
    Returns:
        DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        # Ensure cache directory exists
        cache_dir = os.path.expanduser("~\\AppData\\Local\\nsehistory-stock\\nsehistory-stock\\Cache")
        logger.info("Cache directory: %s", cache_dir)
        logger.info("Directory exists before creation: %s", os.path.exists(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)
        logger.info("Directory exists after creation: %s", os.path.exists(cache_dir))
        logger.info("Directory writable: %s", os.access(cache_dir, os.W_OK))
        
        # Remove '.NS' suffix if present
        ticker = ticker.replace('.NS', '')
        
        # Calculate start and end dates
        end_date = date.today()
        if period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "30y":
            start_date = end_date - timedelta(days=365 * 30)
        else:
            raise ValueError("Unsupported period. Use '1y' or '30y'.")
        
        # Fetch data using stock_df
        df = stock_df(symbol=ticker, from_date=start_date, to_date=end_date, series="EQ")
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Rename columns to match expected format
        df = df.rename(columns={
            'DATE': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close',
            'VOLUME': 'Volume'
        })
        
        # Ensure required columns exist
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for {ticker}: {missing_cols}")
        
        # Convert Date to datetime and keep as column
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[required_cols]
        
        # Ensure numeric types
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
        
        logger.info("Fetched data for %s: %d rows, columns: %s", ticker, len(df), df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Error fetching data for %s: %s", ticker, e)
        raise

def fetch_sentiment(ticker):
    np.random.seed(hash(ticker) % 2**32)
    sentiment = np.random.uniform(-1, 1)
    logger.info("Sentiment for %s: %.3f", ticker, sentiment)
    return sentiment

def fetch_macro_features():
    return {
        'interest_rate': 0.065,
        'inflation': 0.042,
        'market_volatility': 0.15
    }