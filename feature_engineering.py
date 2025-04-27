# feature_engineering.py
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import logging
from data import fetch_sentiment, fetch_macro_features

logger = logging.getLogger(__name__)

def add_technical_indicators(df, ticker=None):
    try:
        df = df.copy()
        required_cols = ['Date', 'Close', 'High', 'Low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df['rsi'] = RSIIndicator(close=df['Close'], window=14).rsi()
        df['sma'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['ema'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        df['volatility'] = df['Close'].rolling(window=20).std()
        bb = BollingerBands(close=df['Close'], window=20)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['atr'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        df['stoch'] = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14).stoch()
        
        if ticker:
            df['sentiment'] = fetch_sentiment(ticker)
        else:
            df['sentiment'] = 0.0
        
        macro = fetch_macro_features()
        for key, val in macro.items():
            df[key] = val
        
        logger.info("NaN counts after adding indicators:\n%s", 
                    df[['rsi', 'sma', 'ema', 'macd', 'volatility', 'bb_high', 'bb_low', 'atr', 'stoch', 'sentiment', 
                        'interest_rate', 'inflation', 'market_volatility']].isna().sum())
        df = df.dropna()
        logger.info("Columns after adding indicators: %s", df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Error in add_technical_indicators: %s", e)
        raise

def detect_volume_spike(df, window=20, threshold=2.0):
    try:
        df = df.copy()
        required_cols = ['Date', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        df['volume_mean'] = df['Volume'].rolling(window=window).mean()
        df['volume_std'] = df['Volume'].rolling(window=window).std()
        df['volume_spike'] = ((df['Volume'] - df['volume_mean']) > threshold * df['volume_std']).astype(int)
        df = df.drop(['volume_mean', 'volume_std'], axis=1)
        return df
    except Exception as e:
        logger.error("Error in detect_volume_spike: %s", e)
        raise