# app.py
import streamlit as st
import pandas as pd
import numpy as np
import logging
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import joblib
from data import fetch_stock_data, fetch_sentiment, fetch_macro_features
from feature_engineering import add_technical_indicators, detect_volume_spike
from model import load_jump_model, predict_jump

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Jump Predictor", layout="wide")
st.title("📈 Real-Time Stock Jump Prediction (India)")

# Configuration
TICKERS = ['RELIANCE.NS', 'ADANIPOWER.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
FEATURES = ['Close', 'rsi', 'macd', 'sma', 'ema', 'volatility', 'bb_high', 'bb_low', 'atr', 'stoch', 
            'sentiment', 'interest_rate', 'inflation', 'market_volatility', 'volume_spike']
THRESHOLD = 0.5

try:
    model = load_jump_model()
    scaler = joblib.load("models/scaler.save")
    logger.info("DNN model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Failed to load DNN model/scaler: {e}")
    st.stop()

xgb = None
try:
    xgb = joblib.load("models/xgb_model.joblib")
    logger.info("XGBoost model loaded successfully.")
except Exception as e:
    logger.warning("Failed to load XGBoost model: %s. Proceeding with DNN only.", e)
    st.warning("XGBoost model not available. Using DNN model only.")

def prepare_and_predict(ticker):
    try:
        df = fetch_stock_data(ticker, period="1y")
        logger.info("Data columns for %s: %s", ticker, df.columns.tolist())
        required_cols = ['Date', 'Close', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing columns: {required_cols}")
        
        df = add_technical_indicators(df, ticker=ticker)
        df = detect_volume_spike(df)
        
        missing_features = [f for f in FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = df[FEATURES].iloc[[-1]].copy()
        if X.isna().any().any():
            raise ValueError("NaN values in features")
        
        X_scaled = scaler.transform(X)
        
        jump_prob = predict_jump(model, X_scaled)[0][0]
        if not 0 <= jump_prob <= 1:
            raise ValueError(f"Invalid probability: {jump_prob}")
        
        jump_label = "Jump" if jump_prob > THRESHOLD else "No Jump"
        
        xgb_prob = 'N/A'
        xgb_label = 'N/A'
        if xgb is not None:
            xgb_prob = xgb.predict_proba(X_scaled)[:, 1][0]
            xgb_label = "Jump" if xgb_prob > THRESHOLD else "No Jump"
        
        trade_signal = "Buy" if jump_label == "Jump" else "Hold"
        
        df['future_close'] = df['Close'].shift(-1)
        df['pct_change'] = (df['future_close'] - df['Close']) / df['Close']
        df['jump'] = (df['pct_change'] > 0.02).astype(int)
        
        return {
            'Ticker': ticker,
            'Jump Likelihood (DNN)': round(float(jump_prob), 3),
            'Jump Likelihood (XGBoost)': round(float(xgb_prob), 3) if xgb_prob != 'N/A' else 'N/A',
            'Sentiment': round(float(df['sentiment'].iloc[-1]), 2),
            'Prediction (DNN)': jump_label,
            'Prediction (XGBoost)': xgb_label,
            'Trade Signal': trade_signal,
            'Historical Data': df[['Date', 'Close', 'jump']].dropna()
        }
    except Exception as e:
        logger.error(f"{ticker} - Prediction error: %s", e)
        return {
            'Ticker': ticker,
            'Jump Likelihood (DNN)': 'Error',
            'Jump Likelihood (XGBoost)': 'Error',
            'Sentiment': 'N/A',
            'Prediction (DNN)': 'Error',
            'Prediction (XGBoost)': 'Error',
            'Trade Signal': 'N/A',
            'Historical Data': None
        }

st.subheader("Prediction Results")
results = []
for ticker in TICKERS:
    result = prepare_and_predict(ticker)
    results.append(result)

df_result = pd.DataFrame([{
    'Ticker': r['Ticker'],
    'Jump Likelihood (DNN)': r['Jump Likelihood (DNN)'],
    'Jump Likelihood (XGBoost)': r['Jump Likelihood (XGBoost)'],
    'Sentiment': r['Sentiment'],
    'Prediction (DNN)': r['Prediction (DNN)'],
    'Prediction (XGBoost)': r['Prediction (XGBoost)'],
    'Trade Signal': r['Trade Signal']
} for r in results]).sort_values(by='Jump Likelihood (DNN)', ascending=False)
st.dataframe(df_result)

st.subheader("Visualizations")
probs_dnn = df_result[df_result['Jump Likelihood (DNN)'].apply(lambda x: isinstance(x, float))]['Jump Likelihood (DNN)']
probs_xgb = df_result[df_result['Jump Likelihood (XGBoost)'].apply(lambda x: isinstance(x, float))]['Jump Likelihood (XGBoost)']
if not probs_dnn.empty:
    fig = px.histogram(pd.DataFrame({
        'DNN': probs_dnn,
        'XGBoost': probs_xgb
    }), title="Prediction Probability Distribution", barmode='overlay')
    st.plotly_chart(fig)

st.subheader("Historical Jump Trends")
ticker_select = st.selectbox("Select Ticker for Historical Trends", TICKERS)
selected_result = next(r for r in results if r['Ticker'] == ticker_select)
if selected_result['Historical Data'] is not None:
    df_hist = selected_result['Historical Data']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist['Date'], y=df_hist['Close'], mode='lines', name='Close Price'))
    jump_dates = df_hist[df_hist['jump'] == 1]['Date']
    jump_prices = df_hist[df_hist['jump'] == 1]['Close']
    fig.add_trace(go.Scatter(x=jump_dates, y=jump_prices, mode='markers', name='Jumps', marker=dict(color='red', size=10)))
    fig.update_layout(title=f"{ticker_select} Close Price and Jumps", xaxis_title="Date", yaxis_title="Close Price")
    st.plotly_chart(fig)
else:
    st.warning(f"No historical data available for {ticker_select}")

try:
    rf_importance = pd.read_csv("feature_importance.csv")
    st.subheader("Feature Importance")
    fig = px.bar(rf_importance, x='Feature', y='Importance', title="Random Forest Feature Importance")
    st.plotly_chart(fig)
except:
    st.warning("Feature importance file not found.")

st.subheader("Notes")
st.write("- Predictions are based on the latest data point for real-time trading.")
st.write(f"- Jump threshold: {THRESHOLD} (probability above this indicates a jump).")
st.write("- Trade Signal: 'Buy' if a jump is predicted, 'Hold' otherwise.")
st.write("- Features include technical indicators, sentiment, macro data, and volume spikes.")
st.write("- XGBoost model provides an alternative prediction if available.")