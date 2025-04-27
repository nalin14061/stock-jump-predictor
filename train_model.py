# train_model_multi_ticker.py
import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from imblearn.over_sampling import ADASYN
from data import fetch_stock_data
from feature_engineering import add_technical_indicators, detect_volume_spike

# Config
TICKERS = ["RELIANCE.NS", "ADANIPOWER.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
JUMP_THRESHOLD = 0.02
MODEL_PATH = "models/dnn_jump_model_multi.h5"
SCALER_PATH = "models/scaler.save"
XGB_PATH = "models/xgb_model.joblib"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def label_jump(df, threshold=0.02):
    df['future_close'] = df['Close'].shift(-1)
    df['pct_change'] = (df['future_close'] - df['Close']) / df['Close']
    df['jump'] = (df['pct_change'] > threshold).astype(int)
    df.dropna(inplace=True)
    return df

def prepare_dataset(tickers):
    logger.info("Preparing data for tickers: %s", tickers)
    try:
        dfs = []
        for ticker in tickers:
            logger.info("Fetching data for %s", ticker)
            df = fetch_stock_data(ticker, period="30y")
            logger.info("Raw data columns for %s: %s", ticker, df.columns.tolist())
            df[['Close', 'High', 'Low']] = df[['Close', 'High', 'Low']].astype(float)
            df = add_technical_indicators(df, ticker=ticker)
            df = detect_volume_spike(df)
            df = label_jump(df, JUMP_THRESHOLD)
            df['ticker'] = ticker
            dfs.append(df)
        
        df_combined = pd.concat(dfs, ignore_index=True)
        logger.info("Combined data shape: %s", df_combined.shape)
        logger.info("Columns after processing: %s", df_combined.columns.tolist())
        
        required_features = ['Close', 'rsi', 'macd', 'sma', 'ema', 'volatility', 'bb_high', 'bb_low', 'atr', 'stoch', 
                           'sentiment', 'interest_rate', 'inflation', 'market_volatility', 'volume_spike']
        missing = [col for col in required_features if col not in df_combined.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        feature_names = required_features
        
        logger.info("Feature head:\n%s", df_combined[['ticker', 'High', 'Low', 'Close'] + feature_names].head())
        logger.info("Feature stats:\n%s", df_combined[feature_names].describe())
        
        X = df_combined[feature_names].values
        y = df_combined['jump'].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.05, random_state=42, stratify=y), scaler, df_combined, feature_names
    except Exception as e:
        logger.error("Error in prepare_dataset: %s", e)
        raise

def build_model(input_dim):
    model = Sequential([
        Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

class KerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        return self.model.predict(X)

def train_and_save():
    try:
        (X_train, X_test, y_train, y_test), scaler, df_combined, feature_names = prepare_dataset(TICKERS)
    except Exception as e:
        logger.error("Error in prepare_dataset: %s", e)
        return
    
    adasyn = ADASYN(sampling_strategy=0.3, random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    
    class_weights = {0: 1.0, 1: 2.0}
    logger.info("Class weights: %s", class_weights)
    logger.info("Original class distribution: %s", np.bincount(y_train))
    logger.info("Resampled class distribution: %s", np.bincount(y_train_resampled))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    f1_scores = []
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True), ReduceLROnPlateau(factor=0.5, patience=3)]
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        X_tr_res, y_tr_res = adasyn.fit_resample(X_tr, y_tr)
        model = build_model(X_train.shape[1])
        model.fit(X_tr_res, y_tr_res, epochs=100, batch_size=32, validation_split=0.1,
                  class_weight=class_weights, callbacks=callbacks, verbose=0)
        y_val_pred = model.predict(X_val).flatten()
        auc_scores.append(roc_auc_score(y_val, y_val_pred))
        f1_scores.append(f1_score(y_val, y_val_pred > 0.5))
    logger.info("Cross-validated ROC AUC: %.3f ± %.3f", np.mean(auc_scores), np.std(auc_scores))
    logger.info("Cross-validated F1-score (class 1): %.3f ± %.3f", np.mean(f1_scores), np.std(f1_scores))
    
    model = build_model(X_train.shape[1])
    model.fit(X_train_resampled, y_train_resampled, epochs=100, batch_size=32,
              validation_split=0.1, class_weight=class_weights, callbacks=callbacks, verbose=1)
    
    os.makedirs("models", exist_ok=True)
    try:
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("DNN model and scaler saved to %s, %s", MODEL_PATH, SCALER_PATH)
    except Exception as e:
        logger.error("Error saving DNN model/scaler: %s", e)
    
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(scale_pos_weight=2.0, random_state=42)
        xgb.fit(X_train_resampled, y_train_resampled)
        joblib.dump(xgb, XGB_PATH)
        logger.info("XGBoost model saved to %s", XGB_PATH)
        y_pred_prob_xgb = xgb.predict_proba(X_test)[:, 1]
        print("✅ XGBoost ROC AUC:", roc_auc_score(y_test, y_pred_prob_xgb))
        print("✅ XGBoost Classification Report:\n", classification_report(y_test, y_pred_prob_xgb > 0.5))
        print("✅ XGBoost F1-score (class 1):", f1_score(y_test, y_pred_prob_xgb > 0.5))
    except Exception as e:
        logger.error("Error in XGBoost training/saving: %s", e)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train_resampled, y_train_resampled)
        rf_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf.feature_importances_
        })
        rf_importance.to_csv("feature_importance.csv", index=False)
        logger.info("RF Feature Importance saved to feature_importance.csv")
        for name, score in zip(feature_names, rf.feature_importances_):
            print(f"{name}: {score:.4f}")
    except Exception as e:
        logger.error("Error in RF feature importance: %s", e)
    
    y_pred_prob = model.predict(X_test).flatten()
    logger.info("Class distribution in y_test: %s", np.bincount(y_test))
    
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    logger.info("DNN ROC AUC Score: %.3f", auc_roc)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recalls, precisions)
    logger.info("DNN Precision-Recall AUC: %.3f", pr_auc)
    
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores)] if np.max(f1_scores) > 0 and not np.isnan(np.max(f1_scores)) else 0.5
    logger.info("Optimal threshold: %.3f", optimal_threshold)
    
    predictions = pd.DataFrame({
        'ticker': df_combined.iloc[-len(y_test):]['ticker'],
        'true_label': y_test,
        'pred_prob': y_pred_prob,
        'pred_label': (y_pred_prob > optimal_threshold).astype(int),
        'pred_label_str': np.where(y_pred_prob > optimal_threshold, 'Jump', 'No Jump')
    })
    print("✅ Predictions Sample:\n", predictions.head())
    predictions.to_csv("predictions.csv", index=False)
    
    print("✅ DNN Classification Report (optimal threshold):\n",
          classification_report(y_test, y_pred_prob > optimal_threshold))
    
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"\n✅ DNN Classification Report (threshold={thr:.3f}):")
        print(classification_report(y_test, y_pred_prob > thr))
    
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig("pr_curve.png")
    plt.close()
    
    logger.info("Training completed successfully.")

if __name__ == '__main__':
    train_and_save()