# modules/ml_models.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import streamlit as st

def create_features(df, window_sizes=[5, 10, 20]):
    df_features = df.copy()
    df_features.columns = ['price']
    df_features['time_idx'] = range(len(df_features))
    for lag in range(1, 4):
        df_features[f'lag_{lag}'] = df_features['price'].shift(lag)
    for window in window_sizes:
        df_features[f'ma_{window}'] = df_features['price'].rolling(window=window).mean()
        df_features[f'std_{window}'] = df_features['price'].rolling(window=window).std()
    return df_features

@st.cache_data(persist="disk")
def calculate_ml_prediction(data_tuple, days_ahead=30):
    df = pd.DataFrame(list(data_tuple), columns=['Close', 'High', 'Low', 'Volume']).astype(float)
    if df.empty or len(df) < 60:
        return None, None
    try:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['boll_upper'] = sma20 + (std20 * 2)
        df['boll_lower'] = sma20 - (std20 * 2)
        
        if 'Volume' in df.columns:
            df['volume_change'] = np.log(df['Volume'].replace(0, 1).astype(float)).diff()
        else:
            df['volume_change'] = 0
        
        df_features = create_features(pd.DataFrame({'price': df['Close']}))
        df_final = pd.concat([df_features, df[['rsi', 'macd', 'boll_upper', 'boll_lower', 'volume_change']]], axis=1)
        
        df_train = df_final.copy()
        df_train['target'] = df_train['price'].shift(-days_ahead)
        df_train = df_train.dropna()

        if df_train.empty: return None, None

        feature_cols = [col for col in df_train.columns if col not in ['price', 'target', 'time_idx']]
        X_train, y_train = df_train[feature_cols], df_train['target']
        
        # Using XGBoost for better TS prediction
        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        
        last_row = df_final.iloc[[-1]][feature_cols]
        predicted_price = model.predict(last_row)[0]
        
        current_price = df['Close'].iloc[-1]
        if current_price == 0:
            return None, None
        
        pct_change = ((predicted_price - current_price) / current_price) * 100
        return float(predicted_price), float(pct_change)
    except Exception as e:
        print(f"Erreur calculate_ml_prediction (XGBoost): {e}")
        return None, None

@st.cache_data(persist="disk")
def calculate_smart_prediction(prices_tuple, days_ahead=30):
    prices = list(prices_tuple)
    if not prices or len(prices) < 20:
        return None, None, None, None
    try:
        y = np.array(prices)
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 2)
        poly_func = np.poly1d(coeffs)
        
        future_x = len(x) + days_ahead
        future_price = poly_func(future_x)
        if future_price <= 0: future_price = 0.01
        
        residuals = y - poly_func(x)
        std_dev_residuals = np.std(residuals)
        prediction_range = std_dev_residuals * 1.5 

        future_low = future_price - prediction_range
        future_high = future_price + prediction_range
        if future_low <= 0: future_low = 0.01

        current_price = prices[-1]
        if current_price == 0:
            return None, None, None, None
            
        pct_change = ((future_price - current_price) / current_price) * 100
        
        return float(future_price), float(pct_change), float(future_low), float(future_high)
    except Exception as e:
        print(f"Erreur calculate_smart_prediction (Polyfit): {e}")
        return None, None, None, None
