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

@st.cache_data(show_spinner=False)
def calculate_ml_prediction(data_tuple, days_ahead=30):
    df = pd.DataFrame(list(data_tuple), columns=['Close', 'High', 'Low', 'Volume']).astype(float)
    if df.empty or len(df) < 60:
        return None, None
    try:
        # RSI (simplifié)
        delta = df['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Moyenner avec Pandas rolling plus efficacement
        avg_gain = pd.Series(gain).rolling(window=14, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10) # évite division par zéro
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (vectorisé via EWM)
        df['macd'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
        
        # Bollinger Bands
        sma20 = df['Close'].rolling(window=20, min_periods=1).mean()
        std20 = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
        df['boll_upper'] = sma20 + (std20 * 2)
        df['boll_lower'] = sma20 - (std20 * 2)
        
        # Volume change
        if 'Volume' in df.columns:
            df['volume_change'] = df['Volume'].pct_change().fillna(0)
            # Remplacer les infinis potentiels
            df['volume_change'] = df['volume_change'].replace([np.inf, -np.inf], 0)
        else:
            df['volume_change'] = 0
            
        df_features = create_features(df[['Close']].rename(columns={'Close': 'price'}))
        
        df_final = pd.concat([df_features, df[['rsi', 'macd', 'boll_upper', 'boll_lower', 'volume_change']]], axis=1)
        
        # Création de la cible (décalage vers le haut)
        df_final['target'] = df_final['price'].shift(-days_ahead)
        
        # Pour l'entraînement, supprimer les lignes où la cible est NaN
        df_train = df_final.dropna(subset=['target']).copy()
        
        # Nettoyage des NaNs dans les features pour éviter les plantages de XGBoost
        df_train = df_train.fillna(method='bfill').fillna(0)

        if df_train.empty or len(df_train) < 10: 
            return None, None

        feature_cols = [col for col in df_train.columns if col not in ['price', 'target', 'time_idx']]
        X_train, y_train = df_train[feature_cols], df_train['target']
        
        # Modèle allégé pour Streamlit Cloud (très rapide, conso RAM minime)
        model = XGBRegressor(
            n_estimators=30,      # Réduit de 100 à 30 (plus rapide)
            max_depth=3,          # Réduit de 4 à 3 (moins lourd en mémoire)
            learning_rate=0.1,    # Compensé par un lr plus élevé
            random_state=42,
            n_jobs=1              # Empêche XGBoost de saturer le CPU virtuel (single_thread mode)
        )
        model.fit(X_train, y_train)
        
        # Prédiction sur la toute dernière ligne connue
        last_row = df_final.iloc[[-1]][feature_cols].fillna(0)
        predicted_price = model.predict(last_row)[0]
        
        current_price = df['Close'].iloc[-1]
        if current_price <= 0:
            return None, None
        
        pct_change = ((predicted_price - current_price) / current_price) * 100
        return float(predicted_price), float(pct_change)
    except Exception as e:
        print(f"Erreur calculate_ml_prediction (XGBoost): {e}")
        return None, None

@st.cache_data(show_spinner=False)
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
