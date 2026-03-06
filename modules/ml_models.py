# modules/ml_models.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import streamlit as st
import logging

# On retire les logs Prophet qui peuvent spammer la console Streamlit
logging.getLogger('prophet').setLevel(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from prophet import Prophet
except ImportError:
    Prophet = None

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
    # data_tuple contient maintenant l'index de date en plus des autres colonnes
    # index: 0, Close: 1, High: 2, Low: 3, Volume: 4
    df = pd.DataFrame(list(data_tuple), columns=['Date', 'Close', 'High', 'Low', 'Volume'])
    df = df.set_index('Date').astype(float)
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
        
        # Modèle optimisé pour plus de précision (compromis Cloud / Performance)
        model = XGBRegressor(
            n_estimators=100,     # Augmenté de 30 à 100 pour plus de précision
            max_depth=5,          # Augmenté de 3 à 5 pour capter plus de nuances
            learning_rate=0.05,   # Réduit pour une convergence plus fine
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
        
        # Ajout de poids exponentiels pour donner plus d'importance aux données récentes
        # Le dernier point aura un poids de 1.0, le plus ancien aura un poids plus faible
        weights = np.exp(np.linspace(-2, 0, len(y)))
        
        coeffs = np.polyfit(x, y, 2, w=weights)
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


@st.cache_data(show_spinner=False)
def calculate_prophet_prediction(data_tuple, days_ahead=30):
    if Prophet is None:
        return None, None, None, None
        
    # data_tuple contient Date, Close, High, Low, Volume
    df_raw = pd.DataFrame(list(data_tuple), columns=['Date', 'Close', 'High', 'Low', 'Volume'])
    
    if df_raw.empty or len(df_raw) < 20: 
        # Prophet a besoin d'un minimum de points, mais il est plus flexible que XGBoost.
        return None, None, None, None

    try:
        # Prophet a spécifiquement besoin de colonnes 'ds' (datestamp) et 'y' (valeur)
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(df_raw['Date']), # Formatage stricte de la date
            'y': df_raw['Close'].astype(float) # Valeur de clôture
        })
        
        # Filtre de sécurité: Si un prix est = 0, on le retire pour la log-transformation
        df_prophet = df_prophet[df_prophet['y'] > 0]
        if df_prophet.empty: return None, None, None, None
        
        # Transformation Logarithmique : Réduit l'impact de l'hyper-volatilité (très fréquent en crypto)
        # Cela empêche les prédictions (notamment la fourcehette haute/basse) d'exploser vers l'infini ou le négatif.
        df_prophet['y'] = np.log(df_prophet['y'])

        # Paramétrage de Prophet optimisé pour de la finance de court/moyen terme
        model = Prophet(
            daily_seasonality=False, # Pas assez de granularité (on est en journalier)
            weekly_seasonality=True, # Le week-end est important en crypto, faible en bourse
            yearly_seasonality=False, # Historique de 3 mois = pas de tendance annuelle pertinente
            changepoint_prior_scale=0.05 # Flexibilité de la tendance (0.05 par défaut)
        )
        
        # Entraînement
        model.fit(df_prophet)
        
        # Création du DataFrame pour la prédiction
        future = model.make_future_dataframe(periods=days_ahead)
        forecast = model.predict(future)
        
        # Récupération des données du dernier jour prédit
        last_prediction = forecast.iloc[-1]
        
        # Inversion de la transformation Logarithmique (np.exp) pour revenir aux prix réels
        pred_price = np.exp(last_prediction['yhat'])
        pred_low = np.exp(last_prediction['yhat_lower'])
        pred_high = np.exp(last_prediction['yhat_upper'])
        
        # Récupération du prix actuel (dernier prix connu)
        current_price = df_raw['Close'].iloc[-1]
        
        if current_price <= 0:
            return None, None, None, None
            
        # Calcul de l'évolution en %
        pct_change = ((pred_price - current_price) / current_price) * 100
        
        # Sécurité basique
        if pred_low <= 0: pred_low = 0.01

        return float(pred_price), float(pct_change), float(pred_low), float(pred_high)
        
    except Exception as e:
        print(f"Erreur calculate_prophet_prediction : {e}")
        return None, None, None, None

