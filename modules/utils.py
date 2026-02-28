import pandas as pd
import re
from .config import TICKER_FIXES

def clean_currency_series(series):
    """Version vectorisée de clean_currency pour un gain de performance massif."""
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0.0)
    
    # Gère les espaces, symboles monétaires, et la virgule comme séparateur décimal.
    cleaned = series.astype(str).str.replace(r'[€ \u202f]', '', regex=True).str.replace(',', '.')
    return pd.to_numeric(cleaned, errors='coerce').fillna(0.0)

def extract_ticker(name, saved_tickers=None):
    """
    Extrait un ticker potentiel depuis le nom de l'actif.
    Priorité : 1) Session state (config manuelle), 2) TICKER_FIXES, 3) Détection automatique.
    La logique parcourt les parenthèses de droite à gauche et prend le premier contenu
    qui ne soit pas un terme générique (comme 'ACC', 'DIST', etc.).
    Permet d'ignorer les mentions comme '(Acc)' pour trouver le vrai ticker comme '(ZPDX)'.
    """
    # Priorité 1 : configuration manuelle de l'utilisateur (session state)
    if saved_tickers and name in saved_tickers:
        return saved_tickers[name]

    matches = re.findall(r'\((.*?)\)', name)
    if matches:
        for match in reversed(matches):
            if match.upper() not in ['ACC', 'DIST', 'C/D', 'EUR', 'USD', 'HEDGED', 'UNHEDGED']:
                raw_ticker = match
                # Priorité 2 : corrections connues (TICKER_FIXES)
                return TICKER_FIXES.get(raw_ticker, raw_ticker)
        raw_ticker = matches[-1]
        return TICKER_FIXES.get(raw_ticker, raw_ticker)
    return None

def is_ticker_usd_heuristic(ticker):
    """
    Devine si un ticker est probablement coté en USD.
    - Vrai pour les cryptos (-USD), les taux de change (=X) ou les actions US (ex: 'AAPL').
    - Faux pour les actions européennes (ex: '.PA', '.DE').
    """
    if not isinstance(ticker, str) or not ticker:
        return False
    ticker = ticker.upper()
    if ticker.endswith("-USD") or ticker.endswith("=X"):
        return True
    if "-" not in ticker and "." not in ticker:
        return True
    return False
