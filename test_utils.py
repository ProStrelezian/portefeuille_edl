"""
Tests unitaires pour les fonctions utilitaires de TESTETATDESLIEUX_ML.py
Lancer avec : pytest tests/test_utils.py -v
"""
import pytest
import sys
import os

# Permet d'importer les fonctions depuis le fichier principal
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from modules.utils import clean_currency_series, extract_ticker, is_ticker_usd_heuristic
from modules.ml_models import calculate_smart_prediction

def clean_currency(value):
    s = pd.Series([value])
    return clean_currency_series(s).iloc[0]

# ──────────────────────────────────────────────
# clean_currency
# ──────────────────────────────────────────────

class TestCleanCurrency:
    def test_standard_euro(self):
        assert clean_currency("1 234,56 €") == 1234.56

    def test_standard_float(self):
        assert clean_currency(42.5) == 42.5

    def test_integer(self):
        assert clean_currency(100) == 100.0

    def test_empty_string(self):
        assert clean_currency("") == 0.0

    def test_none(self):
        assert clean_currency(None) == 0.0

    def test_nan(self):
        import math
        import numpy as np
        assert clean_currency(float("nan")) == 0.0

    def test_no_euro_symbol(self):
        assert clean_currency("2,07") == 2.07

    def test_narrow_no_break_space(self):
        # Format avec espace insécable fine (U+202F) utilisé par certaines locales françaises
        assert clean_currency("1\u202f000,00 €") == 1000.0

    def test_already_dot_decimal(self):
        assert clean_currency("59.00") == 59.0

    def test_invalid_string(self):
        assert clean_currency("N/A") == 0.0

    def test_zero_string(self):
        assert clean_currency("0,00 €") == 0.0


# ──────────────────────────────────────────────
# extract_ticker
# ──────────────────────────────────────────────

class TestExtractTicker:
    def test_simple_ticker(self):
        assert extract_ticker("Apple Inc (AAPL)") == "AAPL"

    def test_ticker_with_fix(self):
        # ZPDX doit être corrigé en ZPDX.DE via TICKER_FIXES
        result = extract_ticker("SPDR STOXX (ZPDX)")
        assert result == "ZPDX.DE"

    def test_ignore_acc_suffix(self):
        # (Acc) doit être ignoré, le vrai ticker est avant
        result = extract_ticker("iShares ETF (CSPX) (Acc)")
        assert result == "CSPX"

    def test_ignore_dist_suffix(self):
        result = extract_ticker("Amundi ETF (LYPS) (Dist)")
        assert result == "LYPS"

    def test_ignore_eur_suffix(self):
        result = extract_ticker("Amundi ETF (LYM9) (EUR) (C/D)")
        # LYM9 est dans TICKER_FIXES → NRJ.PA
        assert result == "NRJ.PA"

    def test_no_parentheses(self):
        assert extract_ticker("Bitcoin") is None

    def test_session_state_priority(self):
        """Le session state doit avoir la priorité absolue sur TICKER_FIXES et la détection auto."""
        saved = {"Mon ETF (ZPDX)": "ZPDX.PA"}
        result = extract_ticker("Mon ETF (ZPDX)", saved_tickers=saved)
        assert result == "ZPDX.PA"  # Pas ZPDX.DE (TICKER_FIXES)

    def test_session_state_overrides_auto_detection(self):
        saved = {"Apple (AAPL)": "AAPL.PA"}
        result = extract_ticker("Apple (AAPL)", saved_tickers=saved)
        assert result == "AAPL.PA"

    def test_no_session_state_fallback(self):
        result = extract_ticker("Apple (AAPL)", saved_tickers={})
        assert result == "AAPL"

    def test_near_eur_fix(self):
        result = extract_ticker("NEAR Protocol (NEAR-EUR)")
        assert result == "NEAR-USD"


# ──────────────────────────────────────────────
# is_ticker_usd_heuristic
# ──────────────────────────────────────────────

class TestIsTickerUsdHeuristic:
    def test_crypto_usd(self):
        assert is_ticker_usd_heuristic("BTC-USD") is True

    def test_fx_rate(self):
        assert is_ticker_usd_heuristic("EURUSD=X") is True

    def test_us_stock_no_suffix(self):
        assert is_ticker_usd_heuristic("AAPL") is True

    def test_european_stock_pa(self):
        assert is_ticker_usd_heuristic("MC.PA") is False

    def test_european_stock_de(self):
        assert is_ticker_usd_heuristic("SAP.DE") is False

    def test_london_stock(self):
        assert is_ticker_usd_heuristic("URNU.L") is False

    def test_empty_string(self):
        assert is_ticker_usd_heuristic("") is False

    def test_none(self):
        assert is_ticker_usd_heuristic(None) is False

    def test_crypto_eur(self):
        # NEAR-EUR est coté en EUR, pas USD
        assert is_ticker_usd_heuristic("NEAR-EUR") is False


# ──────────────────────────────────────────────
# calculate_smart_prediction (régression polynomiale)
# ──────────────────────────────────────────────

class TestCalculateSmartPrediction:
    def _make_prices(self, n=60, start=100.0, step=0.5):
        """Série de prix linéairement croissants."""
        return tuple(start + i * step for i in range(n))

    def test_returns_four_values(self):
        prices = self._make_prices()
        result = calculate_smart_prediction(prices, days_ahead=30)
        assert len(result) == 4

    def test_future_price_positive(self):
        prices = self._make_prices()
        future_price, _, low, high = calculate_smart_prediction(prices, days_ahead=30)
        assert future_price > 0
        assert low > 0
        assert high > 0

    def test_low_below_high(self):
        prices = self._make_prices()
        _, _, low, high = calculate_smart_prediction(prices, days_ahead=30)
        assert low < high

    def test_insufficient_data_returns_none(self):
        # Moins de 20 points → doit retourner None
        prices = tuple(range(10))
        result = calculate_smart_prediction(prices, days_ahead=30)
        assert result == (None, None, None, None)

    def test_empty_tuple_returns_none(self):
        result = calculate_smart_prediction((), days_ahead=30)
        assert result == (None, None, None, None)

    def test_pct_change_direction_uptrend(self):
        """Une tendance haussière claire doit produire un pct_change positif."""
        prices = tuple(float(100 + i) for i in range(60))
        _, pct_change, _, _ = calculate_smart_prediction(prices, days_ahead=30)
        assert pct_change > 0

    def test_pct_change_direction_downtrend(self):
        """Une tendance baissière claire doit produire un pct_change négatif."""
        prices = tuple(float(200 - i) for i in range(60))
        _, pct_change, _, _ = calculate_smart_prediction(prices, days_ahead=30)
        assert pct_change < 0

    def test_days_ahead_7_vs_30(self):
        """La projection à 30 jours doit aller plus loin que celle à 7 jours (tendance haussière)."""
        prices = tuple(float(100 + i) for i in range(60))
        price_30, _, _, _ = calculate_smart_prediction(prices, days_ahead=30)
        price_7, _, _, _ = calculate_smart_prediction(prices, days_ahead=7)
        assert price_30 > price_7
