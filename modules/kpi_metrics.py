# modules/kpi_metrics.py
import numpy as np
import pandas as pd
import yfinance as yf
import requests_cache

# Cache de court terme pour éviter de spammer Yahoo Finance pour le taux sans risque
session = requests_cache.CachedSession('yfinance.cache')
session.headers = {'User-agent': 'Mozilla/5.0'}

def get_dynamic_risk_free_rate(default_rate=0.03):
    """
    Récupère le rendement du bon du Trésor américain à 10 ans (^TNX).
    En cas d'erreur ou d'indisponibilité, retourne le taux par défaut.
    """
    try:
        # ^TNX est en pourcentage (ex: 4.2 pour 4.2%)
        ticker = yf.Ticker("^TNX", session=session)
        # On récupère les deux derniers jours pour être sûr d'avoir une valeur
        hist = ticker.history(period="5d")
        if not hist.empty:
            # On prend la dernière valeur de clôture et on divise par 100
            rate = hist['Close'].iloc[-1] / 100.0
            return rate
        return default_rate
    except Exception:
        return default_rate

def calculate_portfolio_kpis(portfolio_series, risk_free_rate=None):
    """
    Calcule les KPI de risque (Sharpe, Sortino, Max Drawdown, Volatilité)
    pour une série temporelle de la valeur du portefeuille.
    """
    # Si le taux sans risque n'est pas fourni, on tente de le récupérer dynamiquement
    if risk_free_rate is None:
        risk_free_rate = get_dynamic_risk_free_rate()

    # Filtrer les valeurs à 0 qui créent des rendements infinis lors de l'ajout d'un nouvel actif
    portfolio_series = portfolio_series[portfolio_series > 0]
    
    if portfolio_series is None or len(portfolio_series) < 2:
        return {"Sharpe": 0.0, "Sortino": 0.0, "Max Drawdown": 0.0, "Volatilité": 0.0, "Period Return": 0.0}

    # Calcul des rendements LOGARITHMIQUES au lieu des simples pourcentages (plus précis sur le long terme)
    # R(t) = ln(P(t) / P(t-1))
    daily_returns = np.log(portfolio_series / portfolio_series.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(daily_returns) < 2:
        return {"Sharpe": 0.0, "Sortino": 0.0, "Max Drawdown": 0.0, "Volatilité": 0.0, "Period Return": 0.0}

    # Volatilité quotidienne et annualisée
    daily_volatility = daily_returns.std()
    if pd.isna(daily_volatility):
        daily_volatility = 0.0
    volatility = daily_volatility * np.sqrt(252)

    sharpe_ratio = 0.0
    sortino_ratio = 0.0

    if daily_volatility >= 1e-4:  # Si la volatilité n'est pas anormalement basse
        daily_risk_free = risk_free_rate / 252
        # Excès de rendement moyen
        excess_daily_returns = daily_returns - daily_risk_free
        
        # --- Ratio de Sharpe ---
        daily_sharpe = excess_daily_returns.mean() / daily_volatility
        sharpe_ratio = daily_sharpe * np.sqrt(252)
        
        # --- Ratio de Sortino ---
        # Ne prend en compte que la volatilité des rendements NÉGATIFS
        downside_returns = excess_daily_returns[excess_daily_returns < 0]
        if not downside_returns.empty:
            downside_volatility = downside_returns.std()
            if downside_volatility >= 1e-4:
                daily_sortino = excess_daily_returns.mean() / downside_volatility
                sortino_ratio = daily_sortino * np.sqrt(252)
        
    # Borner les ratios pour éviter les chiffres extrêmes sur de très courtes périodes avec forte variance
    sharpe_ratio = max(min(sharpe_ratio, 10.0), -10.0)
    sortino_ratio = max(min(sortino_ratio, 15.0), -15.0)

    # Max Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100 # En pourcentage

    # Rendement sur la période (Plus pertinent que d'annualiser une donnée "court terme")
    period_return = (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]

    return {
        "Sharpe": sharpe_ratio,
        "Sortino": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Volatilité": volatility * 100,
        "Period Return": period_return * 100
    }
