# modules/kpi_metrics.py
import numpy as np
import pandas as pd

def calculate_portfolio_kpis(portfolio_series, risk_free_rate=0.03):
    """
    Calcule les KPI de risque pour une série temporelle de la valeur du portefeuille.
    """
    # Filtrer les valeurs à 0 qui créent des rendements infinis lors de l'ajout d'un nouvel actif
    portfolio_series = portfolio_series[portfolio_series > 0]
    
    if portfolio_series is None or len(portfolio_series) < 2:
        return {"Sharpe": 0.0, "Max Drawdown": 0.0, "Volatilité": 0.0, "Period Return": 0.0}

    # Calcul des rendements journaliers (en nettoyant les anomalies mathématiques)
    daily_returns = portfolio_series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(daily_returns) < 2:
        return {"Sharpe": 0.0, "Max Drawdown": 0.0, "Volatilité": 0.0, "Period Return": 0.0}

    # Volatilité quotidienne et annualisée
    daily_volatility = daily_returns.std()
    if pd.isna(daily_volatility):
        daily_volatility = 0.0
    volatility = daily_volatility * np.sqrt(252)

    # Sharpe Ratio classique
    if daily_volatility < 1e-4:  # Si la volatilité est anormalement basse (< 0.01% par jour)
        sharpe_ratio = 0.0
    else:
        daily_risk_free = risk_free_rate / 252
        daily_sharpe = (daily_returns.mean() - daily_risk_free) / daily_volatility
        sharpe_ratio = daily_sharpe * np.sqrt(252)
        
    # Borner le Sharpe pour éviter les chiffres extrêmes sur de courtes périodes
    sharpe_ratio = max(min(sharpe_ratio, 10.0), -10.0)

    # Max Drawdown
    cumulative_max = portfolio_series.cummax()
    drawdown = (portfolio_series - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100 # En pourcentage

    # Rendement sur la période (Plus pertinent que d'annualiser une donnée "court terme")
    period_return = (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]

    return {
        "Sharpe": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Volatilité": volatility * 100,
        "Period Return": period_return * 100
    }
