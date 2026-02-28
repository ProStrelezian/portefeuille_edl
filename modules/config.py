# modules/config.py

DEFAULT_PORTFOLIO_CSV = """Nom de l'actif,Type d'actif,Valeur d'une unité,Unités,Gain de staking,Dividende,Date d'obtention,Frais,Total de l'actif,Date de vente,Prix de vente
NEAR Protocol (NEAR-EUR),Cryptomonnaie,"2,07 €","23,158471","0,21498651",,05/09/2025 08:57,"1,99 €","48,46 €",,
Argent (XAGUSD),CFD,"35,34 €","1,669436",,,05/09/2025 12:04,"1,00 €","59,00 €",28/11/2025 15:38,"77,31€"
SPDR STOXX Europe 600 SRI UCITS ETF EUR Unhedged (Acc) (ZPDX),ETF,"32,67 €","1,224552",,,05/09/2025 12:18,"0,00 €","40,01 €",,
Amundi PEA MSCI Emerging Asia ESG Leaders UCITS ETF - EUR (C/D) (18MB),ETF,"28,35 €","3,492063",,,11/09/2025 17:49,"1,00 €","99,00 €",,
Amundi MSCI New Energy ESG Screened UCITS ETF (Dist) (LYM9),ETF,"29,09 €","5,122035",,"0,71 €",11/09/2025 17:42,"1,00 €","149,71 €",,
Global X Uranium UCITS ETF AccumUSD (URNU),ETF,"24,73 €","8,046907",,,26/09/2025 10:04,"1,00 €","199,00 €",,
"""

TICKER_FIXES = {
    'ZPDX': 'ZPDX.DE',
    '18MB': 'PAASI.PA',
    'LYM9': 'NRJ.PA',
    'NEAR-EUR': 'NEAR-USD',
    'URNU': 'URNU.L',
}

CUSTOM_CSS = """
<style>
    /* Import de la police Overpass (Style Anilist) */
    @import url('https://fonts.googleapis.com/css2?family=Overpass:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Overpass', sans-serif;
        color: #bcbedc;
        background-color: #0b1622;
    }
    
    .stApp {
        background-color: #0b1622;
    }

    h1 {
        font-family: 'Overpass', sans-serif;
        color: #edf1f5;
        font-weight: 800 !important;
        padding-bottom: 15px;
        text-align: center;
        margin-bottom: 30px;
        text-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    h2 { font-family: 'Overpass', sans-serif; color: #edf1f5; font-weight: 700; margin-top: 2rem; margin-bottom: 1rem; }

    h3 {
        font-family: 'Overpass', sans-serif; color: #edf1f5; border-left: 5px solid #3db4f2;
        padding-left: 15px; margin-top: 3rem; margin-bottom: 1.5rem; font-weight: 700;
        background: linear-gradient(90deg, rgba(61, 180, 242, 0.1) 0%, transparent 100%);
        padding-top: 10px; padding-bottom: 10px; border-radius: 0 4px 4px 0;
    }
    
    h4 { font-family: 'Overpass', sans-serif; color: #bcbedc; font-weight: 600; margin-top: 1.5rem; margin-bottom: 0.5rem; }

    div[data-testid="stMetric"] {
        background-color: #151f2e; border-radius: 4px; padding: 20px;
        box-shadow: 0 14px 30px rgba(0,0,0,.1), 0 4px 4px rgba(0,0,0,.04); border: none;
        text-align: center; transition: transform 0.2s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-5px); box-shadow: 0 20px 40px rgba(0,0,0,.2); }
    div[data-testid="stMetricLabel"] { font-size: 0.9rem; color: #8ba0b2; font-weight: 600; justify-content: center; width: 100%; }
    div[data-testid="stMetricValue"] { font-family: 'Overpass', sans-serif; font-size: 1.6rem; font-weight: 700; color: #3db4f2; }

    div.stButton > button {
        background-color: #3db4f2; color: white; border: none; border-radius: 3px;
        font-weight: 700; font-family: 'Overpass', sans-serif; transition: all 0.2s ease;
    }
    div.stButton > button:hover { background-color: #62c3f5; color: white; box-shadow: 0 2px 10px rgba(61, 180, 242, 0.4); }

    div[data-testid="stTabs"] button[role="tab"] { color: #8ba0b2; font-weight: 600; font-family: 'Overpass', sans-serif; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color: #3db4f2; border-bottom-color: #3db4f2 !important; }
    
    @media (max-width: 768px) {
        .block-container { padding-top: 2rem !important; }
        h1 { font-size: 1.8rem !important; }
    }
</style>
"""
