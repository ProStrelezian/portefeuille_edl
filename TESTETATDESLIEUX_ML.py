import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
import re
import yfinance as yf # API pour les données de marché
import time
import numpy as np # Nécessaire pour les calculs de prédiction
import requests_cache
from modules.ml_models import calculate_ml_prediction, calculate_smart_prediction, calculate_prophet_prediction
from modules.kpi_metrics import calculate_portfolio_kpis
from modules.config import CUSTOM_CSS, DEFAULT_PORTFOLIO_CSV, TICKER_FIXES
from modules.utils import clean_currency_series, extract_ticker, is_ticker_usd_heuristic

# Initialisation du système de cache SQLite pour yfinance, qui intercepte automatiquement toutes les requêtes HTTP (Requests).
# Adapté pour Streamlit Cloud (utilisation de tempfile pour éviter les problèmes de droits d'écriture et de lock SQLite).
import tempfile
import os
cache_path = os.path.join(tempfile.gettempdir(), 'portfolio_yf_cache')
requests_cache.install_cache(cache_path, expire_after=3600)


try:
    import financedatabase as fd
except ImportError:
    fd = None

# Configuration de la page
st.set_page_config(
    page_title="Portefeuille - État des Lieux",
    page_icon="📈",
    layout="wide"
)

# --- CUSTOM CSS FOR UI IMPROVEMENT ---
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- INITIALISATION DE LA MÉMOIRE (SESSION STATE) --- #
# Le Session State de Streamlit permet de conserver des données entre les rechargements de la page.
# C'est ici qu'on stocke les configurations de tickers et de devises de l'utilisateur.
if "saved_tickers" not in st.session_state:
    st.session_state.saved_tickers = {}
if "saved_currencies" not in st.session_state:
    st.session_state.saved_currencies = {}

# Configuration stockée dans modules/config.py

# --- FONCTIONS UTILITAIRES DEPORTEES DANS modules/utils.py ---

def add_technical_indicators(df):
    """Calcule les indicateurs techniques une seule fois pour le cache."""
    if df.empty: return df
    df = df.copy()
    close = df['Close']
    
    # MM & MME
    df['MM_200'] = close.rolling(window=200).mean()
    df['MME_9'] = close.ewm(span=9, adjust=False).mean()
    df['MME_21'] = close.ewm(span=21, adjust=False).mean()
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR
    if 'High' in df.columns and 'Low' in df.columns:
        high = df['High']
        low = df['Low']
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        df['Highest_50'] = df['High'].rolling(window=50).max()
        
        # AJOUT: Stochastic Oscillator (14)
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        denom = high_max - low_min
        # Gestion division par zéro et calcul %K
        df['Stoch_K'] = np.where(denom == 0, 50, 100 * ((close - low_min) / denom))
    else:
        df['ATR'] = np.nan
        df['Highest_50'] = np.nan
        df['Stoch_K'] = np.nan
        
    # AJOUT: Bandes de Bollinger (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    
    if 'Volume' in df.columns:
        df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
    return df

@st.cache_data(show_spinner=False, ttl=86400)
def search_ticker_in_db(query, category):
    """Recherche un ticker dans FinanceDatabase."""
    if fd is None:
        return pd.DataFrame({"Erreur": ["Module 'financedatabase' non installé."]})
    
    try:
        if category == "Actions":
            db = fd.Equities()
        elif category == "ETFs":
            db = fd.ETFs()
        elif category == "Cryptos":
            db = fd.Cryptos()
        elif category == "Indices":
            db = fd.Indices()
        elif category == "Devises":
            db = fd.Currencies()
        else:
            return pd.DataFrame()
            
        # Recherche (case_insensitive=True permet de trouver 'lvmh' même si c'est 'LVMH')
        res = db.search(name=query, case_insensitive=True)
        
        if res.empty:
            return pd.DataFrame()
            
        # Nettoyage pour affichage (Le ticker est souvent l'index dans FinanceDatabase)
        res = res.reset_index() 
        
        # Sélection des colonnes pertinentes selon ce qui est dispo
        cols = ['symbol', 'name', 'currency', 'country', 'sector', 'industry', 'category', 'market']
        final_cols = [c for c in cols if c in res.columns]
        
        return res[final_cols].head(20) # Top 20 résultats
    except Exception as e:
        return pd.DataFrame({"Erreur": [str(e)]})

def get_ticker_data(data_obj, tick, col='Close'):
    """Extrait proprement les données pour un ticker, peu importe le format renvoyé par yfinance."""
    try:
        if data_obj is None or data_obj.empty:
            return pd.Series(dtype=float)
        
        # Cas MultiIndex (plusieurs tickers) : Colonnes = (Ticker, OHLC)
        if isinstance(data_obj.columns, pd.MultiIndex):
            if tick in data_obj.columns.get_level_values(0):
                df_tick = data_obj[tick]
                if col in df_tick.columns:
                    return df_tick[col]
        
        # Cas Index Simple (un seul ticker ou structure plate)
        else:
            # Si on a demandé un seul ticker, data_obj est directement le DF de ce ticker
            if col in data_obj.columns:
                return data_obj[col]
    except Exception:
        pass
    return pd.Series(dtype=float)

# --- CACHE 1: HISTORIQUE LONG TERME (4h - Disque) ---
@st.cache_data(ttl=14400)
def fetch_historical_data(tickers):
    """Télécharge l'historique complet (2 ans) et intraday (1 mois) + Calculs lourds."""
    # Force cache invalidation for new indicators
    if not tickers:
        return {}, pd.DataFrame(), pd.DataFrame()
    try:
        valid_tickers = [t for t in tickers if t and isinstance(t, str)]
        if not valid_tickers:
            return {}, pd.DataFrame(), pd.DataFrame()

        tickers_to_fetch = list(set(valid_tickers + ["EURUSD=X", "GBPEUR=X", "CHFEUR=X"]))
        
        try:
            # Helper avec réessai pour yfinance qui est souvent instable
            def robust_download(tks, per, itv):
                for attempt in range(3):
                    try:
                        res = yf.download(tks, period=per, interval=itv, progress=False, auto_adjust=False, group_by='ticker', threads=True)
                        if res is not None and not res.empty:
                            return res
                    except Exception as e:
                        if attempt == 2:
                            print(f"Erreur yf.download finale pour {tks}: {e}")
                    time.sleep(1 + attempt)
                return pd.DataFrame()

            # 1. Données journalières COMPLÈTES (OHLCV) pour FinRL (2 ans pour MM200)
            data_daily_full = robust_download(tickers_to_fetch, "2y", "1d")

            # 2. Données intraday (5 min) pour le graphique détaillé (1 mois pour assurer 7 jours)
            data_intraday = robust_download(tickers_to_fetch, "1mo", "5m")
            
        except Exception as e:
            st.error(f"Erreur de connexion Yahoo Finance : {e}. Réessayez plus tard.")
            return {}, pd.DataFrame(), pd.DataFrame()
        
        full_ticker_data = {}
        data_daily_close = pd.DataFrame()
        
        for ticker in valid_tickers:
            # Construction du DataFrame complet pour FinRL (OHLCV)
            try:
                # Extraction du DF complet pour le ticker
                if not data_daily_full.empty and isinstance(data_daily_full.columns, pd.MultiIndex) and ticker in data_daily_full.columns.get_level_values(0):
                    df_t = data_daily_full[ticker].copy()
                else:
                    # Cas mono-ticker
                    df_t = data_daily_full.copy()
                
                # Vérification des colonnes requises
                req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(c in df_t.columns for c in req_cols):
                    # On remplit les jours fériés avec la dernière valeur connue AVANT de calculer les indicateurs.
                    # Cela préserve la continuité de l'index temporel.
                    full_ticker_data[ticker] = add_technical_indicators(df_t.ffill())
                else:
                    full_ticker_data[ticker] = pd.DataFrame()
            except Exception:
                full_ticker_data[ticker] = pd.DataFrame()

        # Reconstruction d'un DataFrame propre pour l'analyse de corrélation
        for ticker in valid_tickers:
            s = get_ticker_data(data_daily_full, ticker, 'Close')
            if not s.empty:
                data_daily_close[ticker] = s

        # Appliquer ffill() puis bfill() pour une robustesse maximale.
        # 1. ffill() propage la dernière valeur connue (gère les jours fériés/weekends).
        # 2. bfill() remplit les éventuels NaN au début si un actif a un historique plus court.
        if not data_daily_close.empty:
            data_daily_close = data_daily_close.ffill().bfill()

        return full_ticker_data, data_intraday, data_daily_close

    except Exception as e:
        print(f"Erreur Hist: {e}")
        return {}, pd.DataFrame(), pd.DataFrame()

# --- CACHE 2: PRIX TEMPS RÉEL (1 min - Mémoire) ---
@st.cache_data(ttl=60)
def fetch_real_time_data(tickers):
    """Télécharge uniquement le dernier prix (très rapide)."""
    if not tickers: return pd.DataFrame()
    valid_tickers = [t for t in tickers if t and isinstance(t, str)]
    tickers_to_fetch = list(set(valid_tickers + ["EURUSD=X", "GBPEUR=X", "CHFEUR=X"]))
    
    try:
        # Essais multiples pour la robustesse du temps réel
        for attempt in range(3):
            try:
                data_live = yf.download(tickers_to_fetch, period="1d", interval="1m", progress=False, auto_adjust=False, group_by='ticker', threads=True)
                if data_live is not None and not data_live.empty:
                    return data_live
            except Exception:
                pass
            time.sleep(0.5)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_market_data(tickers):
    """Orchestrateur : Combine historique (cache long) et temps réel (cache court)."""
    if not tickers:
        return {}, {}, {'USD': 1.0, 'GBP': 1.0, 'CHF': 1.0}, {}, pd.DataFrame(), pd.DataFrame(), {}
    
    # 1. Récupération des données
    full_ticker_data, data_intraday, data_daily_close = fetch_historical_data(tickers)
    data_live = fetch_real_time_data(tickers)
    
    valid_tickers = [t for t in tickers if t and isinstance(t, str)]
    
    current_prices = {}
    reference_prices = {}
    history_data = {}
    exchange_rates = {'USD': 1.0, 'GBP': 1.0, 'CHF': 1.0}
    
    # 2. Extraction des taux de change (Live > Intraday > Daily)
    def extract_rate(ticker):
        rate_series = get_ticker_data(data_live, ticker, 'Close')
        if rate_series.empty:
            rate_series = get_ticker_data(data_intraday, ticker, 'Close')
        if not rate_series.empty:
            valid_rate = rate_series.dropna()
            if not valid_rate.empty:
                r = float(valid_rate.iloc[-1])
                if r > 0: return r
        return 1.0

    exchange_rates['USD'] = extract_rate("EURUSD=X")
    exchange_rates['GBP'] = extract_rate("GBPEUR=X")
    exchange_rates['CHF'] = extract_rate("CHFEUR=X")

    # 3. Construction des prix et références
    for ticker in valid_tickers:
        # Récupération des DataFrames disponibles
        df_hist = full_ticker_data.get(ticker, pd.DataFrame())
        
        # Prix Actuel : Live > Intraday > Daily (Hist)
        price_live = get_ticker_data(data_live, ticker, 'Close').dropna()
        price_intra = get_ticker_data(data_intraday, ticker, 'Close').dropna()
        
        if not price_live.empty:
            current_prices[ticker] = float(price_live.iloc[-1])
        elif not price_intra.empty:
            current_prices[ticker] = float(price_intra.iloc[-1])
        elif not df_hist.empty:
            current_prices[ticker] = float(df_hist['Close'].iloc[-1])
        else:
            current_prices[ticker] = 0.0
        
        # Historique et Référence (Basés sur Daily/Hist pour cohérence)
        if not df_hist.empty:
            # Restriction à 3 mois (~90 jours calendaires) pour correspondre à "Tendance (3 mois)"
            # et pour que la polynomiale (PolyFit) modélise la tendance récente, pas l'historique de 2 ans.
            cutoff_date = df_hist.index.max() - pd.Timedelta(days=90)
            history_data[ticker] = df_hist.loc[df_hist.index >= cutoff_date, 'Close'].tolist()
            # Reference = Clôture veille (avant-dernière valeur si la dernière est aujourd'hui, ou dernière si data pas à jour)
            # Simplification : on prend l'avant-dernière valeur du daily
            if len(df_hist) > 1:
                reference_prices[ticker] = float(df_hist['Close'].iloc[-2])
            else:
                reference_prices[ticker] = float(df_hist['Close'].iloc[-1])
        else:
            reference_prices[ticker] = 0.0
            history_data[ticker] = []
            
    return current_prices, reference_prices, exchange_rates, history_data, data_intraday, data_daily_close, full_ticker_data

# --- CACHE 2: AVIS ANALYSTES (Long terme: 24h) ---
@st.cache_data(ttl=86400)
def fetch_asset_details(tickers):
    """Récupère les recommandations et le Type d'actif."""
    details = {}
    if not tickers:
        return {}

    valid_tickers = [t for t in tickers if t and isinstance(t, str)]
    
    for ticker in valid_tickers:
        details[ticker] = {'Avis': "N/A", 'Type': "N/A"}
        for attempt in range(2):
            try:
                t = yf.Ticker(ticker)
                info = t.info
                
                # Recommandation
                rec = info.get('recommendationKey', 'N/A')
                translations = {
                    'buy': 'Acheter 🟢', 'strong_buy': 'Achat Fort 🟢🟢',
                    'hold': 'Conserver 🟡', 'sell': 'Vendre 🔴',
                    'strong_sell': 'Vente Forte 🔴🔴', 'none': 'N/A', 'N/A': 'N/A'
                }
                rec_text = translations.get(rec, rec.capitalize())
                
                details[ticker] = {
                    'Avis': rec_text,
                    'Type': info.get('quoteType', 'N/A')
                }
                break # Succès, on sort de la boucle d'essai
            except Exception:
                time.sleep(0.5)
            
    return details

def process_portfolio_data(df, saved_tickers=None):
    """Nettoie et formate les données brutes du portefeuille."""
    if df is None or df.empty:
        return None
    try:
        # Conversion des colonnes monétaires et de dates dans les bons formats.
        money_cols = ["Valeur d'une unité", "Total de l'actif", "Frais", "Gain de staking", "Dividende", "Intérêts", "Prix de vente", "Unités"]
        for col in money_cols:
            if col in df.columns:
                df[col] = clean_currency_series(df[col])
        date_cols = ["Date d'obtention", "Date de vente"]
        # 'dayfirst=True' pour interpréter correctement le format JJ/MM/AAAA.
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
        if "Nom de l'actif" in df.columns:
            df['Ticker'] = df["Nom de l'actif"].apply(lambda n: extract_ticker(n, saved_tickers))
        return df
    except Exception as e:
        st.error(f"Erreur traitement données : {e}")
        return None

def load_data(file_input, saved_tickers=None):
    """Charge les données depuis un fichier CSV ou Excel ou une chaîne."""
    try:
        # Check if file_input is a string (StringIO) or an UploadedFile
        if isinstance(file_input, str) or hasattr(file_input, 'getvalue') and isinstance(file_input.getvalue(), str):
            df = pd.read_csv(file_input)
        elif hasattr(file_input, 'name'):
            # Detect file type by extension
            file_name = file_input.name.lower()
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_input)
            elif file_name.endswith(('.xls', '.xlsx', '.ods')):
                df = pd.read_excel(file_input)
            else:
                st.error("Format de fichier non supporté. Veuillez utiliser CSV, XLS, XLSX ou ODS.")
                return None
        else:
            # Fallback for StringIO
            df = pd.read_csv(file_input)
        
        return process_portfolio_data(df, saved_tickers)
    except Exception as e:
        st.error(f"Erreur lecture fichier : {e}")
        return None

@st.cache_data(ttl=600)
def load_data_from_gsheet(url, saved_tickers_json="{}"):
    """Charge les données depuis un Google Sheet public via export CSV."""
    import json
    saved_tickers = json.loads(saved_tickers_json)
    try:
        # Extraction de l'ID et construction de l'URL d'export CSV (Regex plus permissive)
        pattern = r"/spreadsheets/d/([a-zA-Z0-9-_]+)"
        match = re.search(pattern, url)
        if match:
            sheet_id = match.group(1)
            export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            # Gestion des onglets spécifiques (gid)
            gid_match = re.search(r"[#&]gid=([0-9]+)", url)
            if gid_match:
                export_url += f"&gid={gid_match.group(1)}"
            
            df = pd.read_csv(export_url)
            return process_portfolio_data(df, saved_tickers)
        return None
    except Exception:
        return None

# --- INTERFACE PRINCIPALE ---

# Titre avec span pour éviter que l'emoji ne soit affecté par le dégradé de texte transparent
st.markdown("# Portefeuille • État des Lieux 💰", unsafe_allow_html=True)
st.caption(f"Dernière actualisation : {pd.Timestamp.now(tz='Europe/Paris').strftime('%H:%M:%S')}")

# --- BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.header("Importation du portefeuille")
    
    # Sélecteur de source
    source_mode = st.radio("Source des données", ["Google Sheet (Public)", "Fichier Local (CSV/Excel)"], label_visibility="collapsed")
    
    uploaded_file = None
    gsheet_url = ""
    if source_mode == "Fichier Local (CSV/Excel)":
        uploaded_file = st.file_uploader("📂 Chargez votre fichier (CSV ou Excel)", type=["csv", "xls", "xlsx", "ods"])
        st.caption("Le fichier doit contenir au moins: `Nom de l'actif`, `Unités`, `Valeur d'une unité`, `Total de l'actif`.")
        if uploaded_file is None:
            st.info("Utilisation des données 'Placeholder' par défaut.")
        else:
            st.success("Fichier chargé !")
    else:
        st.markdown("Collez le lien de votre Google Sheet (Accès 'Tous les utilisateurs disposant du lien').")
        # Tentative de récupération auto depuis secrets
        default_url = "https://docs.google.com/spreadsheets/d/1MtRBv8XF-i6d43XqMLtyLIDWfZp8fPomWUBRzf5sfqQ/edit?usp=sharing"
        try:
            secret_url = st.secrets.get("public_gsheet_url", "")
            if secret_url:
                default_url = secret_url
        except Exception:
            pass
        gsheet_url = st.text_input("URL Google Sheet", value=default_url, placeholder="https://docs.google.com/.../edit?usp=sharing")
        if gsheet_url:
            st.caption("✅ URL détectée")
    
    st.markdown("---")
    st.header("🧭 Navigation")
    app_page = st.radio(
        "Choisissez une section :",
        ["📊 Tableau de Bord", "📈 Performance & Prévisions", "🧠 Analyse Technique & Risques", "💡 Signaux & Opportunités", "⚙️ Configuration & Archives"],
        label_visibility="collapsed"
    )

    # Vidage du cache si changement de page détecté
    if "current_page" not in st.session_state:
        st.session_state.current_page = app_page
    
    if st.session_state.current_page != app_page:
        st.session_state.current_page = app_page
        # Cache conservé entre les pages pour la persistance
    
    st.markdown("---")
    
    # Interrupteur pour le rafraîchissement automatique des données.
    st.header("⏱️ Rafraîchissement")
    
    if st.button("🔄 Actualiser maintenant", width='stretch'):
        st.cache_data.clear()
        st.rerun()
    
    auto_refresh = st.toggle("Mode Auto", value=False)
    refresh_interval = 30
    if auto_refresh:
        refresh_interval = st.slider("Intervalle (sec)", 30, 300, 30, 30)
        st.caption(f"⚠️ Rechargement auto toutes les {refresh_interval}s.")

    # Placeholder for the countdown timer, will be populated by the refresh logic
    countdown_placeholder = st.empty()

# --- CHARGEMENT ET PRÉPARATION DES DONNÉES ---
df = None
if source_mode == "Google Sheet (Public)" and gsheet_url:
    with st.spinner("Connexion au Google Sheet..."):
        import json
        df = load_data_from_gsheet(gsheet_url, json.dumps(st.session_state.saved_tickers))
        if df is None:
            st.error("Erreur : Impossible de lire le Google Sheet. Vérifiez qu'il est public (Lecture seule).")
elif uploaded_file is not None:
    df = load_data(uploaded_file, st.session_state.saved_tickers)

if df is None:
    df = load_data(StringIO(DEFAULT_PORTFOLIO_CSV), st.session_state.saved_tickers)

if df is not None:
    is_sold = (df["Date de vente"].notna()) | (df["Prix de vente"] > 0)
    df_sold = df[is_sold].copy()  # Actifs vendus
    df_hold = df[~is_sold].copy() # Actifs actuellement détenus

    # Mise à jour des tickers avec la configuration sauvegardée (si existante)
    if not df_hold.empty and st.session_state.saved_tickers:
        df_hold["Ticker"] = df_hold.apply(lambda x: st.session_state.saved_tickers.get(x["Nom de l'actif"], x["Ticker"]), axis=1)

    # --- TÉLÉCHARGEMENT DES DONNÉES DE MARCHÉ ---
    # Cette section s'exécute après la configuration pour utiliser les bons tickers.
    if not df_hold.empty:
        unique_tickers = [t for t in df_hold['Ticker'].unique() if t]
        
        if unique_tickers:
            if not auto_refresh:
                with st.spinner('Analyse des marchés en cours...'):
                    market_prices, ref_prices, exchange_rates, history_data, raw_history_df, daily_history_df, full_ticker_data = fetch_market_data(unique_tickers)
                    asset_details = fetch_asset_details(unique_tickers)
            else:
                market_prices, ref_prices, exchange_rates, history_data, raw_history_df, daily_history_df, full_ticker_data = fetch_market_data(unique_tickers)
                asset_details = fetch_asset_details(unique_tickers)
            
            # --- AJOUT: Alerte Jours Fériés / Week-end ---
            if not daily_history_df.empty:
                # On récupère la date la plus récente parmi toutes les données récupérées
                last_fetch_date = daily_history_df.index.max()
                
                # Gestion des Timezones pour éviter les erreurs de comparaison
                if last_fetch_date.tzinfo is not None:
                    last_fetch_date = last_fetch_date.tz_convert(None)
                
                # Calcul du nombre de jours depuis la dernière donnée
                days_diff = (pd.Timestamp.now() - last_fetch_date).days
                
                # Si les données ont plus de 2 jours (Week-end standard = 2 jours max de creux, donc >2 = Jours Fériés ou Cache)
                if days_diff > 2:
                    st.info(
                        f"📅 **Info Dates** : Les dernières données boursières remontent au **{last_fetch_date.strftime('%d/%m/%Y')}**.\n\n"
                        "C'est normal si les marchés étaient fermés récemment (**Week-end** ou **Jours Fériés** comme le *Presidents' Day* aux US). "
                        "Les cours se mettront à jour automatiquement à la prochaine ouverture."
                    )
            # ---------------------------------------------
        else:
            market_prices, ref_prices, exchange_rates, history_data, asset_details, raw_history_df, daily_history_df, full_ticker_data = {}, {}, {'USD': 1.0, 'GBP': 1.0, 'CHF': 1.0}, {}, {}, pd.DataFrame(), pd.DataFrame(), {}
        
        st.sidebar.markdown("---")
        if exchange_rates['USD'] != 1.0:
            st.sidebar.metric("Taux (1€ = $)", f"{exchange_rates['USD']:.4f} $")
        if exchange_rates['GBP'] != 1.0:
            st.sidebar.metric("Taux (1£ = €)", f"{exchange_rates['GBP']:.4f} €")
        if exchange_rates['CHF'] != 1.0:
            st.sidebar.metric("Taux (1 CHF = €)", f"{exchange_rates['CHF']:.4f} €")

        # --- ENRICHISSEMENT DU DATAFRAME AVEC LES DONNÉES DE MARCHÉ ---
        def get_row_currency(asset_name, ticker):
            """Détermine la devise pour une ligne : priorité à la config manuelle, sinon heuristique."""
            # Priorité absolue à la configuration manuelle
            if asset_name in st.session_state.saved_currencies:
                return st.session_state.saved_currencies[asset_name]
            # Sinon heuristique
            return "USD" if is_ticker_usd_heuristic(ticker) else "EUR"

        df_hold['Devise'] = df_hold.apply(lambda x: get_row_currency(x["Nom de l'actif"], x['Ticker']), axis=1)

        def get_converted_price(price_dict, ticker, currency, rates):
            """Récupère un prix et le convertit en EUR si nécessaire."""
            raw_price = price_dict.get(ticker, 0.0)
            
            if raw_price == 0.0: return 0.0
            
            if currency == "USD":
                return raw_price / rates['USD']
            elif currency == "GBP":
                return raw_price * rates['GBP']
            elif currency == "CHF":
                return raw_price * rates['CHF']
            return raw_price
        
        def get_history(ticker):
            return history_data.get(ticker, [])
        
        def get_weekly_evolution(ticker):
            if daily_history_df.empty or ticker not in daily_history_df.columns:
                return None
            series = daily_history_df[ticker].dropna()
            if series.empty:
                return None
            
            last_date = series.index[-1]
            last_price = series.iloc[-1]
            
            # Recul exact de 7 jours calendaires (au lieu de 5 index car crypto = 7j/7)
            target_date = last_date - pd.Timedelta(days=7)
            past_series = series[series.index <= target_date]
            
            if past_series.empty:
                # Fallback sur la première s'il y a trop peu d'historique
                prev_price = series.iloc[0]
            else:
                prev_price = past_series.iloc[-1]
                
            if prev_price == 0: return 0.0
            return ((last_price - prev_price) / prev_price) * 100

        def get_prediction_display(ticker):
            hist = history_data.get(ticker, [])
            hist_tuple = tuple(hist) # Conversion en tuple pour le cache
            if not hist_tuple: return (None, None, None, None, None, None, None, None, None)
            
            pred_price, pct_change, pred_low, pred_high = calculate_smart_prediction(hist_tuple, days_ahead=30)
            
            _, pct_change_7, pred_low_7, pred_high_7 = calculate_smart_prediction(hist_tuple, days_ahead=7)
            
            if pct_change is None: return (None, None, None, None, None, None, None, None, None)
            
            diff_7d = None
            diff_low_7d = None
            diff_high_7d = None
            
            # Recul basés sur 7 jours calendaires plutôt qu'un nombre d'index fixe (-5)
            if not daily_history_df.empty and ticker in daily_history_df.columns:
                series = daily_history_df[ticker].dropna()
                if not series.empty:
                    target_date = series.index[-1] - pd.Timedelta(days=7)
                    past_series = series[series.index <= target_date]
                    if len(past_series) > 20:
                        past_hist_tuple = tuple(past_series.tolist())
                        past_pred, _, past_low, past_high = calculate_smart_prediction(past_hist_tuple, days_ahead=30)
                        if past_pred is not None and pred_price is not None:
                            diff_7d = pred_price - past_pred
                            diff_low_7d = pred_low - past_low if pred_low is not None and past_low is not None else None
                            diff_high_7d = pred_high - past_high if pred_high is not None and past_high is not None else None
            
            return (pct_change, pct_change_7, pred_low, pred_high, pred_low_7, pred_high_7, diff_7d, diff_low_7d, diff_high_7d)

        def get_details(ticker, key):
            return asset_details.get(ticker, {}).get(key, None)

        def get_trend_7j(ticker):
            # Tendance exacte sur les 7 derniers jours calendaires
            if daily_history_df is None or daily_history_df.empty or ticker not in daily_history_df.columns:
                return []
            series = daily_history_df[ticker].dropna()
            if series.empty:
                return []
            cutoff_date = series.index[-1] - pd.Timedelta(days=7)
            trend_series = series[series.index > cutoff_date]
            return trend_series.tolist()

        def get_ml_prediction_display(ticker):
            df_t = full_ticker_data.get(ticker, pd.DataFrame())
            required_cols = ['Close', 'High', 'Low', 'Volume']
            if df_t is None or df_t.empty or not all(c in df_t.columns for c in required_cols):
                return (None, None, None, None)
            
            # Création d'un tuple hashable pour la mise en cache
            # IMPORT: index=True est critique ici car Prophet et le nouveau XGBoost ont besoin des dates (l'index de df_t).
            # Cela crée un tuple où le premier élément de chaque ligne est l'index (la date).
            data_tuple = tuple(df_t[required_cols].reset_index().itertuples(index=False, name=None))
            
            # XGBoost
            _pred_price_30_ml, pct_change_30_ml = calculate_ml_prediction(data_tuple, days_ahead=30)
            _pred_price_7_ml, pct_change_7_ml = calculate_ml_prediction(data_tuple, days_ahead=7)
            
            # Prophet
            _pred_price_30_p, pct_change_30_p, _, _ = calculate_prophet_prediction(data_tuple, days_ahead=30)
            _pred_price_7_p, pct_change_7_p, _, _ = calculate_prophet_prediction(data_tuple, days_ahead=7)
            
            return (pct_change_30_ml, pct_change_7_ml, pct_change_30_p, pct_change_7_p)

        def get_technical_indicators(ticker):
            df_t = full_ticker_data.get(ticker, pd.DataFrame())
            
            # --- FIX ROBUSTESSE CACHE ---
            # Si les nouvelles colonnes sont absentes à cause d'un cache obsolète, on recalcule.
            required_techs = ['BB_Upper', 'Stoch_K', 'MACD_Signal', 'Highest_50']
            if df_t is not None and not df_t.empty and not all(c in df_t.columns for c in required_techs):
                df_t = add_technical_indicators(df_t)
                full_ticker_data[ticker] = df_t
            
            if df_t is None or df_t.empty or 'MM_200' not in df_t.columns:
                return {}
            
            return df_t.iloc[-1].to_dict()

        # --- CALCUL PRIX 30 MIN ---
        def get_price_30m_ago(ticker):
            try:
                if raw_history_df is None or raw_history_df.empty: return 0.0
                s = get_ticker_data(raw_history_df, ticker, 'Close').dropna()
                if s.empty: return 0.0
                target = s.index[-1] - pd.Timedelta(minutes=30)
                past = s[s.index <= target]
                if past.empty: return float(s.iloc[0])
                return float(past.iloc[-1])
            except Exception:
                return 0.0
                
        prices_30m = {t: get_price_30m_ago(t) for t in df_hold['Ticker'].unique() if pd.notna(t)}

        # --- ENRICHISSEMENT EN UNE SEULE PASSE ---
        # On parcourt df_hold une seule fois pour calculer toutes les colonnes dérivées,
        # au lieu de faire 10+ .apply() successifs qui repassent chaque fois sur tout le DataFrame.
        enriched_rows = []
        for row in df_hold.to_dict('records'):
            ticker = row['Ticker']
            asset_name = row["Nom de l'actif"]
            devise = get_row_currency(asset_name, ticker)

            # Prix convertis
            prix_actuel = get_converted_price(market_prices, ticker, devise, exchange_rates)
            prix_ref = get_converted_price(ref_prices, ticker, devise, exchange_rates)
            prix_30m = get_converted_price(prices_30m, ticker, devise, exchange_rates)

            # Historique & évolution
            historique = history_data.get(ticker, [])
            evol_hebdo = get_weekly_evolution(ticker)
            trend_7j = get_trend_7j(ticker)

            # Détails fondamentaux
            avis = asset_details.get(ticker, {}).get('Avis', None)
            type_actif = asset_details.get(ticker, {}).get('Type', None)

            # Prédictions polynomiales
            pred_vals = get_prediction_display(ticker)
            pct_30, pct_7, bas_30, haut_30, bas_7, haut_7, evol_7d, evol_bas_7d, evol_haut_7d = pred_vals

            # Conversion des fourchettes de prix en EUR
            if devise in ['USD', 'GBP', 'CHF']:
                def _convert_range(val):
                    if val is None: return None
                    if devise == 'USD': return val / exchange_rates['USD']
                    if devise == 'GBP': return val * exchange_rates['GBP']
                    if devise == 'CHF': return val * exchange_rates['CHF']
                    return val
                    
                bas_30 = _convert_range(bas_30)
                haut_30 = _convert_range(haut_30)
                bas_7 = _convert_range(bas_7)
                haut_7 = _convert_range(haut_7)

            # Prédictions ML & Prophet
            ml_pct_30, ml_pct_7, proph_pct_30, proph_pct_7 = get_ml_prediction_display(ticker)

            # Indicateurs techniques
            techs = get_technical_indicators(ticker)
            mm200 = techs.get('MM_200')
            mme9 = techs.get('MME_9')
            mme21 = techs.get('MME_21')
            macd = techs.get('MACD')
            macd_signal = techs.get('MACD_Signal')
            atr = techs.get('ATR')
            highest_50 = techs.get('Highest_50')
            bb_haut = techs.get('BB_Upper')
            bb_bas = techs.get('BB_Lower')
            stoch_k = techs.get('Stoch_K')
            vol = techs.get('Volume')
            vol_sma_20 = techs.get('Vol_SMA_20')
            
            # Conversion EUR
            if devise in ['USD', 'GBP', 'CHF']:
                def _convert_ind(v):
                    if pd.isna(v) or v is None: return None
                    if devise == 'USD': return v / exchange_rates['USD']
                    if devise == 'GBP': return v * exchange_rates['GBP']
                    if devise == 'CHF': return v * exchange_rates['CHF']
                    return v
                    
                mm200, mme9, mme21, macd, macd_signal, atr, highest_50, bb_haut, bb_bas = (
                    _convert_ind(mm200), _convert_ind(mme9), _convert_ind(mme21), 
                    _convert_ind(macd), _convert_ind(macd_signal),
                    _convert_ind(atr), _convert_ind(highest_50), 
                    _convert_ind(bb_haut), _convert_ind(bb_bas)
                )

            enriched_rows.append({
                'Devise': devise,
                'Prix Actuel': prix_actuel,
                'Prix Reference': prix_ref,
                'Prix 30m': prix_30m,
                'Historique': historique,
                'Evol. Hebdo %': evol_hebdo,
                'Trend 7j': trend_7j,
                'Avis Analyste': avis,
                'Type': type_actif,
                'Proj. 30j (%)': pct_30,
                'Proj. 7j (%)': pct_7,
                'Proj. 30j Bas': bas_30,
                'Proj. 30j Haut': haut_30,
                'Proj. 7j Bas': bas_7,
                'Proj. 7j Haut': haut_7,
                'Evol. 7j': evol_7d,
                'Evol. Bas 7j': evol_bas_7d,
                'Evol. Haut 7j': evol_haut_7d,
                'Proj. 30j (ML)': ml_pct_30,
                'Proj. 7j (ML)': ml_pct_7,
                'Proj. 30j (Prophet)': proph_pct_30,
                'Proj. 7j (Prophet)': proph_pct_7,
                'MM 200': mm200,
                'MME 9': mme9,
                'MME 21': mme21,
                'MACD': macd,
                'MACD Signal': macd_signal,
                'ATR': atr,
                'Highest 50': highest_50,
                'BB Haut': bb_haut,
                'BB Bas': bb_bas,
                'Stoch K': stoch_k,
                'Volume': vol,
                'Vol SMA 20': vol_sma_20,
            })

        df_enriched = pd.DataFrame(enriched_rows, index=df_hold.index)
        # On supprime de df_hold les colonnes qui vont être ajoutées par df_enriched
        # pour éviter les doublons (ex: 'Devise' déjà présente) qui causent DuplicateError.
        cols_to_drop = [c for c in df_enriched.columns if c in df_hold.columns]
        df_hold = df_hold.drop(columns=cols_to_drop)
        df_hold = pd.concat([df_hold, df_enriched], axis=1)

        # Conversion explicite en numérique pour gérer les None (qui deviennent NaN)
        cols_tech = ['MM 200', 'MME 9', 'MME 21', 'MACD', 'MACD Signal', 'ATR', 'Highest 50', 'BB Haut', 'BB Bas', 'Stoch K', 'Volume', 'Vol SMA 20']
        for col in cols_tech:
            df_hold[col] = pd.to_numeric(df_hold[col], errors='coerce')
        
        # Calcul du Signal Technique (vectorisé)
        # On définit des conditions plus prioritaires d'abord
        conditions = [
            (df_hold['Prix Actuel'] < df_hold['Highest 50'] - 2.5 * df_hold['ATR']), # Stop ATR Touché
            ((df_hold['Prix Actuel'] - df_hold['MM 200']) / df_hold['MM 200'] > 0.35), # Elastique MM200
            ((df_hold['Volume'] > 3 * df_hold['Vol SMA 20']) & (df_hold['Prix Actuel'] > df_hold['MME 21'])), # Climax d'Achat
            ((df_hold['MACD'] > 0) & (df_hold['MACD Signal'] > 0) & (df_hold['MACD'] < df_hold['MACD Signal'])), # MACD Baissier
            (df_hold['Prix Actuel'] < df_hold['BB Bas']), # Prix sous la bande basse -> Rebond possible
            (df_hold['Prix Actuel'] > df_hold['BB Haut']), # Prix sur la bande haute -> Correction possible
            (df_hold['MME 9'] > df_hold['MME 21']),
            (df_hold['MME 9'] < df_hold['MME 21'])
        ]
        choices = [
            "Stop ATR 🔴", "Surchauffe MM200 🔴", "Climax Achat ⚠️", "MACD Baisse 🔴",
            "Sursell (BB) 🟢", "Surchauffe (BB) 🔴", "Achat (MME) 🟢", "Vente (MME) 🔴"
        ]
        df_hold['Signal Technique'] = np.select(conditions, choices, default="N/A")
        
        # Calculs de valeurs (vectorisés)
        df_hold['Valeur Actuelle'] = np.where(df_hold['Prix Actuel'] > 0, df_hold['Unités'] * df_hold['Prix Actuel'], df_hold["Total de l'actif"])
        df_hold['Valeur Reference'] = np.where(df_hold['Prix Reference'] > 0, df_hold['Unités'] * df_hold['Prix Reference'], df_hold["Total de l'actif"])
        df_hold['Valeur 30m'] = np.where(df_hold['Prix 30m'] > 0, df_hold['Unités'] * df_hold['Prix 30m'], df_hold["Total de l'actif"])
        
        # Calcul de l'évolution journalière (vectorisé)
        df_hold['Evol. Jour %'] = np.where(
            df_hold['Valeur Reference'] > 0,
            (df_hold['Valeur Actuelle'] - df_hold['Valeur Reference']) / df_hold['Valeur Reference'] * 100,
            0.0
        )
        
        # Séparation de la Plus-value Latente et des Gains Annexes
        staking_col = df_hold['Gain de staking'].fillna(0) if 'Gain de staking' in df_hold.columns else 0
        div_col = df_hold['Dividende'].fillna(0) if 'Dividende' in df_hold.columns else 0
        int_col = df_hold['Intérêts'].fillna(0) if 'Intérêts' in df_hold.columns else 0
        
        df_hold['Gains Annexes'] = staking_col + div_col + int_col
        df_hold['Plus-value Latente'] = df_hold['Valeur Actuelle'] - df_hold["Total de l'actif"]
        df_hold['P&L Total'] = df_hold['Plus-value Latente'] + df_hold['Gains Annexes']
        
        # Calcul de la performance (vectorisé)
        df_hold['Performance %'] = np.where(
            df_hold["Total de l'actif"] > 0,
            (df_hold['P&L Total'] / df_hold["Total de l'actif"]) * 100,
            0
        )

    # --- AFFICHAGE DE LA PAGE SÉLECTIONNÉE ---

    if app_page == "📊 Tableau de Bord":
        st.markdown("###  📊 Vue d'ensemble")
        
        # Calcul des métriques globales du portefeuille.
        total_invested = df_hold["Total de l'actif"].sum()
        current_value_total = df_hold["Valeur Actuelle"].sum()
        
        # Calcul de la performance globale incluant Staking, Dividendes et Intérêts des actifs détenus
        total_staking_hold = df_hold["Gain de staking"].sum() if "Gain de staking" in df_hold.columns else 0
        total_div_hold = df_hold["Dividende"].sum() if "Dividende" in df_hold.columns else 0
        total_int_hold = df_hold["Intérêts"].sum() if "Intérêts" in df_hold.columns else 0
        # On utilise la somme de la colonne globale
        total_pnl_hold = df_hold["P&L Total"].sum()

        # Calcul de la variation journalière (depuis la clôture de la veille).
        reference_value_total = df_hold["Valeur Reference"].sum()
        daily_change_value = current_value_total - reference_value_total
        daily_change_percent = (daily_change_value / reference_value_total * 100) if reference_value_total > 0 else 0.0
        
        # Calcul de la variation sur 30 minutes
        value_30m_total = df_hold["Valeur 30m"].sum()
        change_30m_value = current_value_total - value_30m_total
        change_30m_percent = (change_30m_value / value_30m_total * 100) if value_30m_total > 0 else 0.0
        
        # Calcul des gains réalisés (plus-values de vente + dividendes + staking + intérêts).
        capital_gains = df_sold["Prix de vente"].sum() - df_sold["Total de l'actif"].sum() if not df_sold.empty else 0 # Gains sur ventes
        dividends = df["Dividende"].sum() if "Dividende" in df.columns else 0
        staking = df["Gain de staking"].sum() if "Gain de staking" in df.columns else 0
        interests = df["Intérêts"].sum() if "Intérêts" in df.columns else 0
        realized_gains = capital_gains + dividends + staking + interests

        st.caption("Synthèse globale de la performance et de la valeur de vos investissements.")

        # Ligne 1 : Principaux indicateurs de valeur
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Total Investi", f"{total_invested:,.2f} €".replace(',', ' '), help="La somme initiale investie sur les actifs actuellement détenus.")
        c2.metric("💎 Valeur Actuelle", f"{current_value_total:,.2f} €".replace(',', ' '), delta=f"{daily_change_value:+.2f}€ ({daily_change_percent:+.3f}%)", help="Valeur en temps réel. La variation correspond à l'évolution (Gain/Perte latente) depuis la clôture de la veille.")
        c3.metric("🚀 Performance Totale", f"{(total_pnl_hold/total_invested)*100:+.2f} %" if total_invested > 0 else "0%", delta=f"{total_pnl_hold:+.2f}€".replace(',', ' '), help=f"Inclut la Plus-value latente + Staking ({total_staking_hold:.2f}€) + Dividendes ({total_div_hold:.2f}€) + Intérêts ({total_int_hold:.2f}€)")

        st.markdown("<br>", unsafe_allow_html=True) # Espace entre les lignes

        # Ligne 2 : Indicateurs secondaires court terme & gains réalisés
        c4, c5, c6 = st.columns(3)
        c4.metric("⏱️ Var. 30 min", f"{change_30m_percent:+.3f} %", delta=f"{change_30m_value:+.2f}€".replace(',', ' '), help="Évolution intragroupe sur la dernière demi-heure. Indique la dynamique très court terme.")
        c5.metric("💸 Gains Réalisés", f"{realized_gains:,.2f} €".replace(',', ' '), help=f"Gains actés. Plus-values sur ventes: {capital_gains:.2f}€ | Dividendes/Staking/Intérêts perçus et hors-portefeuille: {dividends+staking+interests:.2f}€")
        val_gains = f"{realized_gains/total_invested*100:,.3f} %" if total_invested > 0 else "0.00 %"
        c6.metric("🏦 Ratio Gains / Investi", val_gains, help="Ratio des gains déjà sécurisés par rapport au total de vos investissements en cours.")

        st.markdown("---")

        # --- GRAPHIQUE ÉVOLUTION 7 JOURS ---
        if not df_hold.empty and not raw_history_df.empty:
            try:
                # Helper pour reconstruire la courbe globale d'un portefeuille selon les unités
                def build_portfolio_series(base_df):
                    if base_df.empty: return pd.Series(dtype=float)
                    df_work = base_df.copy()
                    if isinstance(df_work, pd.Series): df_work = df_work.to_frame()
                    
                    df_work.index = pd.to_datetime(df_work.index)
                    if df_work.index.tz is None:
                        df_work.index = df_work.index.tz_localize('UTC')
                    df_work.index = df_work.index.tz_convert('Europe/Paris')
                    df_work = df_work.ffill().bfill().fillna(0) # CORRECTION: bfill() empêche les actifs sans historique lointain de valoir 0

                    rates_dict = {
                        "USD": pd.Series(1.0, index=df_work.index),
                        "GBP": pd.Series(1.0, index=df_work.index),
                        "CHF": pd.Series(1.0, index=df_work.index)
                    }
                    
                    for cur, symbol in [("USD", "EURUSD=X"), ("GBP", "GBPEUR=X"), ("CHF", "CHFEUR=X")]:
                        if symbol in df_work.columns:
                            cur_data = df_work[symbol]
                            if isinstance(cur_data, pd.DataFrame) and 'Close' in cur_data.columns:
                                rates_dict[cur] = cur_data['Close']
                            elif isinstance(cur_data, pd.Series):
                                rates_dict[cur] = cur_data
                        rates_dict[cur] = rates_dict[cur].replace(0, np.nan).ffill().bfill().fillna(1.0)

                    port_series = pd.Series(0.0, index=df_work.index)
                    
                    # NOUVEAU: Optimisation forte avec groupby
                    # On regroupe par Ticker et Devise pour éviter de parser 5x le même graphique si on a 5 lignes du même ETF
                    df_hold_grouped = df_hold.groupby(['Ticker', 'Devise'], as_index=False)['Unités'].sum()
                    
                    for row in df_hold_grouped.to_dict('records'):
                        t = row['Ticker']
                        if t in df_work.columns:
                            data_t = df_work[t]
                            ps = pd.Series(dtype=float)
                            if isinstance(data_t, pd.DataFrame):
                                if 'Close' in data_t.columns:
                                    ps = data_t['Close']
                                else:
                                    # Fallback si pas de colonne 'Close' mais c'est un DataFrame de shape (N, 1)
                                    if len(data_t.columns) > 0:
                                        ps = data_t.iloc[:, 0]
                            elif isinstance(data_t, pd.Series):
                                ps = data_t
                                
                            if not ps.empty:
                                ps = ps.reindex(port_series.index, method='ffill').bfill().fillna(0) # CORRECTION: bfill() ici
                                if row['Devise'] == 'USD':
                                    ps = ps / rates_dict['USD']
                                elif row['Devise'] == 'GBP':
                                    ps = ps * rates_dict['GBP']
                                elif row['Devise'] == 'CHF':
                                    ps = ps * rates_dict['CHF']
                                port_series = port_series.add(ps * row['Unités'], fill_value=0)
                    return port_series

                # 1. Courbe court-terme (intraday sur 1 mois) pour graphique & rendement 7J
                portfolio_series = build_portfolio_series(raw_history_df)
                # 2. Courbe long-terme (journalier sur 2 ans) pour volatilité, Sharpe et MaxDD
                portfolio_series_hist = build_portfolio_series(daily_history_df)

                # Sélection des 30 derniers jours (Filtrage temporel pour données intraday)
                if not portfolio_series.empty:
                    cutoff_date = portfolio_series.index.max() - pd.Timedelta(days=30)
                    portfolio_last_30d = portfolio_series[portfolio_series.index > cutoff_date]
                else:
                    portfolio_last_30d = pd.Series(dtype=float)
                
                # --- AJOUT: KPIs AVANCÉS DE PORTEFEUILLE ---
                if not portfolio_series_hist.empty: 
                    # Resample journalier de l'historique COMPLET (2 ans) 
                    # Dropna sécurise les calculs s'il y a des trous
                    daily_portfolio_hist = portfolio_series_hist.resample("B").last().dropna()
                    
                    if not portfolio_last_30d.empty:
                        # Resample journalier sur les 30 derniers jours de la courbe intraday
                        daily_portfolio_30d = portfolio_last_30d.resample("1D").last().dropna()
                    else:
                        daily_portfolio_30d = pd.Series(dtype=float)
                        
                    # Calculs
                    kpis_hist = calculate_portfolio_kpis(daily_portfolio_hist)
                    kpis_30d = calculate_portfolio_kpis(daily_portfolio_30d)
                    
                    st.subheader(" ⚖️ Risque et Performance Globale (Historique Long Terme & 30J)")
                    st.write("") # Espace ajouté
                    kcol1, kcol2, kcol3, kcol4, kcol5 = st.columns(5)
                    kcol1.metric("Ratio de Sharpe (LT)", f"{kpis_hist['Sharpe']:.2f}", help="Calculé sur 2 ans d'historique. Indicateur de rendement ajusté au risque global (vs T-Note 10A US). >1 est bon, >2 est fantastique.")
                    kcol2.metric("Ratio de Sortino (LT)", f"{kpis_hist['Sortino']:.2f}", help="Calculé sur 2 ans. Ne pénalise que le risque de Baisse (contrairement au Sharpe). Idéalement plus élevé que le Sharpe.")
                    kcol3.metric("Max Drawdown (LT)", f"{kpis_hist['Max Drawdown']:.2f} %", help="Calculé sur 2 ans d'historique. Perte maximale (du plus haut au plus bas). Mesure votre pire scénario.")
                    kcol4.metric("Volatilité Annuelle (LT)", f"{kpis_hist['Volatilité']:.2f} %", help="Calculé sur 2 ans d'historique. Indice de turbulence. Plus le % est élevé, plus le portefeuille fait des montagnes russes.")
                    if not daily_portfolio_30d.empty:
                        kcol5.metric("Rendement (30 Jours)", f"{kpis_30d['Period Return']:.2f} %", help="Performance totale du portefeuille uniquement sur les 30 derniers jours.")
                    else:
                        kcol4.metric("Rendement (30 Jours)", "0.00 %")
                    st.markdown("---")

                if not portfolio_last_30d.empty:
                    st.subheader(" 📈 Évolution de la valeur (30 derniers jours)")
                    
                    # On utilise les données telles quelles pour préserver les plateaux (week-end/nuit)
                    chart_series = portfolio_last_30d
                    
                    # Calcul dynamique de l'échelle Y pour zoomer sur les variations
                    y_min = chart_series.min()
                    y_max = chart_series.max()
                    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else y_max * 0.01

                    fig_evol = go.Figure()
                    fig_evol.add_trace(go.Scatter(
                        x=chart_series.index, 
                        y=chart_series.values,
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#3db4f2', width=3),
                        fillcolor='rgba(61, 180, 242, 0.1)',
                        name='Valeur Portefeuille',
                        hovertemplate='<b>%{x|%d/%m %H:%M}</b><br>Valeur: %{y:.2f} €<extra></extra>'
                    ))
                    
                    fig_evol.update_layout(
                        margin=dict(t=10, b=10, l=0, r=0), 
                        height=250, 
                        showlegend=False, 
                        yaxis=dict(range=[y_min - y_margin, y_max + y_margin], gridcolor='rgba(128,128,128,0.1)', tickfont=dict(color='#8ba0b2')),
                        xaxis=dict(showgrid=False, tickfont=dict(color='#8ba0b2')), 
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)',
                        hovermode="x unified",
                        font=dict(color='#bcbedc')
                    )
                    st.plotly_chart(fig_evol, width='stretch', config={'displayModeBar': False})
                    st.markdown("---")
            except Exception as e:
                st.warning(f"Impossible d'afficher l'historique global : {e}")

        # Affichage des graphiques de synthèse.
        col_left, col_right = st.columns([1, 2])
        with col_left:
            if not df_hold.empty:
                # Camembert pour la répartition par type d'actif.
                fig_pie = px.pie(df_hold, values="Valeur Actuelle", names="Type d'actif", hole=0.5, 
                                 color_discrete_sequence=px.colors.qualitative.Bold, title="Répartition du Portefeuille")
                fig_pie.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Valeur: %{value:.2f} €<br>Part: %{percent}<extra></extra>')
                # Hauteur fixe pour éviter l'écrasement sur mobile
                fig_pie.update_layout(
                    margin=dict(t=40, b=20, l=20, r=20), 
                    showlegend=False, 
                    uniformtext_minsize=10, 
                    uniformtext_mode='hide',
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#bcbedc'),
                    title_font=dict(size=18, color='#edf1f5')
                )
                st.plotly_chart(fig_pie, width='stretch', config={'displayModeBar': False})
        with col_right:
            if not df_hold.empty:
                # Graphique en barres pour la performance de chaque actif.
                df_chart = df_hold.sort_values(by="P&L Total", ascending=True) # Tri pour afficher les plus gros gains en haut.
                # Couleurs plus modernes (Vert Néon / Rouge Néon)
                df_chart['Color'] = df_chart['P&L Total'].apply(lambda x: '#00ff9d' if x >= 0 else '#ff0055')
                df_chart['Perf_Pct'] = df_chart.apply(lambda x: (x['P&L Total'] / x["Total de l'actif"] * 100) if x["Total de l'actif"] > 0 else 0, axis=1)
                
                # Calcul dynamique de la hauteur (40px par barre + marge) pour lisibilité sur mobile
                dynamic_height = max(350, len(df_chart) * 40)
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    y=df_chart["Nom de l'actif"], x=df_chart['P&L Total'], orientation='h',
                    marker_color=df_chart['Color'], 
                    text=df_chart['P&L Total'].apply(lambda x: f"{x:+.2f} €"), 
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Gain/Perte Total: %{x:.2f} €<br>Performance: %{customdata:.2f}%<extra></extra>',
                    customdata=df_chart['Perf_Pct']
                ))
                fig_bar.update_layout(
                    title="Performance Latente par Actif",
                    margin=dict(t=40, b=0, l=0, r=0), 
                    xaxis_title="Gain/Perte (€)",
                    xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='#8ba0b2')),
                    yaxis=dict(showgrid=False, automargin=True, tickfont=dict(color='#8ba0b2')),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#bcbedc'),
                    title_font=dict(size=18, color='#edf1f5'),
                    height=dynamic_height,
                    autosize=True
                )
                st.plotly_chart(fig_bar, width='stretch', config={'displayModeBar': False})

    elif app_page == "📈 Performance & Prévisions":
        st.subheader("📊 Détail des positions & Prédictions")
        st.write("") # Espace ajouté
        
        def style_trend_text(val):
            if pd.isna(val): return ''
            color = '#2ecc71' if val > 0 else '#ff4b4b' if val < 0 else ''
            if color: return f'color: {color}; font-weight: bold;'
            return ''

        def style_signal_text(val):
            if pd.isna(val): return ''
            val_str = str(val)
            if "🟢" in val_str: return 'color: #2ecc71; font-weight: bold;'
            if "🔴" in val_str: return 'color: #ff4b4b; font-weight: bold;'
            return ''
            
        with st.expander("ℹ️ Comment analyser ce tableau ?", expanded=False):
            st.markdown("""
            **1. Position & Performance**
            *   **Evol. Jour %** : Variation par rapport à la veille. Utile pour suivre l'humeur immédiate du marché.
            *   **Performance %** : Votre gain ou perte total depuis l'achat (incluant dividendes/staking/intérêts, qui agissent comme un bonus fluctuant réduisant les pertes).
            
            **2. Indicateurs Techniques (La "Météo" du marché)**
            *   **Signal** : Affiche les alertes critiques en priorité (Stop ATR 🔴, Surchauffe MM200 🔴, Climax Achat ⚠️, MACD 🔴), puis se rabat sur MME 9/21 et Bollinger.
            *   **ATR (Volatilité)** : Indique la nervosité de l'actif. Un chiffre élevé = gros mouvements de prix (risque plus élevé).
            *   **MACD** : Indicateur d'élan. Positif = Poussée haussière. Négatif = Poussée baissière.
            *   **MM 200** : La "Juge de Paix". Si le prix est au-dessus, la tendance de fond (long terme) est saine.
            *   **Bandes Bollinger** : Canaux de volatilité. Si le prix touche le haut, risque de correction. Si bas, rebond possible.
            *   **Stoch %K** : Oscillateur (0-100). >80 = Surchauffe (Vente?), <20 = Survendu (Achat?).
            
            **3. Prédictions : 🤖 (ML XGBoost) vs 🔮 (Prophet) vs 📐 (Polynomiale)**
            *   **🤖 (ML XGBoost)** : Détecte des patterns techniques complexes (RSI, Volatilité) pour des prévisions agressives. **Colonne "7j/30j 🤖 %".**
            *   **🔮 (Prophet)** : L'algorithme de Facebook, spécialisé dans les cycles et la saisonnalité (effets week-end). Plus modéré et naturel. **Colonne "7j/30j 🔮 %".**
            *   **📐 (Poly)** : Simple prolongement géométrique de la tendance. ⚠️ *Souvent trop extrême, à utiliser avec précaution.* **Colonne "7j/30j 📐 %".**
            """)

        st.write("") # Espace ajouté
        # Tri du DataFrame pour l'affichage par valeur actuelle décroissante
        df_details_sorted = df_hold.sort_values(by="Valeur Actuelle", ascending=False)

        # st.dataframe est utilisé pour un affichage interactif avec des mini-graphiques.
        st.dataframe(
            df_details_sorted[["Nom de l'actif", "Type", "Prix Actuel", "Valeur Actuelle", "Plus-value Latente", "Gains Annexes", "Evol. Jour %", "Evol. Hebdo %", "Trend 7j", "Performance %", "Signal Technique", "Stoch K", "ATR", "MACD", "MM 200", "BB Haut", "BB Bas", "Proj. 7j (Prophet)", "Proj. 7j (ML)", "Proj. 7j (%)", "Proj. 30j (Prophet)", "Proj. 30j (ML)", "Proj. 30j (%)", "Historique"]]
            .style
            .map(style_trend_text, subset=['Plus-value Latente', 'Gains Annexes', 'Evol. Jour %', 'Evol. Hebdo %', 'Performance %', 'MACD', 'Proj. 7j (%)', 'Proj. 30j (%)', 'Proj. 7j (ML)', 'Proj. 30j (ML)', 'Proj. 7j (Prophet)', 'Proj. 30j (Prophet)'])
            .map(style_signal_text, subset=['Signal Technique']),
            column_config={
                "Nom de l'actif": st.column_config.TextColumn("Actif", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Prix Actuel": st.column_config.NumberColumn("Cours Actuel", format="%.2f €"),
                "Valeur Actuelle": st.column_config.NumberColumn("Val. Actuelle", format="%.2f €"),
                "Plus-value Latente": st.column_config.NumberColumn("Plus-value", format="%+.2f €", help="Gain ou perte sur la seule variation du prix de l'actif"),
                "Gains Annexes": st.column_config.NumberColumn("Gains Annexes", format="%+.2f €", help="Cumul des revenus passifs (Staking, Dividendes, Intérêts)"),
                "Evol. Jour %": st.column_config.NumberColumn("Evol. Jour", format="%+.2f %%", help="Variation par rapport à la clôture précédente"),
                "Evol. Hebdo %": st.column_config.NumberColumn("Evol. Hebdo", format="%+.2f %%", help="Variation sur 7 jours (5 jours de bourse)"),
                "Trend 7j": st.column_config.LineChartColumn("Trend 7j", width="small", help="Tendance des 7 derniers jours"),
                "Performance %": st.column_config.NumberColumn("Perf %", format="%+.2f %%"),
                "Signal Technique": st.column_config.TextColumn("Signal", help="Alerte critique (ATR/MM200/Volume/MACD), MME ou Bollinger"),
                "MM 200": st.column_config.NumberColumn("[1] MM 200", format="%.2f €", help="Moyenne Mobile Simple 200j"),
                "ATR": st.column_config.NumberColumn("[1] ATR (14)", format="%.2f €", help="Average True Range (Volatilité)"),
                "MACD": st.column_config.NumberColumn("[2] MACD", format="%.2f", help="MACD (12, 26)"),
                "BB Haut": st.column_config.NumberColumn("[2] BB Haut", format="%.2f €", help="Bande de Bollinger Haute (20, 2)"),
                "BB Bas": st.column_config.NumberColumn("[2] BB Bas", format="%.2f €", help="Bande de Bollinger Basse (20, 2)"),
                "Stoch K": st.column_config.NumberColumn("[3] Stoch %K", format="%.0f", help="Oscillateur Stochastique (14, 3)"),
                "Proj. 7j (%)": st.column_config.NumberColumn("7j 📐 Poly %", format="%+.2f %%", help="Projection Polynomiale (mathématique simple) sur 7 jours"),
                "Proj. 7j (ML)": st.column_config.NumberColumn("7j 🤖 XGBoost %", format="%+.2f %%", help="Projection XGBoost (Machine Learning technique) sur 7 jours"),
                "Proj. 7j (Prophet)": st.column_config.NumberColumn("7j 🔮 Prophet %", format="%+.2f %%", help="Projection Prophet (Saisonnalité & Historique) sur 7 jours"),
                "Proj. 30j (%)": st.column_config.NumberColumn("30j 📐 Poly %", format="%+.2f %%", help="Projection Polynomiale (mathématique simple) sur 30 jours"),
                "Proj. 30j (ML)": st.column_config.NumberColumn("30j 🤖 XGBoost %", format="%+.2f %%", help="Projection XGBoost (Machine Learning technique) sur 30 jours"),
                "Proj. 30j (Prophet)": st.column_config.NumberColumn("30j 🔮 Prophet %", format="%+.2f %%", help="Projection Prophet (Saisonnalité & Historique) sur 30 jours"),
                "Historique": st.column_config.LineChartColumn("Tendance (3 mois)"),
            },
            width="stretch",
            hide_index=True
        )

        # --- EXPORT DATA (NOUVEAU) ---
        st.markdown("")
        csv_data = df_hold.drop(columns=['Historique'], errors='ignore').to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données enrichies (CSV)",
            data=csv_data,
            file_name="portefeuille_enrichi.csv",
            mime="text/csv",
            help="Télécharge le tableau ci-dessus avec les prédictions."
        )

        # --- NOUVELLE SECTION: GRAPHIQUE EN CHANDELIERS ---
        st.markdown("---")
        st.subheader("🔍 Analyse Détaillée par Actif")
        st.write("") # Espace ajouté
        
        # Sélection de l'actif
        selected_asset = st.selectbox(
            "Choisissez un actif pour voir le graphique détaillé :",
            options=df_hold["Nom de l'actif"].unique(),
            label_visibility="collapsed"
        )
        
        if selected_asset:
            # Récupération du ticker et des données complètes
            ticker = df_hold[df_hold["Nom de l'actif"] == selected_asset]["Ticker"].iloc[0]
            df_full = full_ticker_data.get(ticker)
            
            if df_full is not None and not df_full.empty and all(c in df_full.columns for c in ['Open', 'High', 'Low', 'Close', 'Volume']):
                # On prend les 6 derniers mois pour la lisibilité
                if not df_full.index.empty:
                    six_months_ago = df_full.index[-1] - pd.DateOffset(months=6)
                    df_chart = df_full.loc[six_months_ago:].copy()
                else:
                    df_chart = df_full.copy()

                # Création de la figure avec subplots (prix + volume + stochastique)
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                      vertical_spacing=0.05, subplot_titles=(f'Cours de {selected_asset} ({ticker})', 'Volume', 'Stochastique'),
                                      row_heights=[0.6, 0.2, 0.2])

                # 1. Graphique en chandeliers (avec couleurs custom)
                fig.add_trace(go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'],
                                high=df_chart['High'],
                                low=df_chart['Low'],
                                close=df_chart['Close'],
                                name='OHLC',
                                increasing_line_color='#2ecc71', decreasing_line_color='#ff4b4b'),
                              row=1, col=1)

                # 2. Ajout des Moyennes Mobiles et Bandes de Bollinger (avec couleurs et groupes)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart.get('MME_9'), line=dict(color='#f1c40f', width=1.5), name='MME 9', legendgroup="MAs"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart.get('MME_21'), line=dict(color='#e67e22', width=1.5), name='MME 21', legendgroup="MAs"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart.get('MM_200'), line=dict(color='#9b59b6', width=2, dash='dash'), name='MM 200', legendgroup="MAs"), row=1, col=1)
                
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart.get('BB_Upper'), line=dict(color='rgba(142, 150, 170, 0.5)', width=1, dash='dot'), name='Bollinger', legendgroup="Bollinger"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart.get('BB_Lower'), line=dict(color='rgba(142, 150, 170, 0.5)', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(142, 150, 170, 0.1)', showlegend=False, name='Bollinger', legendgroup="Bollinger"), row=1, col=1)

                # --- AMÉLIORATION: Niveaux de Fibonacci avec Annotations ---
                max_p = df_chart['High'].max()
                min_p = df_chart['Low'].min()
                diff = max_p - min_p
                
                fib_levels = {
                    0.236: ("23.6%", "rgba(235, 59, 90, 0.8)"),
                    0.382: ("38.2%", "rgba(250, 130, 49, 0.8)"),
                    0.5: ("50%", "rgba(254, 202, 87, 0.9)"),
                    0.618: ("61.8%", "rgba(32, 191, 107, 0.9)"), # Golden Pocket
                    0.786: ("78.6%", "rgba(45, 152, 218, 0.8)"),
                }
                
                # On ajoute une seule trace "invisible" pour la légende
                fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines',
                                         line=dict(color='gray', width=1, dash='dot'),
                                         name='Fibonacci', legendgroup="Fibonacci"), row=1, col=1)

                for ratio, (label, color) in fib_levels.items():
                    level_price = max_p - (diff * ratio)
                    # Ligne sur le graphique
                    fig.add_shape(type='line',
                                  x0=df_chart.index[0], y0=level_price,
                                  x1=df_chart.index[-1], y1=level_price,
                                  line=dict(color=color, width=1, dash='dot'),
                                  row=1, col=1)
                    # Annotation sur le côté
                    fig.add_annotation(x=df_chart.index[-1], y=level_price,
                                       xref="x", yref="y",
                                       text=f" {label} ({level_price:.2f})",
                                       showarrow=False,
                                       xanchor="left",
                                       xshift=5,
                                       font=dict(color=color, size=10),
                                       bgcolor="rgba(11, 22, 34, 0.7)",
                                       row=1, col=1)

                # --- AJOUT: Signaux d'Achat/Vente sur Croisement MME ---
                if 'MME_9' in df_chart.columns and 'MME_21' in df_chart.columns:
                    # 1 = MME9 > MME21, 0 = MME9 < MME21
                    signals = np.where(df_chart['MME_9'] > df_chart['MME_21'], 1.0, 0.0)
                    # Différence pour détecter les changements (1 = Croisement Achat, -1 = Croisement Vente)
                    crossovers = pd.Series(signals, index=df_chart.index).diff()
                    
                    buy_signals = df_chart[crossovers == 1.0]
                    sell_signals = df_chart[crossovers == -1.0]
                    
                    if not buy_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=buy_signals.index, y=buy_signals['Low'] * 0.98,
                            mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00', line=dict(width=1, color='black')),
                            name='Signal Achat (MME)'
                        ), row=1, col=1)
                        
                    if not sell_signals.empty:
                        fig.add_trace(go.Scatter(
                            x=sell_signals.index, y=sell_signals['High'] * 1.02,
                            mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000', line=dict(width=1, color='black')),
                            name='Signal Vente (MME)'
                        ), row=1, col=1)

                # 3. Graphique en barres pour le volume (Vectorisé pour les performances)
                colors = np.where(df_chart['Close'] >= df_chart['Open'], '#2ecc71', '#ff4b4b')
                fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], name='Volume', marker_color=colors), row=2, col=1)

                # 4. Oscillateur Stochastique
                if 'Stoch_K' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Stoch_K'], line=dict(color='#bd93f9', width=2), name='Stoch %K'), row=3, col=1)
                    fig.add_hline(y=80, line_dash="dot", line_color="#ff5555", row=3, col=1)
                    fig.add_hline(y=20, line_dash="dot", line_color="#50fa7b", row=3, col=1)
                    fig.update_yaxes(range=[0, 100], row=3, col=1)

                # --- AJOUT: Visualisation des Prédictions (Courbe + Cible IA) ---
                # 1. Projection Polynomiale (Basée sur la tendance visible - 6 mois)
                if len(df_chart) > 30:
                    y_hist = df_chart['Close'].values
                    x_hist = np.arange(len(y_hist))
                    coeffs = np.polyfit(x_hist, y_hist, 2)
                    poly_func = np.poly1d(coeffs)
                    
                    # Projection sur 30 jours
                    future_days = 30
                    x_future = np.arange(len(y_hist), len(y_hist) + future_days)
                    y_future = poly_func(x_future)
                    
                    last_date = df_chart.index[-1]
                    # Génération des dates futures (Adaptatif Bourse vs Crypto)
                    has_weekends = len(df_chart[df_chart.index.dayofweek > 4]) > 0
                    if has_weekends:
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
                    else:
                        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                    
                    # Calcul de la marge d'erreur (cône d'incertitude)
                    residuals = y_hist - poly_func(x_hist)
                    std_dev_residuals = np.std(residuals)
                    prediction_range = std_dev_residuals * 1.5 # +/- 1.5 écarts-types

                    y_future_high = y_future + prediction_range
                    y_future_low = np.maximum(0, y_future - prediction_range) # Empêche le cône d'aller en négatif

                    # Cône d'incertitude (dessiné en premier pour être en arrière-plan)
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=y_future_high, mode='lines',
                        line=dict(width=0), showlegend=False, name='Proj. Haut'
                    ), row=1, col=1)
                    
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=y_future_low, mode='lines',
                        line=dict(width=0), fill='tonexty',
                        fillcolor='rgba(61, 180, 242, 0.2)',
                        showlegend=False, name='Proj. Bas'
                    ), row=1, col=1)
                    
                    # Courbe de tendance (dessinée après le cône pour être au-dessus)
                    fig.add_trace(go.Scatter(
                        x=future_dates, y=y_future, mode='lines',
                        line=dict(color='#3db4f2', width=2, dash='dash'),
                        name='Tendance (Proj. 30j)'
                    ), row=1, col=1)

                # 2. Cible IA (Machine Learning)
                # On utilise les données complètes pour le calcul ML. 
                # IMPORTANT : reset_index() obligatoire pour que la Date fasse partie du tuple attendu par ml_models.py
                data_tuple = tuple(df_full[['Close', 'High', 'Low', 'Volume']].reset_index().itertuples(index=False, name=None))
                pred_price_ml, _ = calculate_ml_prediction(data_tuple, days_ahead=30)
                
                if pred_price_ml:
                    # Alignement de la cible ML sur la fin du cône de prédiction (gère les Jours Ouvrés vs Crypto)
                    target_date = future_dates[-1] if 'future_dates' in locals() else df_chart.index[-1] + pd.Timedelta(days=30)
                    fig.add_trace(go.Scatter(
                        x=[target_date], y=[pred_price_ml], mode='markers',
                        marker=dict(symbol='star', size=14, color='#f1c40f', line=dict(width=1, color='black')),
                        name=f'Cible IA ({pred_price_ml:.2f})'
                    ), row=1, col=1)

                # 5. Mise en forme
                fig.update_layout(
                    height=700,
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#bcbedc'),
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor='rgba(0,0,0,0)',
                        bordercolor='rgba(128,128,128,0.2)',
                        borderwidth=1
                    ),
                    margin=dict(t=50, r=100), # Marge à droite pour les annotations Fibo
                )
                # Style des titres de subplots
                fig.update_annotations(patch={"font": {"color": "#8ba0b2", "size": 12}})
                if fig.layout.annotations:
                    fig.layout.annotations[0].update(font=dict(color='#edf1f5', size=16), y=0.98)
                fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
                fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)')

                st.plotly_chart(fig, width='stretch')
                
                # --- Analyse Textuelle Automatique ---
                last_row = df_chart.iloc[-1]
                close_p = last_row['Close']
                analysis_points = []
                
                # 1. Tendance Long Terme (MM200)
                if pd.notnull(last_row.get('MM_200')):
                    trend = "haussière 🟢" if close_p > last_row['MM_200'] else "baissière 🔴"
                    analysis_points.append(f"La tendance de fond est **{trend}** (par rapport à la MM200).")
                
                # 2. Dynamique Court Terme (MME)
                if pd.notnull(last_row.get('MME_9')) and pd.notnull(last_row.get('MME_21')):
                    momentum = "positive 🚀" if last_row['MME_9'] > last_row['MME_21'] else "fragile 📉"
                    analysis_points.append(f"La dynamique court terme est **{momentum}**.")
                
                # 3. Bandes de Bollinger
                if pd.notnull(last_row.get('BB_Upper')) and pd.notnull(last_row.get('BB_Lower')):
                    if close_p >= last_row['BB_Upper'] * 0.98:
                        analysis_points.append("⚠️ **Attention** : Le prix approche de la borne haute (risque de correction).")
                    elif close_p <= last_row['BB_Lower'] * 1.02:
                        analysis_points.append("💡 **Opportunité** : Le prix est proche de la borne basse (rebond possible).")

                # 4. RSI (Surachat / Survente)
                if pd.notnull(last_row.get('RSI')):
                    rsi_val = last_row['RSI']
                    if rsi_val > 70:
                        analysis_points.append(f"L'actif est en zone de **surachat** (RSI={rsi_val:.0f}), un repli est possible.")
                    elif rsi_val < 30:
                        analysis_points.append(f"L'actif est en zone de **survente** (RSI={rsi_val:.0f}), un rebond est possible.")

                # 5. Stochastique
                if pd.notnull(last_row.get('Stoch_K')):
                    stoch_val = last_row['Stoch_K']
                    if stoch_val > 80:
                        analysis_points.append(f"L'oscillateur Stochastique est en zone de **surchauffe** ({stoch_val:.0f}), signalant une potentielle baisse.")
                    elif stoch_val < 20:
                        analysis_points.append(f"L'oscillateur Stochastique est en zone de **survente** ({stoch_val:.0f}), signalant un potentiel rebond.")

                # 6. Signal Croisement MME Récent
                if 'MME_9' in df_chart.columns and 'MME_21' in df_chart.columns:
                     # Recalcul rapide sur les derniers jours pour le texte
                     recent_signals = np.where(df_chart['MME_9'].tail(5) > df_chart['MME_21'].tail(5), 1.0, 0.0)
                     recent_crossovers = pd.Series(recent_signals).diff().dropna()
                     if (recent_crossovers == 1.0).any():
                         analysis_points.append("🚀 **Signal Achat** : Croisement haussier des moyennes mobiles (MME 9 > MME 21) détecté récemment.")
                     elif (recent_crossovers == -1.0).any():
                         analysis_points.append("🔻 **Signal Vente** : Croisement baissier des moyennes mobiles (MME 9 < MME 21) détecté récemment.")
                
                # 6b. Signaux d'épuisement et de sécurité (MACD, ATR, Volume)
                if 'MACD' in df_chart.columns and 'MACD_Signal' in df_chart.columns:
                    macd = last_row.get('MACD')
                    macd_sig = last_row.get('MACD_Signal')
                    if pd.notnull(macd) and pd.notnull(macd_sig) and macd > 0 and macd_sig > 0 and macd < macd_sig:
                        analysis_points.append("⚠️ **Essoufflement** : Croisement baissier du MACD détecté en zone positive.")
                
                if 'Highest_50' in df_chart.columns and 'ATR' in df_chart.columns:
                    h50 = last_row.get('Highest_50')
                    atr = last_row.get('ATR')
                    if pd.notnull(h50) and pd.notnull(atr) and close_p < h50 - 2.5 * atr:
                        analysis_points.append("🛑 **Alerte Sécurité** : Le prix a cassé son seuil d'invalidation (Stop ATR). Tendance haussière rompue.")
                
                if 'Volume' in df_chart.columns and 'Vol_SMA_20' in df_chart.columns and 'MME_21' in df_chart.columns:
                    vol = last_row.get('Volume')
                    vol_sma = last_row.get('Vol_SMA_20')
                    mme21 = last_row.get('MME_21')
                    if pd.notnull(vol) and pd.notnull(vol_sma) and vol_sma > 0 and pd.notnull(mme21) and close_p > mme21:
                        if vol > 3 * vol_sma:
                            analysis_points.append(f"💣 **Climax** : Volume disproportionné détecté (x{vol/vol_sma:.1f}). Possible fin de cycle acheteur imminent.")
                            
                if pd.notnull(last_row.get('MM_200')) and last_row.get('MM_200') > 0:
                    dist_mm200 = (close_p - last_row['MM_200']) / last_row['MM_200']
                    if dist_mm200 > 0.35:
                        analysis_points.append(f"🔥 **Surchauffe Extrême** : Le prix est en lévitation à **+{dist_mm200*100:.0f}%** de sa MM200 historique. Chute probable à court terme.")

                # 7. Prédiction IA
                if pred_price_ml:
                    current_p = df_chart['Close'].iloc[-1]
                    diff_pct = ((pred_price_ml - current_p) / current_p) * 100
                    direction = "haussière 🚀" if diff_pct > 0 else "baissière 📉"
                    analysis_points.append(f"🤖 **Prédiction IA** : Cible à 30 jours estimée à **{pred_price_ml:.2f}** ({direction} {diff_pct:+.2f}%).")

                # 8. Proximité Fibonacci
                max_p = df_chart['High'].max()
                min_p = df_chart['Low'].min()
                diff = max_p - min_p
                for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
                    level_val = max_p - (diff * ratio)
                    if abs(close_p - level_val) / close_p < 0.015: # 1.5% de marge
                        analysis_points.append(f"📐 **Fibonacci** : Le prix teste le niveau clé **{ratio*100}%** ({level_val:.2f}).")

                if analysis_points:
                    st.info("  \n".join(analysis_points))
                    
                st.write("") # Espace
                # --- Bouton de téléchargement des données du graphique ---
                csv_chart = df_chart.to_csv().encode('utf-8')
                st.download_button(
                    label=f"📥 Télécharger l'historique de {selected_asset} (CSV)",
                    data=csv_chart,
                    file_name=f"{ticker}_data.csv",
                    mime="text/csv",
                    key=f"download_{ticker}"
                )
            else:
                st.warning(f"Données historiques complètes (OHLCV) non disponibles pour {selected_asset}.")

    elif app_page == "🧠 Analyse Technique & Risques":
        st.subheader("⚖️ Analyse Avancée des Risques & Corrélations")
        st.write("") # Espace ajouté
        
        if not daily_history_df.empty:
            # Conversion en DataFrame si Série unique
            if isinstance(daily_history_df, pd.Series):
                daily_history_df = daily_history_df.to_frame()

            # Nettoyage : On garde uniquement les tickers présents dans le portefeuille
            valid_cols = [t for t in daily_history_df.columns if t in df_hold['Ticker'].values]
            
            if len(valid_cols) > 0:
                analysis_df = daily_history_df[valid_cols].copy()
                
                # --- SÉLECTEUR DE PÉRIODE ---
                st.markdown("### ⏱️ Période d'Analyse")
                periode_options = {"30 Jours": 30, "3 Mois": 90, "6 Mois": 180, "1 An": 365, "Max": None}
                selected_period = st.selectbox("Sélectionnez la période pour l'analyse des risques et corrélations :", list(periode_options.keys()), index=4)
                
                days_to_keep = periode_options[selected_period]
                if days_to_keep is not None:
                    cutoff_date = analysis_df.index.max() - pd.Timedelta(days=days_to_keep)
                    analysis_df = analysis_df[analysis_df.index >= cutoff_date]
                
                returns_df = analysis_df.pct_change(fill_method=None).dropna()
                
                # --- KPIs DE RISQUE AVANCÉS ---
                st.markdown("---")
                st.markdown(f"### 📊 Indicateurs de Risque et Performance ({selected_period})")
                
                kpi_data = []
                for asset in valid_cols:
                    asset_prices = analysis_df[asset].dropna()
                    if len(asset_prices) > 1:
                        asset_kpis = calculate_portfolio_kpis(asset_prices)
                        asset_name = df_hold[df_hold['Ticker'] == asset]["Nom de l'actif"].iloc[0] if not df_hold[df_hold['Ticker'] == asset].empty else asset
                        kpi_data.append({
                            "Actif": asset_name,
                            "Ticker": asset,
                            "Rendement (%)": asset_kpis.get("Period Return", 0.0),
                            "Volatilité Ann. (%)": asset_kpis.get("Volatilité", 0.0),
                            "Ratio de Sharpe": asset_kpis.get("Sharpe", 0.0),
                            "Ratio de Sortino": asset_kpis.get("Sortino", 0.0),
                            "Max Drawdown (%)": asset_kpis.get("Max Drawdown", 0.0)
                        })
                
                if kpi_data:
                    df_kpi = pd.DataFrame(kpi_data).sort_values("Ratio de Sharpe", ascending=False)
                    st.dataframe(
                        df_kpi.style.format({
                            "Rendement (%)": "{:+,.2f}%",
                            "Volatilité Ann. (%)": "{:,.2f}%",
                            "Ratio de Sharpe": "{:,.2f}",
                            "Ratio de Sortino": "{:,.2f}",
                            "Max Drawdown (%)": "{:,.2f}%"
                        }).background_gradient(subset=["Ratio de Sharpe"], cmap="RdYlGn")
                        .background_gradient(subset=["Max Drawdown (%)"], cmap="Reds_r")
                        .background_gradient(subset=["Rendement (%)"], cmap="RdYlGn")
                        .background_gradient(subset=["Ratio de Sortino"], cmap="RdYlGn"),
                        width="stretch",
                        hide_index=True
                    )
                
                st.markdown("---")
                # --- CORRÉLATION ET VOLATILITÉ ---
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    st.markdown("#### 🔥 Matrice de Corrélation")    
                    st.caption("Mesure à quel point vos actifs bougent ensemble (1 = identique, -1 = opposé).")
                    
                    if len(valid_cols) > 1 and not returns_df.empty:
                        corr_matrix = returns_df.corr()
                        
                        # Filtre interactif
                        highlight_high_corr = st.toggle("Mettre en évidence les fortes corrélations (> 0.6)", value=False)
                        if highlight_high_corr:
                            corr_display = corr_matrix.copy()
                            corr_display[(abs(corr_display) < 0.6) | (corr_display >= 0.999)] = np.nan
                            
                            fig_corr = px.imshow(
                                corr_display, 
                                text_auto=".2f", 
                                color_continuous_scale='RdBu_r', 
                                zmin=-1, zmax=1,
                                aspect="auto"
                            )
                        else:
                            fig_corr = px.imshow(
                                corr_matrix, 
                                text_auto=".2f", 
                                color_continuous_scale='RdBu_r', 
                                zmin=-1, zmax=1,
                                aspect="auto"
                            )
                            
                        fig_corr.update_layout(
                            margin=dict(t=10, b=0, l=0, r=0), 
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#bcbedc')
                        )
                        st.plotly_chart(fig_corr, width='stretch', config={'displayModeBar': False})
                    else:
                        st.info("Pas assez de données pour afficher la corrélation.")
                
                with col_risk2:
                    st.markdown("#### ⚖️ Risque vs Rendement (Nuage de Points)")
                    st.caption("Viser en haut à gauche (Haut rendement, Faible risque).")
                    
                    if kpi_data:
                        fig_scatter = px.scatter(
                            df_kpi,
                            x="Volatilité Ann. (%)",
                            y="Rendement (%)",
                            text="Actif",
                            color="Ratio de Sharpe",
                            color_continuous_scale="RdYlGn",
                            size_max=15
                        )
                        fig_scatter.update_traces(textposition='top center', marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                        
                        # Ajout des lignes médianes
                        median_vol = df_kpi["Volatilité Ann. (%)"].median()
                        median_ret = df_kpi["Rendement (%)"].median()
                        fig_scatter.add_vline(x=median_vol, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                        fig_scatter.add_hline(y=median_ret, line_dash="dash", line_color="rgba(255,255,255,0.3)")
                        
                        fig_scatter.update_layout(
                            margin=dict(t=10, b=0, l=0, r=0),
                            height=350,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#bcbedc'),
                            xaxis=dict(gridcolor='rgba(128,128,128,0.2)', title="Volatilité (Risque)"),
                            yaxis=dict(gridcolor='rgba(128,128,128,0.2)', title="Rendement")
                        )
                        st.plotly_chart(fig_scatter, width='stretch', config={'displayModeBar': False})
                    
                st.markdown("---")
                st.markdown("#### 📉 Évolution de la Volatilité Glissante (30j)")
                st.caption("Visualisez comment le risque (écart-type annualisé) de chaque actif évolue au fil du temps.")
                
                if not returns_df.empty:
                    # Calcul de la volatilité glissante (fenêtre de 30 jours)
                    rolling_vol = returns_df.rolling(window=30).std() * (252 ** 0.5) * 100
                    rolling_vol = rolling_vol.dropna()
                    
                    if not rolling_vol.empty:
                        # Ajout d'un slider pour rendre le nombre de jours configurable
                        days_to_show = st.slider("Nombre de jours à afficher dans le tableau :", min_value=5, max_value=60, value=14, step=1, help="Contrôle le nombre de colonnes de dates dans le tableau de volatilité ci-dessous.")
                        
                        # Transposition pour avoir les Actifs en lignes et Dates en colonnes
                        display_vol = rolling_vol.sort_index(ascending=False).iloc[:days_to_show].T
                        
                        # Remplacer les Tickers par les Noms d'actifs
                        asset_name_map = {}
                        for ticker in display_vol.index:
                            match = df_hold[df_hold['Ticker'] == ticker]
                            if not match.empty:
                                asset_name_map[ticker] = match["Nom de l'actif"].iloc[0]
                            else:
                                asset_name_map[ticker] = ticker
                                
                        display_vol.index = display_vol.index.map(asset_name_map)
                        
                        # Tri du plus volatile au moins volatile (basé sur la date la plus récente)
                        if not display_vol.empty and len(display_vol.columns) > 0:
                            display_vol = display_vol.sort_values(by=display_vol.columns[0], ascending=False)
                        
                        # Calcul évolution hebdo (5 jours de bourse)
                        current = rolling_vol.iloc[-1]
                        prev = rolling_vol.iloc[-6] if len(rolling_vol) >= 6 else current
                        diff = current - prev
                        
                        # Ajustage de l'index de diff pour correspondre à display_vol
                        diff.index = diff.index.map(asset_name_map)
                        
                        # --- Ajout de l'évolution du classement ---
                        ranks_df = rolling_vol.rank(axis=1, ascending=False, method='min')
                        ranks_df.columns = ranks_df.columns.map(asset_name_map)
                        
                        current_rank = ranks_df.iloc[-1]
                        max_date = rolling_vol.index.max()
                        
                        idx_7d = rolling_vol.index[rolling_vol.index <= max_date - pd.Timedelta(days=7)]
                        rank_7d = ranks_df.loc[idx_7d[-1]] if len(idx_7d) > 0 else ranks_df.iloc[0]
                        
                        idx_30d = rolling_vol.index[rolling_vol.index <= max_date - pd.Timedelta(days=30)]
                        rank_30d = ranks_df.loc[idx_30d[-1]] if len(idx_30d) > 0 else ranks_df.iloc[0]
                        
                        def format_rank_diff(val):
                            if pd.isna(val): return "➖ ="
                            if val > 0: return f"🔺 +{int(val)}"
                            if val < 0: return f"🔻 {int(val)}"
                            return "➖ ="
                        
                        display_vol.insert(0, "Evol. Hebdo (pts)", diff)
                        display_vol.insert(0, "Evol. Place 30J", (rank_30d - current_rank).apply(format_rank_diff))
                        display_vol.insert(0, "Evol. Place 7J", (rank_7d - current_rank).apply(format_rank_diff))
                        
                        # Ajout d'une place (rang) en utilisant le rang réel
                        display_vol.insert(0, "Rang", [f"#{int(current_rank.get(idx, i+1))}" for i, idx in enumerate(display_vol.index)])
                        
                        # Formatage des dates en colonnes
                        display_vol.columns = [c if isinstance(c, str) else c.strftime('%d/%m') for c in display_vol.columns]

                        st.dataframe(
                            display_vol.style.format("{:.2f}%", subset=display_vol.columns[4:])
                                             .format("{:+.2f}", subset=["Evol. Hebdo (pts)"])
                                             .background_gradient(cmap='Reds', axis=None, subset=display_vol.columns[4:])
                                             .map(lambda x: 'color: #ff4b4b' if isinstance(x, (int, float)) and x > 0 else 'color: #2ecc71' if isinstance(x, (int, float)) else '', subset=["Evol. Hebdo (pts)"]),
                            width='stretch',
                            height=400
                        )
            else:
                st.info("Il faut au moins 1 actif avec historique pour afficher l'analyse.")
        else:
            st.info("Données historiques insuffisantes pour l'analyse avancée.")

    elif app_page == "💡 Signaux & Opportunités":
        st.subheader("🤖 Synthèse des opportunités")
        st.write("") # Espace ajouté
        st.caption("Analyse croisée entre le consensus des analystes, la tendance technique et les prévisions de l'IA.")
        
        if not df_hold.empty:
            reco_data = []
            all_assets_scores = {}
            
            for _, row in df_hold.iterrows():
                name = row["Nom de l'actif"]
                ticker = row.get("Ticker")
                avis = str(row.get('Avis Analyste', 'N/A'))
                ml_30 = row.get('Proj. 30j (ML)')
                poly_30 = row.get('Proj. 30j (%)')
                perf = row.get('Performance %', 0)
                price = row.get('Prix Actuel')
                mm200 = row.get('MM 200')
                mme9 = row.get('MME 9')
                mme21 = row.get('MME 21')
                macd = row.get('MACD')
                macd_signal = row.get('MACD Signal')
                atr = row.get('ATR')
                highest_50 = row.get('Highest 50')
                bb_haut = row.get('BB Haut')
                stoch_k = row.get('Stoch K')
                vol = row.get('Volume')
                vol_sma_20 = row.get('Vol SMA 20')
                
                # Nettoyage valeurs
                ml_30 = ml_30 if isinstance(ml_30, (int, float)) else 0.0
                poly_30 = poly_30 if isinstance(poly_30, (int, float)) else 0.0
                
                # 0. Indicateurs Techniques (RSI) actuel
                rsi_val = 50
                if ticker in full_ticker_data and not full_ticker_data[ticker].empty:
                    # Utilisation de la valeur pré-calculée dans add_technical_indicators
                    val = full_ticker_data[ticker].iloc[-1].get('RSI')
                    if pd.notnull(val):
                        rsi_val = val

                def get_score_for_offset(offset=0):
                    h_score = 0
                    h_reason = []
                    
                    if offset == 0:
                        h_mme9, h_mme21, h_mm200, h_macd, h_price, h_rsi = mme9, mme21, mm200, macd, price, rsi_val
                        h_bb_haut, h_stoch_k = bb_haut, stoch_k
                        h_macd_sig, h_atr, h_highest_50 = macd_signal, atr, highest_50
                        h_vol, h_vol_sma_20 = vol, vol_sma_20
                    else:
                        if ticker not in full_ticker_data or full_ticker_data[ticker].empty or len(full_ticker_data[ticker]) <= offset:
                            return 0, "Neutre", ["Données insuffisantes"]
                        h_data = full_ticker_data[ticker].iloc[-1 - offset]
                        h_price = h_data.get('Close', 0)
                        h_mme9 = h_data.get('MME_9')
                        h_mme21 = h_data.get('MME_21')
                        h_mm200 = h_data.get('MM_200')
                        h_macd = h_data.get('MACD')
                        h_rsi = h_data.get('RSI', 50)
                        h_bb_haut = h_data.get('BB_Upper')
                        h_stoch_k = h_data.get('Stoch_K')
                        h_macd_sig = h_data.get('MACD_Signal')
                        h_atr = h_data.get('ATR')
                        h_highest_50 = h_data.get('Highest_50')
                        h_vol = h_data.get('Volume')
                        h_vol_sma_20 = h_data.get('Vol_SMA_20')

                    # 1. Analystes
                    if "Achat" in avis or "Buy" in avis: 
                        h_score += 2
                        h_reason.append(f"Analystes: {avis}")
                    elif "Vente" in avis or "Sell" in avis: 
                        h_score -= 2
                        h_reason.append(f"Analystes: {avis}")
                    
                    # 2. IA (Machine Learning)
                    if ml_30 > 2.0: 
                        h_score += 2
                        h_reason.append(f"IA: 🚀 {ml_30:+.1f}%")
                    elif ml_30 < -2.0: 
                        h_score -= 2
                        h_reason.append(f"IA: 📉 {ml_30:+.1f}%")
                    
                    # 3. Technique (Polynomiale, MME, MACD, MM200)
                    if pd.notnull(h_mme9) and pd.notnull(h_mme21):
                        if h_mme9 > h_mme21:
                            h_score += 1
                            h_reason.append("MME: 🟢 (9>21)")
                        elif h_mme9 < h_mme21:
                            h_score -= 1
                            h_reason.append("MME: 🔴 (9<21)")
                    
                    if pd.notnull(h_mm200) and pd.notnull(h_price) and h_price > 0:
                        if h_price > h_mm200:
                            h_score += 1
                            h_reason.append("Fond: 🟢 (>MM200)")
                        else:
                            h_score -= 1
                            h_reason.append("Fond: 🔴 (<MM200)")
                    
                    if poly_30 > 5.0: 
                        h_score += 0.5
                        h_reason.append(f"Trend: 📈 {poly_30:+.1f}%")
                    elif poly_30 < -5.0: 
                        h_score -= 0.5
                        h_reason.append(f"Trend: 📉 {poly_30:+.1f}%")
                    
                    if pd.notnull(h_macd) and pd.notnull(h_macd_sig):
                        if h_macd > 0 and h_macd_sig > 0 and h_macd < h_macd_sig: 
                            h_score -= 1
                            h_reason.append("MACD: Croisement Baissier 🔴")
                        elif h_macd > 0: 
                            h_score += 0.5
                        else: 
                            h_score -= 0.5
                    
                    if pd.notnull(h_rsi):
                        if h_rsi < 30:
                            h_score += 1
                            h_reason.append(f"RSI: Survendu ({h_rsi:.0f})")
                        elif h_rsi > 70:
                            h_score -= 1.5
                            h_reason.append(f"RSI: Surchauffé ({h_rsi:.0f})")
                            
                    if pd.notnull(h_bb_haut) and pd.notnull(h_price):
                        if h_price > h_bb_haut * 0.98:
                            h_score -= 1.5
                            h_reason.append("Excès: Touche BB Haut")
                    
                    if pd.notnull(h_stoch_k):
                        if h_stoch_k > 80:
                            h_score -= 1
                            h_reason.append(f"Stoch: Surchauffe ({h_stoch_k:.0f})")
                        elif h_stoch_k < 20:
                            h_score += 1
                            h_reason.append(f"Stoch: Survendu ({h_stoch_k:.0f})")
                            
                    if pd.notnull(h_highest_50) and pd.notnull(h_atr) and pd.notnull(h_price):
                        if h_price < h_highest_50 - 2.5 * h_atr:
                            h_score -= 1.5
                            h_reason.append("Stop ATR: Rupture tendance 🔴")
                            
                    if pd.notnull(h_vol) and pd.notnull(h_vol_sma_20) and h_vol_sma_20 > 0:
                        if h_vol > 3 * h_vol_sma_20 and pd.notnull(h_mme21) and h_price > h_mme21:
                            h_score -= 1
                            h_reason.append(f"Volume: Climax achat (x{h_vol/h_vol_sma_20:.1f}) ⚠️")
                            
                    if pd.notnull(h_mm200) and h_mm200 > 0 and pd.notnull(h_price):
                        dist_mm200 = (h_price - h_mm200) / h_mm200
                        if dist_mm200 > 0.35:
                            h_score -= 1.5
                            h_reason.append(f"Élastique tendu: +{dist_mm200*100:.0f}% MM200 ⚠️")
                    
                    # 4. Contexte (Buy the dip / Take profit dynamique)
                    if perf < -10.0 and h_score > 0: 
                        h_score += 1
                        h_reason.append(f"Rebond sur chute ({perf:.1f}%)")
                    
                    is_overheated = (pd.notnull(h_rsi) and h_rsi > 65) or (pd.notnull(h_stoch_k) and h_stoch_k > 80) or (pd.notnull(h_price) and pd.notnull(h_bb_haut) and h_price > h_bb_haut * 0.98)
                    if perf > 15.0:
                        if h_score < 0 and is_overheated:
                            h_score -= 1.5
                            h_reason.append(f"Prise de profit stratégique (+{perf:.1f}%)")
                        elif is_overheated:
                            h_score -= 1
                            h_reason.append(f"Alerte marché tendu (+{perf:.1f}%)")

                    h_action = "Neutre"
                    if h_score >= 4: h_action = "Achat Fort 🟢"
                    elif h_score >= 1.5: h_action = "Renforcer 🟢"
                    elif h_score <= -4: h_action = "Vente Forte 🔴"
                    elif h_score <= -1.5: h_action = "Alléger 🔴"
                    
                    return h_score, h_action, h_reason

                score, action, reason = get_score_for_offset(0)
                _, action_j3, reason_j3 = get_score_for_offset(3)
                _, action_j7, reason_j7 = get_score_for_offset(7)
                
                all_assets_scores[name] = {
                    "score": score,
                    "action": action,
                    "reasons": reason
                }
                
                if action != "Neutre" or action_j3 != "Neutre" or action_j7 != "Neutre":
                    reco_data.append({
                        "Actif": name,
                        "Action Suggérée": action,
                        "Score": score,
                        "Raisons Clés": ", ".join(reason),
                        "Action Suggérée J-3": action_j3,
                        "Raisons Clés J-3": ", ".join(reason_j3),
                        "Action Suggérée J-7": action_j7,
                        "Raisons Clés J-7": ", ".join(reason_j7),
                    })
            
            if reco_data:
                df_reco = pd.DataFrame(reco_data).sort_values(by="Score", ascending=False)
                st.dataframe(
                    df_reco[["Actif", "Action Suggérée", "Raisons Clés", "Action Suggérée J-3", "Raisons Clés J-3", "Action Suggérée J-7", "Raisons Clés J-7"]],
                    width="stretch",
                    hide_index=True
                )
            else:
                st.info("Aucun signal fort détecté sur le portefeuille actuel (Consensus Neutre).")
            
            # --- JAUGE DE SCORE ---
            st.markdown("---")
            st.subheader("🧭 Jauge de Recommandation")
            st.write("") # Espace ajouté

            col_gauge_sel, col_gauge_view = st.columns([1, 2])
            with col_gauge_sel:
                selected_asset_gauge = st.selectbox("Sélectionnez un actif :", options=df_hold["Nom de l'actif"].unique())
            
            with col_gauge_view:
                if selected_asset_gauge in all_assets_scores:
                    data_gauge = all_assets_scores[selected_asset_gauge]
                    score_val = data_gauge["score"]
                    
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score_val,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"<b>{data_gauge['action']}</b>", 'font': {'size': 20}},
                        gauge = {
                            'axis': {'range': [-10, 10], 'tickwidth': 1},
                            'bar': {'color': "black"},
                            'steps': [
                                {'range': [-10, -4], 'color': "#FF5252"},  # Vente Forte
                                {'range': [-4, -1.5], 'color': "#FFAB91"},  # Vente
                                {'range': [-1.5, 1.5], 'color': "#EEEEEE"},   # Neutre
                                {'range': [1.5, 4], 'color': "#A5D6A7"},    # Achat
                                {'range': [4, 10], 'color': "#43A047"}     # Achat Fort
                            ],
                        }
                    ))
                    fig_gauge.update_layout(
                        height=250, 
                        margin=dict(t=30, b=10, l=30, r=30),
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#bcbedc')
                    )
                    st.plotly_chart(fig_gauge, width='stretch')
                    
                    if data_gauge["reasons"]:
                        st.info(f"📝 **Facteurs :** {', '.join(data_gauge['reasons'])}")
                    else:
                        st.caption("Aucun facteur technique ou fondamental marquant.")
        else:
            st.info("Chargez un portefeuille pour voir les recommandations.")

    elif app_page == "⚙️ Configuration & Archives":
        st.subheader("💸 Historique des Ventes")
        # Affiche le détail des actifs qui ont été vendus.
        if not df_sold.empty:
            df_sold["P&L"] = df_sold["Prix de vente"] - df_sold["Total de l'actif"]
            st.dataframe(
                df_sold[["Nom de l'actif", "Total de l'actif", "Prix de vente", "P&L", "Date de vente"]],
                column_config={
                    "P&L": st.column_config.NumberColumn("Gain/Perte Net", format="%.2f €"),
                    "Total de l'actif": st.column_config.NumberColumn("Coût Achat", format="%.2f €"),
                    "Date de vente": st.column_config.DateColumn("Date Vente"),
                },
                width="stretch"
            )
        else:
            st.info("Aucune vente enregistrée.")
        
        st.markdown("---")
        st.subheader("⚙️ Configuration Tickers & Devises")
        st.write("") # Espace ajouté
        st.markdown("""
        Vérifiez ici la correspondance de vos actifs. 
        **Si la détection 'USD' est fausse pour un actif hors crypto, forcez 'EUR' dans la colonne Devise.**
        """)
        
        if not df_hold.empty:
            unique_assets = df_hold["Nom de l'actif"].unique()
            editor_data = []
            
            for asset in unique_assets:
                # 1. Récupérer le ticker : priorité à la config sauvegardée, sinon détection auto.
                if asset in st.session_state.saved_tickers:
                    ticker = st.session_state.saved_tickers[asset]
                else:
                    subset = df_hold[df_hold["Nom de l'actif"] == asset]
                    ticker = subset["Ticker"].iloc[0] if not subset.empty else ""
                
                # 2. Devise (Priorité : Mémoire > Heuristique)
                if asset in st.session_state.saved_currencies:
                    currency = st.session_state.saved_currencies[asset]
                else:
                    # Par défaut, on devine
                    currency = "USD" if is_ticker_usd_heuristic(ticker) else "EUR"
                
                editor_data.append({
                    "Nom de l'actif": asset, 
                    "Symbole Yahoo (Ticker)": ticker,
                    "Devise": currency
                })
            
            df_editor_input = pd.DataFrame(editor_data)
            
            # st.data_editor fournit une interface de type tableur pour modifier les données.
            edited_df = st.data_editor(
                df_editor_input,
                column_config={
                    "Nom de l'actif": st.column_config.TextColumn("Actif", disabled=True),
                    "Symbole Yahoo (Ticker)": st.column_config.TextColumn("Symbole Yahoo", required=False),
                    "Devise": st.column_config.SelectboxColumn(
                        "Devise", 
                        options=["EUR", "USD"],
                        help="Forcez 'EUR' pour éviter la conversion si l'actif est coté en Euros.",
                        required=True
                    )
                },
                hide_index=True,
                width="stretch",
                num_rows="fixed",
                key="ticker_config_editor"
            )
            
            if not edited_df.empty:
                # Sauvegarde des modifications dans le session_state pour les conserver.
                new_tickers = dict(zip(edited_df["Nom de l'actif"], edited_df["Symbole Yahoo (Ticker)"]))
                st.session_state.saved_tickers.update(new_tickers)
                
                new_currencies = dict(zip(edited_df["Nom de l'actif"], edited_df["Devise"]))
                st.session_state.saved_currencies.update(new_currencies)
                
                st.success("✅ Configuration sauvegardée. Le tableau de bord se mettra à jour.")
        
        st.markdown("---")
        st.subheader("🔍 Recherche de Ticker (FinanceDatabase)")
        st.write("") # Espace ajouté
        st.caption("Si vous ne trouvez pas le symbole Yahoo de votre actif, utilisez cet outil pour le chercher dans une base de données mondiale.")
        
        if fd is None:
            st.warning("⚠️ Le module `financedatabase` n'est pas installé. Installez-le avec `pip install financedatabase`.")
        else:
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                fd_query = st.text_input("Nom de l'actif (ex: LVMH, Bitcoin, MSCI World)", key="fd_q")
            with c2:
                fd_cat = st.selectbox("Catégorie", ["Actions", "ETFs", "Cryptos", "Indices", "Devises"], key="fd_c")
            with c3:
                st.write("") 
                st.write("") 
                if st.button("🔎 Rechercher", key="fd_btn") and fd_query:
                    with st.spinner("Recherche dans la base de données..."):
                        results = search_ticker_in_db(fd_query, fd_cat)
                        if not results.empty:
                            st.dataframe(results, width="stretch", hide_index=True)
                        else:
                            st.warning("Aucun résultat trouvé.")

    # --- LOGIQUE DE RAFRAÎCHISSEMENT AUTOMATIQUE ---
    # Utilisation du composant natif streamlit_autorefresh pour éviter que time.sleep() ne bloque
    # le thread principal du serveur Streamlit Cloud.
    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            # L'intervalle est en millisecondes. On le passe à st_autorefresh.
            # Cela va déclencher un st.rerun() automatiquement depuis le navigateur (frontend)
            # sans bloquer le backend avec time.sleep().
            st_autorefresh(interval=refresh_interval * 1000, key="data_autorefresh")
            
        except ImportError:
            st.error("Le module `streamlit-autorefresh` n'est pas installé. Veuillez l'ajouter au fichier `requirements.txt` (pip install streamlit-autorefresh).")

if __name__ == "__main__":
    print("Lancez via : streamlit run TESTETATDESLIEUX_ML.py")