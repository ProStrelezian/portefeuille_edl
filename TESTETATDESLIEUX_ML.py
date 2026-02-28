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
from modules.ml_models import calculate_ml_prediction, calculate_smart_prediction, get_llm_analysis, stream_llm_response
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
        
        # AJOUT: Stochastic Oscillator (14)
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        denom = high_max - low_min
        # Gestion division par zéro et calcul %K
        df['Stoch_K'] = np.where(denom == 0, 50, 100 * ((close - low_min) / denom))
    else:
        df['ATR'] = np.nan
        df['Stoch_K'] = np.nan
        
    # AJOUT: Bandes de Bollinger (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
        
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

        tickers_to_fetch = list(set(valid_tickers + ["EURUSD=X"]))
        
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
    tickers_to_fetch = list(set(valid_tickers + ["EURUSD=X"]))
    
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
        return {}, {}, 1.0, {}, pd.DataFrame(), pd.DataFrame(), {}
    
    # 1. Récupération des données
    full_ticker_data, data_intraday, data_daily_close = fetch_historical_data(tickers)
    data_live = fetch_real_time_data(tickers)
    
    valid_tickers = [t for t in tickers if t and isinstance(t, str)]
    
    current_prices = {}
    reference_prices = {}
    history_data = {}
    eur_usd_rate = 1.0
    
    # 2. Extraction du taux EUR/USD (Live > Intraday > Daily)
    rate_series = get_ticker_data(data_live, "EURUSD=X", 'Close')
    if rate_series.empty:
        rate_series = get_ticker_data(data_intraday, "EURUSD=X", 'Close')
    
    if not rate_series.empty:
            valid_rate = rate_series.dropna()
            if not valid_rate.empty:
                r = float(valid_rate.iloc[-1])
                if r > 0: eur_usd_rate = r

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
            history_data[ticker] = df_hist['Close'].tolist()
            # Reference = Clôture veille (avant-dernière valeur si la dernière est aujourd'hui, ou dernière si data pas à jour)
            # Simplification : on prend l'avant-dernière valeur du daily
            if len(df_hist) > 1:
                reference_prices[ticker] = float(df_hist['Close'].iloc[-2])
            else:
                reference_prices[ticker] = float(df_hist['Close'].iloc[-1])
        else:
            reference_prices[ticker] = 0.0
            history_data[ticker] = []
            
    return current_prices, reference_prices, eur_usd_rate, history_data, data_intraday, data_daily_close, full_ticker_data

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
        money_cols = ["Valeur d'une unité", "Total de l'actif", "Frais", "Gain de staking", "Dividende", "Prix de vente", "Unités"]
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
        ["📊 Tableau de Bord", "📈 Performance & Prévisions", "🧠 Analyse Technique & Risques", "💡 Signaux & Opportunités", "🤖 Assistant IA", "⚙️ Configuration & Archives"],
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

    st.markdown("---")
    st.header("🤖 Clé API IA (Optionnel)")
    api_key_input = st.text_input("Votre clé API (Gemini ou OpenAI)", type="password", value=st.session_state.get('api_key', ''), help="Si vous n'avez pas de fichier secrets.toml, entrez votre clé ici pour affiner les prédictions. Conservée uniquement pour la session en cours.")
    if api_key_input:
        st.session_state.api_key = api_key_input

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
                    market_prices, ref_prices, eur_usd_rate, history_data, raw_history_df, daily_history_df, full_ticker_data = fetch_market_data(unique_tickers)
                    asset_details = fetch_asset_details(unique_tickers)
            else:
                market_prices, ref_prices, eur_usd_rate, history_data, raw_history_df, daily_history_df, full_ticker_data = fetch_market_data(unique_tickers)
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
            market_prices, ref_prices, eur_usd_rate, history_data, asset_details, raw_history_df, daily_history_df, full_ticker_data = {}, {}, 1.0, {}, {}, pd.DataFrame(), pd.DataFrame(), {}
        
        if eur_usd_rate != 1.0:
            st.sidebar.markdown("---")
            st.sidebar.metric("Taux change (1€ = $)", f"{eur_usd_rate:.7f} $")

        # --- ENRICHISSEMENT DU DATAFRAME AVEC LES DONNÉES DE MARCHÉ ---
        def get_row_currency(asset_name, ticker):
            """Détermine la devise pour une ligne : priorité à la config manuelle, sinon heuristique."""
            # Priorité absolue à la configuration manuelle
            if asset_name in st.session_state.saved_currencies:
                return st.session_state.saved_currencies[asset_name]
            # Sinon heuristique
            return "USD" if is_ticker_usd_heuristic(ticker) else "EUR"

        df_hold['Devise'] = df_hold.apply(lambda x: get_row_currency(x["Nom de l'actif"], x['Ticker']), axis=1)

        def get_converted_price(price_dict, ticker, currency, rate):
            """Récupère un prix et le convertit en EUR si nécessaire."""
            raw_price = price_dict.get(ticker, 0.0)
            
            if raw_price == 0.0: return 0.0
            
            # Conversion seulement si la devise est USD
            if currency == "USD":
                return raw_price / rate
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
                return (None, None)
            # Création d'un tuple hashable pour la mise en cache
            data_tuple = tuple(df_t[required_cols].itertuples(index=False, name=None))
            _pred_price_30, pct_change_30 = calculate_ml_prediction(data_tuple, days_ahead=30)
            _pred_price_7, pct_change_7 = calculate_ml_prediction(data_tuple, days_ahead=7)
            return (pct_change_30, pct_change_7)

        def get_technical_indicators(ticker):
            df_t = full_ticker_data.get(ticker, pd.DataFrame())
            
            # --- FIX ROBUSTESSE CACHE ---
            # Si les nouvelles colonnes (BB, Stoch) sont absentes à cause d'un cache obsolète, on recalcule.
            if df_t is not None and not df_t.empty and ('BB_Upper' not in df_t.columns or 'Stoch_K' not in df_t.columns):
                df_t = add_technical_indicators(df_t)
                full_ticker_data[ticker] = df_t
            
            if df_t is None or df_t.empty or 'MM_200' not in df_t.columns:
                return (None, None, None, None, None, None, None, None)
            
            last_row = df_t.iloc[-1]
            return (
                last_row.get('MM_200'),
                last_row.get('MME_9'),
                last_row.get('MME_21'),
                last_row.get('MACD'),
                last_row.get('ATR'),
                last_row.get('BB_Upper'),
                last_row.get('BB_Lower'),
                last_row.get('Stoch_K')
            )

        # --- ENRICHISSEMENT EN UNE SEULE PASSE ---
        # On parcourt df_hold une seule fois pour calculer toutes les colonnes dérivées,
        # au lieu de faire 10+ .apply() successifs qui repassent chaque fois sur tout le DataFrame.
        enriched_rows = []
        for row in df_hold.to_dict('records'):
            ticker = row['Ticker']
            asset_name = row["Nom de l'actif"]
            devise = get_row_currency(asset_name, ticker)

            # Prix convertis
            prix_actuel = get_converted_price(market_prices, ticker, devise, eur_usd_rate)
            prix_ref = get_converted_price(ref_prices, ticker, devise, eur_usd_rate)

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

            # Conversion des fourchettes de prix en EUR si USD
            if devise == 'USD':
                bas_30 = bas_30 / eur_usd_rate if bas_30 is not None else None
                haut_30 = haut_30 / eur_usd_rate if haut_30 is not None else None
                bas_7 = bas_7 / eur_usd_rate if bas_7 is not None else None
                haut_7 = haut_7 / eur_usd_rate if haut_7 is not None else None

            # Prédictions ML
            ml_pct_30, ml_pct_7 = get_ml_prediction_display(ticker)

            # Indicateurs techniques
            mm200, mme9, mme21, macd, atr, bb_haut, bb_bas, stoch_k = get_technical_indicators(ticker)
            # Conversion EUR si USD (Stoch K est un % donc pas de conversion)
            if devise == 'USD':
                for v in [mm200, mme9, mme21, macd, atr, bb_haut, bb_bas]:
                    if v is not None: v = v / eur_usd_rate
                # Reconversion propre via division (les variables locales ne mutent pas les originaux)
                def _eur(v): return v / eur_usd_rate if v is not None else None
                mm200, mme9, mme21, macd, atr, bb_haut, bb_bas = (
                    _eur(mm200), _eur(mme9), _eur(mme21), _eur(macd),
                    _eur(atr), _eur(bb_haut), _eur(bb_bas)
                )

            enriched_rows.append({
                'Devise': devise,
                'Prix Actuel': prix_actuel,
                'Prix Reference': prix_ref,
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
                'MM 200': mm200,
                'MME 9': mme9,
                'MME 21': mme21,
                'MACD': macd,
                'ATR': atr,
                'BB Haut': bb_haut,
                'BB Bas': bb_bas,
                'Stoch K': stoch_k,
            })

        df_enriched = pd.DataFrame(enriched_rows, index=df_hold.index)
        # On supprime de df_hold les colonnes qui vont être ajoutées par df_enriched
        # pour éviter les doublons (ex: 'Devise' déjà présente) qui causent DuplicateError.
        cols_to_drop = [c for c in df_enriched.columns if c in df_hold.columns]
        df_hold = df_hold.drop(columns=cols_to_drop)
        df_hold = pd.concat([df_hold, df_enriched], axis=1)

        # Conversion explicite en numérique pour gérer les None (qui deviennent NaN)
        cols_tech = ['MM 200', 'MME 9', 'MME 21', 'MACD', 'ATR', 'BB Haut', 'BB Bas', 'Stoch K']
        for col in cols_tech:
            df_hold[col] = pd.to_numeric(df_hold[col], errors='coerce')
        
        # Calcul du Signal Technique (vectorisé)
        conditions = [
            (df_hold['Prix Actuel'] < df_hold['BB Bas']), # Prix sous la bande basse -> Rebond possible
            (df_hold['Prix Actuel'] > df_hold['BB Haut']), # Prix sur la bande haute -> Correction possible
            (df_hold['MME 9'] > df_hold['MME 21']),
            (df_hold['MME 9'] < df_hold['MME 21'])
        ]
        choices = ["Sursell (BB) 🟢", "Surchauffe (BB) 🔴", "Achat (MME) 🟢", "Vente (MME) 🔴"]
        df_hold['Signal Technique'] = np.select(conditions, choices, default="N/A")
        
        # Calculs de valeurs (vectorisés)
        df_hold['Valeur Actuelle'] = np.where(df_hold['Prix Actuel'] > 0, df_hold['Unités'] * df_hold['Prix Actuel'], df_hold["Total de l'actif"])
        df_hold['Valeur Reference'] = np.where(df_hold['Prix Reference'] > 0, df_hold['Unités'] * df_hold['Prix Reference'], df_hold["Total de l'actif"])
        
        # Calcul de l'évolution journalière (vectorisé)
        df_hold['Evol. Jour %'] = np.where(
            df_hold['Valeur Reference'] > 0,
            (df_hold['Valeur Actuelle'] - df_hold['Valeur Reference']) / df_hold['Valeur Reference'] * 100,
            0.0
        )
        
        # Intégration du Staking et des Dividendes dans la performance (P&L)
        staking_col = df_hold['Gain de staking'].fillna(0) if 'Gain de staking' in df_hold.columns else 0
        div_col = df_hold['Dividende'].fillna(0) if 'Dividende' in df_hold.columns else 0
        df_hold['Plus-value Latente'] = df_hold['Valeur Actuelle'] - df_hold["Total de l'actif"] + staking_col + div_col
        
        # Calcul de la performance (vectorisé)
        df_hold['Performance %'] = np.where(
            df_hold["Total de l'actif"] > 0,
            (df_hold['Plus-value Latente'] / df_hold["Total de l'actif"]) * 100,
            0
        )

    # --- AFFICHAGE DE LA PAGE SÉLECTIONNÉE ---

    if app_page == "📊 Tableau de Bord":
        st.markdown("###  📊 Vue d'ensemble")
        # Calcul des métriques globales du portefeuille.
        total_invested = df_hold["Total de l'actif"].sum()
        current_value_total = df_hold["Valeur Actuelle"].sum()
        
        # Calcul de la performance globale incluant Staking et Dividendes des actifs détenus
        total_staking_hold = df_hold["Gain de staking"].sum() if "Gain de staking" in df_hold.columns else 0
        total_div_hold = df_hold["Dividende"].sum() if "Dividende" in df_hold.columns else 0
        # On utilise la somme de la colonne calculée précédemment
        total_pnl_hold = df_hold["Plus-value Latente"].sum()

        # Calcul de la variation journalière (depuis la clôture de la veille).
        reference_value_total = df_hold["Valeur Reference"].sum()
        daily_change_value = current_value_total - reference_value_total
        daily_change_percent = (daily_change_value / reference_value_total * 100) if reference_value_total > 0 else 0.0
        
        # Calcul des gains réalisés (plus-values de vente + dividendes + staking).
        capital_gains = df_sold["Prix de vente"].sum() - df_sold["Total de l'actif"].sum() if not df_sold.empty else 0 # Gains sur ventes
        dividends = df["Dividende"].sum() if "Dividende" in df.columns else 0
        staking = df["Gain de staking"].sum() if "Gain de staking" in df.columns else 0
        realized_gains = capital_gains + dividends + staking

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Investi", f"{total_invested:,.2f} €")
        col2.metric("Valeur Actuelle", f"{current_value_total:,.2f} €", delta=f"{daily_change_value:+.2f}€ ({daily_change_percent:+.3f}%)", help="Variation par rapport à la clôture de la veille.")
        col3.metric("Performance Totale", f"{(total_pnl_hold/total_invested)*100:+.2f} %" if total_invested > 0 else "0%", delta=f"{total_pnl_hold:+.2f}€", help=f"Inclut Plus-value latente + Staking ({total_staking_hold:.2f}€) + Dividendes ({total_div_hold:.2f}€)")
        col4.metric("Gains Réalisés", f"{realized_gains:,.2f} €", help=f"Plus-values: {capital_gains:.2f}€ | Dividendes/Staking: {dividends+staking:.2f}€")
        val_gains = f"{realized_gains/total_invested*100:,.3f} %" if total_invested > 0 else "0.00 %"
        col5.metric("Gains Réalisés/Total", val_gains)

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
                    df_work = df_work.ffill().fillna(0)

                    rate_series = pd.Series(1.0, index=df_work.index)
                    if "EURUSD=X" in df_work.columns:
                        eur_data = df_work["EURUSD=X"]
                        if isinstance(eur_data, pd.DataFrame) and 'Close' in eur_data.columns:
                            rate_series = eur_data['Close']
                        elif isinstance(eur_data, pd.Series):
                            rate_series = eur_data
                    rate_series = rate_series.replace(0, np.nan).ffill().fillna(1.0)

                    port_series = pd.Series(0.0, index=df_work.index)
                    
                    for row in df_hold.to_dict('records'):
                        t = row['Ticker']
                        if t in df_work.columns:
                            data_t = df_work[t]
                            ps = pd.Series(dtype=float)
                            if isinstance(data_t, pd.DataFrame):
                                if 'Close' in data_t.columns:
                                    ps = data_t['Close']
                                elif len(data_t.columns) > 0:
                                    ps = data_t.iloc[:, 0]
                            elif isinstance(data_t, pd.Series):
                                ps = data_t
                                
                            if not ps.empty:
                                ps = ps.reindex(port_series.index, method='ffill').fillna(0)
                                if row['Devise'] == 'USD':
                                    ps = ps / rate_series
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
                    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
                    kcol1.metric("Ratio de Sharpe (LT)", f"{kpis_hist['Sharpe']:.2f}", help="Calculé sur 2 ans d'historique. Indicateur d'efficience (Rendement ajusté au risque). >1 est bon, >2 est fantastique.")
                    kcol2.metric("Max Drawdown (LT)", f"{kpis_hist['Max Drawdown']:.2f} %", help="Calculé sur 2 ans d'historique. Perte maximale (du plus haut au plus bas). Mesure votre pire scénario.")
                    kcol3.metric("Volatilité Annuelle (LT)", f"{kpis_hist['Volatilité']:.2f} %", help="Calculé sur 2 ans d'historique. Indice de turbulence. Plus le % est élevé, plus le portefeuille fait des montagnes russes.")
                    if not daily_portfolio_30d.empty:
                        kcol4.metric("Rendement (30 Jours)", f"{kpis_30d['Period Return']:.2f} %", help="Performance totale du portefeuille uniquement sur les 30 derniers jours.")
                    else:
                        kcol4.metric("Rendement (30 Jours)", "0.00 %")
                    st.markdown("---")

                if not portfolio_last_30d.empty:
                    st.subheader(" 📈 Évolution de la valeur (30 derniers jours)")
                    
                    # Calcul dynamique de l'échelle Y pour zoomer sur les variations
                    y_min = portfolio_last_30d.min()
                    y_max = portfolio_last_30d.max()
                    y_margin = (y_max - y_min) * 0.1 if y_max > y_min else y_max * 0.01

                    fig_evol = go.Figure()
                    fig_evol.add_trace(go.Scatter(
                        x=portfolio_last_30d.index, 
                        y=portfolio_last_30d.values,
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
                    st.plotly_chart(fig_evol, width='stretch')
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
                df_chart = df_hold.sort_values(by="Plus-value Latente", ascending=True) # Tri pour afficher les plus gros gains en haut.
                # Couleurs plus modernes (Vert Néon / Rouge Néon)
                df_chart['Color'] = df_chart['Plus-value Latente'].apply(lambda x: '#00ff9d' if x >= 0 else '#ff0055')
                df_chart['Perf_Pct'] = df_chart.apply(lambda x: (x['Plus-value Latente'] / x["Total de l'actif"] * 100) if x["Total de l'actif"] > 0 else 0, axis=1)
                
                # Calcul dynamique de la hauteur (40px par barre + marge) pour lisibilité sur mobile
                dynamic_height = max(350, len(df_chart) * 40)
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    y=df_chart["Nom de l'actif"], x=df_chart['Plus-value Latente'], orientation='h',
                    marker_color=df_chart['Color'], 
                    text=df_chart['Plus-value Latente'].apply(lambda x: f"{x:+.2f} €"), 
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Gain/Perte: %{x:.2f} €<br>Performance: %{customdata:.2f}%<extra></extra>',
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
            *   **Performance %** : Votre gain ou perte total depuis l'achat (incluant dividendes/staking, qui agissent comme un bonus fluctuant réduisant les pertes).
            
            **2. Indicateurs Techniques (La "Météo" du marché)**
            *   **Signal** : Combine MME (Tendance) et Bollinger (Extrêmes). "Sursell" = Prix anormalement bas (Opportunité ?).
            *   **ATR (Volatilité)** : Indique la nervosité de l'actif. Un chiffre élevé = gros mouvements de prix (risque plus élevé).
            *   **MACD** : Indicateur d'élan. Positif = Poussée haussière. Négatif = Poussée baissière.
            *   **MM 200** : La "Juge de Paix". Si le prix est au-dessus, la tendance de fond (long terme) est saine.
            *   **Bandes Bollinger** : Canaux de volatilité. Si le prix touche le haut, risque de correction. Si bas, rebond possible.
            *   **Stoch %K** : Oscillateur (0-100). >80 = Surchauffe (Vente?), <20 = Survendu (Achat?).
            
            **3. Prédictions : 🤖 IA (XGBoost) vs 📐 Polynomiale**
            *   **🤖 (ML XGBoost)** : Un modèle de machine learning entraîné sur les indicateurs techniques (RSI, MACD, Volume, ATR…). Il détecte des patterns non-linéaires et est souvent plus fiable pour anticiper les retournements de tendance. **Colonne "7j/30j 🤖 %".**
            *   **📐 (Poly)** : Une **régression polynomiale de degré 2** (mathématique pure, pas d'IA). Elle prolonge simplement la courbe de tendance actuelle. Utile uniquement si la tendance est stable et régulière. **Colonne "7j/30j 📐 %".** ⚠️ *Ne pas confondre avec une prédiction ML.*
            """)

        st.write("") # Espace ajouté
        # Tri du DataFrame pour l'affichage par valeur actuelle décroissante
        df_details_sorted = df_hold.sort_values(by="Valeur Actuelle", ascending=False)

        # st.dataframe est utilisé pour un affichage interactif avec des mini-graphiques.
        st.dataframe(
            df_details_sorted[["Nom de l'actif", "Type", "Prix Actuel", "Valeur Actuelle", "Evol. Jour %", "Evol. Hebdo %", "Trend 7j", "Performance %", "Signal Technique", "Stoch K", "ATR", "MACD", "MM 200", "BB Haut", "BB Bas", "Proj. 7j (ML)", "Proj. 7j (%)", "Proj. 7j Bas", "Proj. 7j Haut", "Proj. 30j (ML)", "Proj. 30j (%)", "Proj. 30j Bas", "Proj. 30j Haut", "Historique"]]
            .style
            .map(style_trend_text, subset=['Evol. Jour %', 'Evol. Hebdo %', 'Performance %', 'MACD', 'Proj. 7j (%)', 'Proj. 30j (%)', 'Proj. 7j (ML)', 'Proj. 30j (ML)'])
            .map(style_signal_text, subset=['Signal Technique']),
            column_config={
                "Nom de l'actif": st.column_config.TextColumn("Actif", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Prix Actuel": st.column_config.NumberColumn("Cours Actuel", format="%.2f €"),
                "Valeur Actuelle": st.column_config.NumberColumn("Val. Actuelle", format="%.2f €"),
                "Evol. Jour %": st.column_config.NumberColumn("Evol. Jour", format="%+.2f %%", help="Variation par rapport à la clôture précédente"),
                "Evol. Hebdo %": st.column_config.NumberColumn("Evol. Hebdo", format="%+.2f %%", help="Variation sur 7 jours (5 jours de bourse)"),
                "Trend 7j": st.column_config.LineChartColumn("Trend 7j", width="small", help="Tendance des 7 derniers jours"),
                "Performance %": st.column_config.NumberColumn("Perf %", format="%+.2f %%"),
                "Signal Technique": st.column_config.TextColumn("Signal", help="MME 9/21 ou Bandes Bollinger"),
                "MM 200": st.column_config.NumberColumn("MM 200", format="%.2f €", help="Moyenne Mobile Simple 200j"),
                "BB Haut": st.column_config.NumberColumn("BB Haut", format="%.2f €", help="Bande de Bollinger Haute (20, 2)"),
                "BB Bas": st.column_config.NumberColumn("BB Bas", format="%.2f €", help="Bande de Bollinger Basse (20, 2)"),
                "Stoch K": st.column_config.NumberColumn("Stoch %K", format="%.0f", help="Oscillateur Stochastique (14, 3)"),
                "MACD": st.column_config.NumberColumn("MACD", format="%.2f", help="MACD (12, 26)"),
                "ATR": st.column_config.NumberColumn("ATR (14)", format="%.2f €", help="Average True Range (Volatilité)"),
                "Proj. 7j (%)": st.column_config.NumberColumn("7j 📐 Poly %", format="%+.2f %%", help="Projection par régression polynomiale (mathématique, pas ML) sur 7 jours"),
                "Proj. 7j (ML)": st.column_config.NumberColumn("7j 🤖 ML %", format="%+.2f %%", help="Projection XGBoost (Machine Learning) sur 7 jours"),
                "Proj. 7j Bas": st.column_config.NumberColumn("7j Min", format="%.2f €", help="Fourchette basse polynomiale 7j"),
                "Proj. 7j Haut": st.column_config.NumberColumn("7j Max", format="%.2f €", help="Fourchette haute polynomiale 7j"),
                "Proj. 30j (%)": st.column_config.NumberColumn("30j 📐 Poly %", format="%+.2f %%", help="Projection par régression polynomiale (mathématique, pas ML) sur 30 jours"),
                "Proj. 30j (ML)": st.column_config.NumberColumn("30j 🤖 ML %", format="%+.2f %%", help="Projection XGBoost (Machine Learning) sur 30 jours"),
                "Proj. 30j Bas": st.column_config.NumberColumn("30j Min", format="%.2f €", help="Fourchette basse 30j"),
                "Proj. 30j Haut": st.column_config.NumberColumn("30j Max", format="%.2f €", help="Fourchette haute 30j"),
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
                df_chart = df_full.last("6ME")

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

                # 3. Graphique en barres pour le volume
                colors = ['#2ecc71' if row['Close'] >= row['Open'] else '#ff4b4b' for index, row in df_chart.iterrows()]
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
                    # Génération des dates futures (Jours ouvrés)
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
                # On utilise les données complètes pour le calcul ML
                data_tuple = tuple(df_full[['Close', 'High', 'Low', 'Volume']].itertuples(index=False, name=None))
                pred_price_ml, _ = calculate_ml_prediction(data_tuple, days_ahead=30)
                
                if pred_price_ml:
                    target_date = df_chart.index[-1] + pd.Timedelta(days=30)
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
                if st.button("🧠 Affiner les prédictions via l'IA (LLM)", key=f"llm_{ticker}", help="Demande à une IA (Gemini ou GPT-4o) de synthétiser ces signaux."):
                    api_key = st.session_state.get('api_key', '')
                    prompt_text = "\n".join(analysis_points)
                    prompt = f"Agis comme un analyste quantitatif expert (Hedge Fund). Affine l'analyse pour l'actif {selected_asset} dont le prix actuel est {close_p:.2f}.\nVoici les signaux techniques détectés :\n{prompt_text}\nConsignes :\n1. Synthétise la situation\n2. Quelle est la zone de risque actuelle ?\n3. Paragraphe conclusif très analytique.\nMax 3 paragraphes concis."
                    st.markdown("#### 🤖 Analyse Quantitative de l'IA")
                    stream = stream_llm_response(prompt, user_api_key=api_key)
                    st.write_stream(stream)
                
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
        st.subheader("🧠 Analyse de Risque & Corrélation")
        st.write("") # Espace ajouté
        if not daily_history_df.empty:
            # Conversion en DataFrame si Série unique
            if isinstance(daily_history_df, pd.Series):
                daily_history_df = daily_history_df.to_frame()

            # Nettoyage : On garde uniquement les tickers présents dans le portefeuille
            valid_cols = [t for t in daily_history_df.columns if t in df_hold['Ticker'].values]
            
            if len(valid_cols) > 1:
                # Le remplissage des données (ffill) est maintenant fait en amont dans la fonction fetch_historical_data.
                analysis_df = daily_history_df[valid_cols].copy()
                # Calcul des rendements quotidiens
                returns_df = analysis_df.pct_change(fill_method=None).dropna()
                
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    st.markdown("#### 🔥 Matrice de Corrélation")    
                    st.caption("Mesure à quel point vos actifs bougent ensemble (1 = identique, -1 = opposé).")
                    corr_matrix = returns_df.corr()
                    fig_corr = px.imshow(
                        corr_matrix, 
                        text_auto=".2f", 
                        color_continuous_scale='RdBu_r', 
                        zmin=-1, zmax=1,
                        aspect="auto"
                    )
                    fig_corr.update_layout(
                        margin=dict(t=30, b=0, l=0, r=0), 
                        height=350,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#bcbedc'),
                        title_font=dict(size=16, color='#edf1f5')
                    )
                    st.plotly_chart(fig_corr, width='stretch')
                
                with col_risk2:
                    st.markdown("#### ⚡ Volatilité (Risque)")
                    st.caption("Écart-type annualisé (Plus c'est haut, plus l'actif est instable).")
                    # Volatilité annualisée (252 jours de bourse)
                    volatility = returns_df.std() * (252 ** 0.5) * 100
                    vol_df = pd.DataFrame({'Actif': volatility.index, 'Volatilité (%)': volatility.values})
                    vol_df = vol_df.sort_values('Volatilité (%)', ascending=False)
                    
                    fig_vol = px.bar(
                        vol_df, x='Volatilité (%)', y='Actif', orientation='h',
                        color='Volatilité (%)', color_continuous_scale='Reds'
                    )
                    fig_vol.update_layout(
                        margin=dict(t=30, b=0, l=0, r=0), 
                        height=350, 
                        yaxis={'categoryorder':'total ascending'},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#bcbedc'),
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)')
                    )
                    st.plotly_chart(fig_vol, width='stretch')
                
                st.markdown("---")
                st.markdown("#### 📉 Évolution de la Volatilité")
                st.caption("Visualisez comment le risque (écart-type annualisé) de chaque actif a évolué récemment.")
                
                # Ajout d'un slider pour rendre le nombre de jours configurable
                days_to_show = st.slider("Nombre de jours à afficher dans le tableau", min_value=5, max_value=60, value=14, step=1, help="Contrôle le nombre de colonnes de dates dans le tableau de volatilité ci-dessous.")
                
                # Calcul de la volatilité glissante (fenêtre de 30 jours)
                rolling_vol = returns_df.rolling(window=30).std() * (252 ** 0.5) * 100
                rolling_vol = rolling_vol.dropna()
                
                if not rolling_vol.empty:
                    # Transposition pour avoir les Actifs en lignes et Dates en colonnes, en utilisant le nombre de jours sélectionné
                    display_vol = rolling_vol.sort_index(ascending=False).iloc[:days_to_show].T
                    
                    # Tri du plus volatile au moins volatile (basé sur la date la plus récente)
                    if not display_vol.empty and len(display_vol.columns) > 0:
                        display_vol = display_vol.sort_values(by=display_vol.columns[0], ascending=False)
                    
                    # Calcul évolution hebdo (5 jours de bourse)
                    current = rolling_vol.iloc[-1]
                    prev = rolling_vol.iloc[-6] if len(rolling_vol) >= 6 else current
                    diff = current - prev
                    
                    # --- Ajout de l'évolution du classement ---
                    ranks_df = rolling_vol.rank(axis=1, ascending=False, method='min')
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
                                         .map(lambda x: 'color: #ff4b4b' if x > 0 else 'color: #2ecc71', subset=["Evol. Hebdo (pts)"]),
                        width='stretch',
                        height=400
                    )
            else:
                st.info("Il faut au moins 2 actifs avec historique pour afficher la corrélation.")
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
                    
                    if pd.notnull(h_macd):
                        if h_macd > 0: h_score += 0.5
                        else: h_score -= 0.5
                    
                    if pd.notnull(h_rsi):
                        if h_rsi < 30:
                            h_score += 1
                            h_reason.append(f"RSI: Survendu ({h_rsi:.0f})")
                        elif h_rsi > 70:
                            h_score -= 1
                            h_reason.append(f"RSI: Surchauffé ({h_rsi:.0f})")
                    
                    # 4. Contexte (Buy the dip / Take profit)
                    if perf < -10.0 and h_score > 0: 
                        h_score += 1
                        h_reason.append(f"Rebond sur chute ({perf:.1f}%)")
                    if perf > 15.0 and h_score < 0:
                        h_score -= 1
                        h_reason.append(f"Prise de profit (+{perf:.1f}%)")

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

    elif app_page == "🤖 Assistant IA":
        st.subheader("🤖 Assistant IA - Discutez avec votre portefeuille")
        st.markdown("Posez des questions sur vos actifs, demandez une analyse globale, ou testez des scénarios.")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Ex: Lequel de mes actifs est le plus risqué en ce moment ?"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            portfolio_ctx = "Portefeuille vide."
            if not df_hold.empty:
                cols = [c for c in ["Nom de l'actif", "Type d'actif", "Valeur Actuelle", "Plus-value Latente"] if c in df_hold.columns]
                portfolio_ctx = df_hold[cols].to_string()
                
            full_prompt = f"Contexte de mon portefeuille :\n{portfolio_ctx}\n\nQuestion de l'utilisateur : {prompt}"
                
            with st.chat_message("assistant"):
                api_key = st.session_state.get('api_key', '')
                stream = stream_llm_response(full_prompt, user_api_key=api_key, history=st.session_state.chat_history[:-1])
                response_text = st.write_stream(stream)
                
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

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