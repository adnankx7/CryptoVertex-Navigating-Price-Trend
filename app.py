import os
import sys
import ccxt
import time
import threading
import pandas as pd
import requests
from flask import Flask, render_template, jsonify, send_file, request, session, flash, redirect, url_for
import pathlib
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet, InvalidToken
from io import StringIO
from main import run_pipeline, run_scheduler  # Import from main.py

# --- Add Project Root to sys.path ---
try:
    PROJECT_ROOT_APP = pathlib.Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT_APP = pathlib.Path.cwd()

if str(PROJECT_ROOT_APP) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_APP))

# --- Import custom modules ---
try:
    from src.pipeline.predict_pipeline import PredictPipeline, CustomSequenceData
except ImportError:
    PredictPipeline = None
    CustomSequenceData = None

app = Flask(__name__)
app.secret_key = os.urandom(24)
exchange = ccxt.binance()

DATA_DIR = PROJECT_ROOT_APP / 'data'
SECURE_DATA_DIR = PROJECT_ROOT_APP / 'secure_data'
if not SECURE_DATA_DIR.exists():
    SECURE_DATA_DIR.mkdir()

# Encryption system
def initialize_encryption_system():
    KEY_FILE = SECURE_DATA_DIR / "secret.key"
    USER_FILE = SECURE_DATA_DIR / "users.enc"
    
    if not KEY_FILE.exists():
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
        return Fernet(key)
    else:
        with open(KEY_FILE, "rb") as key_file:
            key = key_file.read()
        
        if len(key) == 0:
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as key_file:
                key_file.write(key)
            return Fernet(key)
        else:
            try:
                cipher = Fernet(key)
                return cipher
            except ValueError:
                key = Fernet.generate_key()
                with open(KEY_FILE, "wb") as key_file:
                    key_file.write(key)
                return Fernet(key)
    
    if USER_FILE.exists() and USER_FILE.stat().st_size == 0:
        USER_FILE.unlink()

cipher_suite = initialize_encryption_system()

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data).decode()

def save_user_data(df):
    if not df.empty:
        csv_data = df.to_csv(index=False)
        encrypted_data = encrypt_data(csv_data)
        with open(SECURE_DATA_DIR / "users.enc", "wb") as f:
            f.write(encrypted_data)

def load_user_data():
    file_path = SECURE_DATA_DIR / "users.enc"
    
    if not file_path.exists():
        return pd.DataFrame(columns=['username', 'email', 'password'])
    
    if file_path.stat().st_size == 0:
        return pd.DataFrame(columns=['username', 'email', 'password'])
    
    try:
        with open(file_path, "rb") as f:
            encrypted_data = f.read()
        
        if len(encrypted_data) == 0:
            return pd.DataFrame(columns=['username', 'email', 'password'])
        
        decrypted_data = decrypt_data(encrypted_data)
        
        if not decrypted_data.strip():
            return pd.DataFrame(columns=['username', 'email', 'password'])
            
        return pd.read_csv(StringIO(decrypted_data))
    except (InvalidToken, Exception):
        return pd.DataFrame(columns=['username', 'email', 'password'])

symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"]

# CoinGecko ID mapping
coingecko_ids = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "ADA": "cardano",
    "SOL": "solana",
    "XRP": "ripple"
}

live_data = {symbol: {'price': None, 'change': None} for symbol in symbols}
live_data_lock = threading.Lock()

market_data_static = {
    "BTC/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Bitcoin", "ticker_symbol": "BTC", "description": "Bitcoin is the first decentralized cryptocurrency, created in 2009 by an anonymous entity known as Satoshi Nakamoto. It enables peer-to-peer transactions without intermediaries through blockchain technology", "logo_filename_stem": "bitcoin-btc", "chart_color": "#f2a900"},
    "ETH/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Ethereum", "ticker_symbol": "ETH", "description": "Ethereum is a decentralized, open-source blockchain with smart contract functionality. Ether (ETH) is the native cryptocurrency of the platform.", "logo_filename_stem": "ethereum-eth", "chart_color": "#627eea"},
    "ADA/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Cardano", "ticker_symbol": "ADA", "description": "Cardano is a proof-of-stake blockchain platform that says its goal is to allow \"changemakers, innovators and visionaries\" to bring about positive global change.", "logo_filename_stem": "cardano-ada", "chart_color": "#0033ad"},
    "SOL/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Solana", "ticker_symbol": "SOL", "description": "Solana is a highly functional open source project that implements a new, high-performance, permissionless blockchain.", "logo_filename_stem": "solana-sol", "chart_color": "#00ffa3"},
    "XRP/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "XRP", "ticker_symbol": "XRP", "description": "XRP is the native cryptocurrency for products developed by Ripple Labs. Its products are used for payment settlement, asset exchange, and remittance systems.", "logo_filename_stem": "xrp-xrp", "chart_color": "#346aa9"}
}

symbol_map = {
    'btc': 'BTC/USDT',
    'eth': 'ETH/USDT',
    'ada': 'ADA/USDT',
    'sol': 'SOL/USDT',
    'xrp': 'XRP/USDT'
}

predict_pipeline_instance = None
if PredictPipeline:
    try:
        predict_pipeline_instance = PredictPipeline()
    except Exception:
        pass

# CoinGecko data functions
def fetch_coingecko_data():
    for symbol in symbols:
        base_currency = symbol.split('/')[0]
        coin_id = coingecko_ids.get(base_currency)
        
        if not coin_id:
            continue

        url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                market_data = data.get('market_data', {})
                
                market_cap = market_data.get('market_cap', {}).get('usd')
                volume_24h = market_data.get('total_volume', {}).get('usd')
                
                market_cap_str = f"${market_cap:,.0f}" if market_cap else "N/A"
                volume_24h_str = f"${volume_24h:,.0f}" if volume_24h else "N/A"
                
                if symbol in market_data_static:
                    market_data_static[symbol]['market_cap'] = market_cap_str
                    market_data_static[symbol]['volume'] = volume_24h_str
        except Exception:
            pass

def coingecko_updater():
    while True:
        fetch_coingecko_data()
        time.sleep(300)

def fetch_live_data():
    while True:
        with live_data_lock:
            for symbol in symbols:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    live_data[symbol]['price'] = ticker['last']
                    live_data[symbol]['change'] = ticker['percentage']
                except Exception:
                    pass
        time.sleep(5)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/market')
def market():
    return render_template('market.html', market_data_for_template=market_data_static)

@app.route('/<coin_url_slug>')
def coin_page(coin_url_slug):
    coin_symbol_ccxt = symbol_map.get(coin_url_slug.lower())
    if not coin_symbol_ccxt:
        return "Coin not found", 404

    coin_static_info = market_data_static.get(coin_symbol_ccxt, {})
    predicted_price_display = "N/A"

    if predict_pipeline_instance and CustomSequenceData:
        coin_name_for_pipeline = coin_symbol_ccxt.replace('/', '_')
        
        historical_data_path = predict_pipeline_instance.get_historical_data_path_for_coin(coin_name_for_pipeline)

        if historical_data_path and pathlib.Path(historical_data_path).exists():
            custom_data = CustomSequenceData(
                historical_data_csv_path=historical_data_path,
                coin_name=coin_name_for_pipeline
            )
            prediction_result = predict_pipeline_instance.predict(custom_data)

            if prediction_result is not None:
                predicted_price_display = f"${prediction_result:.2f}"
            else:
                predicted_price_display = "Unavailable"
        elif historical_data_path:
            predicted_price_display = "Data File Missing"
        else:
            predicted_price_display = "Data Path Missing"
    elif not PredictPipeline:
        predicted_price_display = "Service Error"
    else:
        predicted_price_display = "Service Error"
        
    template_name = f'{coin_url_slug.lower()}.html'
    logo_filename_stem = coin_static_info.get('logo_filename_stem', coin_url_slug.lower())
    logo_path = f"images/{logo_filename_stem}-logo.png"

    return render_template(
        template_name,
        coin_symbol_ccxt=coin_symbol_ccxt,
        coin_ticker_symbol=coin_static_info.get("ticker_symbol", coin_url_slug.upper()),
        coin_full_name=coin_static_info.get("full_name", coin_url_slug.title()),
        market_cap=coin_static_info.get("market_cap", "N/A"),
        volume_24h=coin_static_info.get("volume", "N/A"),
        description=coin_static_info.get("description", "No description available."),
        predicted_price=predicted_price_display,
        logo_path=logo_path,
        chart_color=coin_static_info.get("chart_color", "#4A90E2")
    )

@app.route('/live_data')
def get_live_data_endpoint():
    with live_data_lock:
        # Filter only valid entries with non-null 'change'
        valid_data = [
            (symbol, data)
            for symbol, data in live_data.items()
            if data['change'] is not None
        ]

        if not valid_data:
            # Graceful fallback
            return jsonify({
                'live_data': live_data,
                'top_gainers': [],
                'top_losers': []
            })

        # Sort descending: highest change first (for gainers)
        sorted_by_change_desc = sorted(valid_data, key=lambda x: x[1]['change'], reverse=True)
        top_gainers = [{'symbol': symbol, **data} for symbol, data in sorted_by_change_desc[:3]]

        # Sort ascending: lowest change first (for losers)
        sorted_by_change_asc = sorted(valid_data, key=lambda x: x[1]['change'])
        top_losers = [{'symbol': symbol, **data} for symbol, data in sorted_by_change_asc[:3]]

        return jsonify({
            'live_data': live_data,
            'top_gainers': top_gainers,
            'top_losers': top_losers
        })


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        users_df = load_user_data()
        
        if users_df.empty:
            flash('No users registered yet', 'error')
            return redirect(url_for('login'))
        
        user = users_df[users_df['email'] == email]
        
        if user.empty:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
        
        if check_password_hash(user.iloc[0]['password'], password):
            session['email'] = email
            session['username'] = user.iloc[0]['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html', title='Log In')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        users_df = load_user_data()
        
        if users_df is None:
            flash('System error. Please try again later.', 'error')
            return redirect(url_for('signup'))
        
        if not users_df.empty and email in users_df['email'].values:
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        
        new_user = pd.DataFrame({
            'username': [username],
            'email': [email],
            'password': [hashed_password]
        })
        
        if users_df.empty:
            users_df = new_user
        else:
            users_df = pd.concat([users_df, new_user], ignore_index=True)
        
        save_user_data(users_df)
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('signup.html', title='Sign Up')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('Disclaimer.html')

@app.route('/historical_data/<path:symbol_ccxt>')
def historical_data_endpoint(symbol_ccxt):
    try:
        clean_symbol_for_file = symbol_ccxt.replace('/', '_').upper()
        file_path = DATA_DIR / f"{clean_symbol_for_file}.csv"

        if not file_path.exists():
            return jsonify({"error": "Data not available"}), 404

        df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.sort_values('Date').tail(30)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return jsonify(df[['Date', 'Close']].to_dict(orient='records'))
    except Exception:
        return jsonify({"error": "Server error"}), 500

@app.route('/download_historical/<path:symbol_ccxt>')
def download_historical_endpoint(symbol_ccxt):
    try:
        clean_symbol_for_file = symbol_ccxt.replace('/', '_').upper()
        file_path = DATA_DIR / f"{clean_symbol_for_file}.csv"
        
        if not file_path.exists():
            return jsonify({"error": "Data not available"}), 404
        
        return send_file(str(file_path), mimetype='text/csv', as_attachment=True,
                         download_name=f"{clean_symbol_for_file}_historical_data.csv")
    except Exception:
        return jsonify({"error": "Server error"}), 500

def start_live_data_fetching():
    thread = threading.Thread(target=fetch_live_data, daemon=True)
    thread.start()

def start_coingecko_updater():
    thread = threading.Thread(target=coingecko_updater, daemon=True)
    thread.start()

@app.route('/api/search_coins')
def search_coins():
    search_data = []
    for slug, ccxt_symbol in symbol_map.items():
        if ccxt_symbol in market_data_static:
            coin_info = market_data_static[ccxt_symbol]
            search_data.append({
                'name': coin_info.get('full_name'),
                'ticker': coin_info.get('ticker_symbol'),
                'slug': slug
            })
    return jsonify(search_data)

if __name__ == '__main__':
    # Clean up any empty files
    key_file = SECURE_DATA_DIR / "secret.key"
    user_file = SECURE_DATA_DIR / "users.enc"
    
    if key_file.exists() and key_file.stat().st_size == 0:
        key_file.unlink()
    
    if user_file.exists() and user_file.stat().st_size == 0:
        user_file.unlink()
    
    cipher_suite = initialize_encryption_system()
    
    # Start data fetching threads
    start_live_data_fetching()
    start_coingecko_updater()
    fetch_coingecko_data()
    
    # Start the scheduler in a background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # Run the pipeline once immediately at startup
    pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
    pipeline_thread.start()
    
    # Start Flask app
    app.run(debug=True, port=5000)