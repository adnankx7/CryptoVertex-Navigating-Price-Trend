import os
import sys
import ccxt
import time
import threading
import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request, session, flash, redirect, url_for
import pathlib
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet, InvalidToken
from io import StringIO  # Added for StringIO

# --- Add Project Root to sys.path ---
try:
    PROJECT_ROOT_APP = pathlib.Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT_APP = pathlib.Path.cwd()

if str(PROJECT_ROOT_APP) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_APP))
    print(f"INFO: Added {PROJECT_ROOT_APP} to sys.path in app.py")
# --- End Project Root Setup ---

# --- Import your custom modules ---
try:
    from src.pipeline.predict_pipeline import PredictPipeline, CustomSequenceData
    print("INFO: Successfully imported PredictPipeline and CustomSequenceData in app.py.")
except ImportError as e:
    print(f"ERROR: Failed to import from src in app.py. Check sys.path and module names: {e}")
    PredictPipeline = None
    CustomSequenceData = None
# --- End Custom Module Import ---

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management
exchange = ccxt.binance()

DATA_DIR = PROJECT_ROOT_APP / 'data'
if not DATA_DIR.exists():
    print(f"WARNING: Data directory {DATA_DIR} not found. Historical chart/download may fail.")

# Create secure_data directory for encrypted user storage
SECURE_DATA_DIR = PROJECT_ROOT_APP / 'secure_data'
if not SECURE_DATA_DIR.exists():
    SECURE_DATA_DIR.mkdir()
    print(f"INFO: Created secure_data directory at {SECURE_DATA_DIR}")

# Validate and initialize encryption system
def initialize_encryption_system():
    KEY_FILE = SECURE_DATA_DIR / "secret.key"
    USER_FILE = SECURE_DATA_DIR / "users.enc"
    
    # Generate or load encryption key with validation
    if not KEY_FILE.exists():
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as key_file:
            key_file.write(key)
        print(f"INFO: Generated new Fernet key: {key.decode()}")
        return Fernet(key)
    else:
        with open(KEY_FILE, "rb") as key_file:
            key = key_file.read()
        
        # Validate key is not empty
        if len(key) == 0:
            print("WARNING: Empty key file found. Generating new key.")
            key = Fernet.generate_key()
            with open(KEY_FILE, "wb") as key_file:
                key_file.write(key)
            print(f"INFO: New key generated: {key.decode()}")
            return Fernet(key)
        else:
            try:
                # Try to initialize Fernet to validate key
                cipher = Fernet(key)
                print("INFO: Valid Fernet key loaded.")
                return cipher
            except ValueError as e:
                print(f"WARNING: Invalid key found: {str(e)}")
                print("INFO: Generating new Fernet key...")
                key = Fernet.generate_key()
                with open(KEY_FILE, "wb") as key_file:
                    key_file.write(key)
                print(f"INFO: New key generated: {key.decode()}")
                return Fernet(key)
    
    # Validate users.enc file
    if USER_FILE.exists() and USER_FILE.stat().st_size == 0:
        print("WARNING: Empty users.enc file found. Removing...")
        USER_FILE.unlink()

# Initialize encryption system
cipher_suite = initialize_encryption_system()

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(encrypted_data):
    return cipher_suite.decrypt(encrypted_data).decode()

def save_user_data(df):
    # Only save if we have data
    if not df.empty:
        csv_data = df.to_csv(index=False)
        encrypted_data = encrypt_data(csv_data)
        with open(SECURE_DATA_DIR / "users.enc", "wb") as f:
            f.write(encrypted_data)
        print(f"Saved user data with {len(df)} records")
    else:
        print("No user data to save")

def load_user_data():
    file_path = SECURE_DATA_DIR / "users.enc"
    
    # If file doesn't exist, return empty DataFrame
    if not file_path.exists():
        print("No users.enc file found")
        return pd.DataFrame(columns=['username', 'email', 'password'])
    
    # Check if file is empty
    if file_path.stat().st_size == 0:
        print("Empty users.enc file found")
        return pd.DataFrame(columns=['username', 'email', 'password'])
    
    try:
        with open(file_path, "rb") as f:
            encrypted_data = f.read()
        
        # If no data, return empty DataFrame
        if len(encrypted_data) == 0:
            print("Empty encrypted data read from file")
            return pd.DataFrame(columns=['username', 'email', 'password'])
        
        decrypted_data = decrypt_data(encrypted_data)
        
        # If decrypted data is empty, return empty DataFrame
        if not decrypted_data.strip():
            print("Decrypted data is empty")
            return pd.DataFrame(columns=['username', 'email', 'password'])
            
        # FIX: Use StringIO from io module
        return pd.read_csv(StringIO(decrypted_data))
    except InvalidToken:
        print("ERROR: Invalid token when decrypting user data")
        return pd.DataFrame(columns=['username', 'email', 'password'])
    except Exception as e:
        print(f"ERROR loading user data: {str(e)}")
        return pd.DataFrame(columns=['username', 'email', 'password'])

symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"]
live_data = {symbol: {'price': None, 'change': None} for symbol in symbols}
live_data_lock = threading.Lock()

# Enhanced static market data
market_data_static = {
    "BTC/USDT": {"market_cap": "$1.38T", "volume": "$42.91B", "full_name": "Bitcoin", "ticker_symbol": "BTC", "description": "Bitcoin is the first decentralized cryptocurrency, created in 2009 by an anonymous entity known as Satoshi Nakamoto. It enables peer-to-peer transactions without intermediaries through blockchain technology.", "logo_filename_stem": "bitcoin-btc", "chart_color": "#f2a900"},
    "ETH/USDT": {"market_cap": "$306.00B", "volume": "$22.74B", "full_name": "Ethereum", "ticker_symbol": "ETH", "description": "Ethereum is a decentralized, open-source blockchain with smart contract functionality. Ether (ETH) is the native cryptocurrency of the platform.", "logo_filename_stem": "ethereum-eth", "chart_color": "#627eea"},
    "ADA/USDT": {"market_cap": "$88.60B", "volume": "$1.12B", "full_name": "Cardano", "ticker_symbol": "ADA", "description": "Cardano is a proof-of-stake blockchain platform that says its goal is to allow \"changemakers, innovators and visionaries\" to bring about positive global change.", "logo_filename_stem": "cardano-ada", "chart_color": "#0033ad"},
    "SOL/USDT": {"market_cap": "$83.31B", "volume": "$7.57B", "full_name": "Solana", "ticker_symbol": "SOL", "description": "Solana is a highly functional open source project that implements a new, high-performance, permissionless blockchain.", "logo_filename_stem": "solana-sol", "chart_color": "#00ffa3"},
    "XRP/USDT": {"market_cap": "$42.91B", "volume": "$1.83B", "full_name": "XRP", "ticker_symbol": "XRP", "description": "XRP is the native cryptocurrency for products developed by Ripple Labs. Its products are used for payment settlement, asset exchange, and remittance systems.", "logo_filename_stem": "xrp-xrp", "chart_color": "#346aa9"}
}

# Maps URL part (e.g. 'btc') to CCXT symbol (e.g. 'BTC/USDT')
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
        print("INFO: PredictPipeline initialized successfully in app.py.")
    except Exception as e:
        print(f"ERROR: Failed to initialize PredictPipeline in app.py: {e}")

def fetch_live_data():
    while True:
        with live_data_lock:
            for symbol in symbols:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    live_data[symbol]['price'] = ticker['last']
                    live_data[symbol]['change'] = ticker['percentage']
                except Exception as e:
                    print(f"Error fetching live data for {symbol}: {e}")
        time.sleep(5)  # Fetch every 5 seconds

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/market')
def market():
    return render_template('market.html', market_data_for_template=market_data_static)

@app.route('/<coin_url_slug>')  # e.g., /btc, /eth
def coin_page(coin_url_slug):
    coin_symbol_ccxt = symbol_map.get(coin_url_slug.lower())
    if not coin_symbol_ccxt:
        return "Coin not found", 404

    coin_static_info = market_data_static.get(coin_symbol_ccxt, {})
    predicted_price_display = "N/A"  # Default prediction display

    if predict_pipeline_instance and CustomSequenceData:
        coin_name_for_pipeline = coin_symbol_ccxt.replace('/', '_')  # Format for pipeline: BTC_USDT
        
        historical_data_path = predict_pipeline_instance.get_historical_data_path_for_coin(coin_name_for_pipeline)

        if historical_data_path and pathlib.Path(historical_data_path).exists():
            print(f"INFO: Using historical data for {coin_name_for_pipeline} prediction from: {historical_data_path}")
            custom_data = CustomSequenceData(
                historical_data_csv_path=historical_data_path,
                coin_name=coin_name_for_pipeline
            )
            prediction_result = predict_pipeline_instance.predict(custom_data)

            if prediction_result is not None:
                predicted_price_display = f"${prediction_result:.2f}"
            else:
                predicted_price_display = "Unavailable"
                print(f"WARNING: Prediction returned None for {coin_name_for_pipeline}.")
        elif historical_data_path:  # Path was found in JSON but file doesn't exist
            predicted_price_display = "Data File Missing"
            print(f"WARNING: Historical data file for prediction NOT FOUND at {historical_data_path} for {coin_name_for_pipeline}.")
        else:  # Path not found in JSON
            predicted_price_display = "Data Path Missing"
            print(f"WARNING: Could not get historical data path from data_paths.json for {coin_name_for_pipeline}.")
    elif not PredictPipeline:
        predicted_price_display = "Service Error (Pipeline Import Failed)"
    else:  # PredictPipeline imported but instance failed
        predicted_price_display = "Service Error (Pipeline Init Failed)"
        
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
        sorted_data = sorted(live_data.items(), key=lambda x: (x[1]['change'] is not None, x[1]['change']), reverse=True)
        top_gainers = [{'symbol': s, **d} for s, d in sorted_data if d['change'] is not None and d['change'] > 0][:3]
        top_losers = [{'symbol': s, **d} for s, d in reversed(sorted_data) if d['change'] is not None and d['change'] < 0][:3]
        return jsonify({'live_data': live_data, 'top_gainers': top_gainers, 'top_losers': top_losers})

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        users_df = load_user_data()
        
        # If the DataFrame is empty, no users exist
        if users_df.empty:
            flash('No users registered yet', 'error')
            return redirect(url_for('login'))
        
        # Find user by email
        user = users_df[users_df['email'] == email]
        
        if user.empty:
            flash('Invalid email or password', 'error')
            return redirect(url_for('login'))
        
        # Verify password
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
        
        # If there was an error loading user data, handle it
        if users_df is None:
            flash('System error. Please try again later.', 'error')
            return redirect(url_for('signup'))
        
        # Check if email already exists
        if not users_df.empty and email in users_df['email'].values:
            flash('Email already registered', 'error')
            return redirect(url_for('signup'))
        
        # Hash the password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        
        # Create new user entry
        new_user = pd.DataFrame({
            'username': [username],
            'email': [email],
            'password': [hashed_password]
        })
        
        # Append new user to existing data
        if users_df.empty:
            users_df = new_user
        else:
            users_df = pd.concat([users_df, new_user], ignore_index=True)
        
        # Save updated user data
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

@app.route('/historical_data/<path:symbol_ccxt>')  # Expects CCXT symbol like BTC/USDT
def historical_data_endpoint(symbol_ccxt):
    try:
        clean_symbol_for_file = symbol_ccxt.replace('/', '_').upper()
        file_path = DATA_DIR / f"{clean_symbol_for_file}.csv"

        if not file_path.exists():
            return jsonify({"error": f"File {file_path.name} not found at {file_path}"}), 404

        df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.sort_values('Date').tail(30)  # Get last 30 days for the chart
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return jsonify(df[['Date', 'Close']].to_dict(orient='records'))
    except Exception as e:
        print(f"Error in /historical_data/{symbol_ccxt}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_historical/<path:symbol_ccxt>')  # Expects CCXT symbol
def download_historical_endpoint(symbol_ccxt):
    try:
        clean_symbol_for_file = symbol_ccxt.replace('/', '_').upper()
        file_path = DATA_DIR / f"{clean_symbol_for_file}.csv"
        
        if not file_path.exists():
            return jsonify({"error": f"File {file_path.name} not found at {file_path}"}), 404
        
        return send_file(str(file_path), mimetype='text/csv', as_attachment=True,
                         download_name=f"{clean_symbol_for_file}_historical_data.csv")
    except Exception as e:
        print(f"Error in /download_historical/{symbol_ccxt}: {str(e)}")
        return jsonify({"error": str(e)}), 500

def start_live_data_fetching():
    thread = threading.Thread(target=fetch_live_data, daemon=True)
    thread.start()

@app.route('/api/search_coins')
def search_coins():
    """Provides a list of coins for the search bar."""
    search_data = []
    for slug, ccxt_symbol in symbol_map.items():
        if ccxt_symbol in market_data_static:
            coin_info = market_data_static[ccxt_symbol]
            search_data.append({
                'name': coin_info.get('full_name'),
                'ticker': coin_info.get('ticker_symbol'),
                'slug': slug  # e.g., 'btc', 'eth'
            })
    return jsonify(search_data)

if __name__ == '__main__':
    print(f"INFO: Starting Flask app. Project Root: {PROJECT_ROOT_APP}")
    print(f"INFO: Data directory for historical charts/downloads: {DATA_DIR}")
    print(f"INFO: Secure data directory for encrypted storage: {SECURE_DATA_DIR}")
    
    # Clean up any empty files
    key_file = SECURE_DATA_DIR / "secret.key"
    user_file = SECURE_DATA_DIR / "users.enc"
    
    if key_file.exists() and key_file.stat().st_size == 0:
        print("WARNING: Removing empty secret.key file")
        key_file.unlink()
    
    if user_file.exists() and user_file.stat().st_size == 0:
        print("WARNING: Removing empty users.enc file")
        user_file.unlink()
    
    # Reinitialize encryption system
    cipher_suite = initialize_encryption_system()
    
    if PredictPipeline and predict_pipeline_instance:
        print("INFO: Prediction pipeline is active.")
    else:
        print("WARNING: Prediction pipeline is NOT active. Predictions will be unavailable.")
    
    start_live_data_fetching()
    app.run(debug=True, port=5000)