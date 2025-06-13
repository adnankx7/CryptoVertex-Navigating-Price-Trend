import os
import sys
import ccxt
import time
import threading
import pandas as pd
from flask import Flask, render_template, jsonify, send_file, request
import pathlib

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
    # from src.LOGGERS import logging # Uncomment if you set up Flask logging
    print("INFO: Successfully imported PredictPipeline and CustomSequenceData in app.py.")
except ImportError as e:
    print(f"ERROR: Failed to import from src in app.py. Check sys.path and module names: {e}")
    PredictPipeline = None
    CustomSequenceData = None
# --- End Custom Module Import ---

app = Flask(__name__)
exchange = ccxt.binance()

DATA_DIR = PROJECT_ROOT_APP / 'data'
if not DATA_DIR.exists():
    print(f"WARNING: Data directory {DATA_DIR} not found. Historical chart/download may fail.")

symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"]
live_data = {symbol: {'price': None, 'change': None} for symbol in symbols}
live_data_lock = threading.Lock()

# Enhanced static market data
market_data_static = {
    "BTC/USDT": {"market_cap": "$1.38T", "volume": "$42.91B", "full_name": "Bitcoin", "ticker_symbol": "BTC", "description": "Bitcoin is the first decentralized cryptocurrency, created in 2009 by an anonymous entity known as Satoshi Nakamoto. It enables peer-to-peer transactions without intermediaries through blockchain technology.", "logo_filename_stem": "bitcoin-btc", "chart_color": "#f2a900"},
    "ETH/USDT": {"market_cap": "$306.00B", "volume": "$22.74B", "full_name": "Ethereum", "ticker_symbol": "ETH", "description": "Ethereum is a decentralized, open-source blockchain with smart contract functionality. Ether (ETH) is the native cryptocurrency of the platform.", "logo_filename_stem": "ethereum-eth", "chart_color": "#627eea"},
    "ADA/USDT": {"market_cap": "$88.60B", "volume": "$1.12B", "full_name": "Cardano", "ticker_symbol": "ADA", "description": "Cardano is a proof-of-stake blockchain platform that says its goal is to allow “changemakers, innovators and visionaries” to bring about positive global change.", "logo_filename_stem": "cardano-ada", "chart_color": "#0033ad"},
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
        time.sleep(5) # Fetch every 5 seconds

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/market')
def market():
    # Pass the static data to the market template
    return render_template('market.html', market_data_for_template=market_data_static)

@app.route('/<coin_url_slug>') # e.g., /btc, /eth
def coin_page(coin_url_slug):
    coin_symbol_ccxt = symbol_map.get(coin_url_slug.lower())
    if not coin_symbol_ccxt:
        return "Coin not found", 404

    coin_static_info = market_data_static.get(coin_symbol_ccxt, {})
    predicted_price_display = "N/A" # Default prediction display

    if predict_pipeline_instance and CustomSequenceData:
        coin_name_for_pipeline = coin_symbol_ccxt.replace('/', '_') # Format for pipeline: BTC_USDT
        
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
        elif historical_data_path: # Path was found in JSON but file doesn't exist
             predicted_price_display = "Data File Missing"
             print(f"WARNING: Historical data file for prediction NOT FOUND at {historical_data_path} for {coin_name_for_pipeline}.")
        else: # Path not found in JSON
            predicted_price_display = "Data Path Missing"
            print(f"WARNING: Could not get historical data path from data_paths.json for {coin_name_for_pipeline}.")
    elif not PredictPipeline:
        predicted_price_display = "Service Error (Pipeline Import Failed)"
    else: # PredictPipeline imported but instance failed
        predicted_price_display = "Service Error (Pipeline Init Failed)"
        
    # Use the coin_url_slug to determine which HTML template to render.
    # This means you need btc.html, eth.html, ada.html, etc.
    # These templates will all use the same internal structure but are called by their specific name.
    template_name = f'{coin_url_slug.lower()}.html'

    # Construct logo path based on the stem in market_data_static
    logo_filename_stem = coin_static_info.get('logo_filename_stem', coin_url_slug.lower())
    logo_path = f"images/{logo_filename_stem}-logo.png" # e.g., images/bitcoin-btc-logo.png

    return render_template(
        template_name,
        coin_symbol_ccxt=coin_symbol_ccxt, # For live data JS & chart JS (e.g., BTC/USDT)
        coin_ticker_symbol=coin_static_info.get("ticker_symbol", coin_url_slug.upper()), # For display (e.g., BTC)
        coin_full_name=coin_static_info.get("full_name", coin_url_slug.title()),
        market_cap=coin_static_info.get("market_cap", "N/A"),
        volume_24h=coin_static_info.get("volume", "N/A"),
        description=coin_static_info.get("description", "No description available."),
        predicted_price=predicted_price_display,
        logo_path=logo_path,
        chart_color=coin_static_info.get("chart_color", "#4A90E2") # Default color if not specified
    )

@app.route('/live_data')
def get_live_data_endpoint():
    with live_data_lock:
        sorted_data = sorted(live_data.items(), key=lambda x: (x[1]['change'] is not None, x[1]['change']), reverse=True)
        top_gainers = [{'symbol': s, **d} for s, d in sorted_data if d['change'] is not None and d['change'] > 0][:3]
        top_losers = [{'symbol': s, **d} for s, d in reversed(sorted_data) if d['change'] is not None and d['change'] < 0][:3]
        return jsonify({'live_data': live_data, 'top_gainers': top_gainers, 'top_losers': top_losers})

@app.route('/about')
def about():
    return render_template('aboutus.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('Disclaimer.html')

@app.route('/historical_data/<path:symbol_ccxt>') # Expects CCXT symbol like BTC/USDT
def historical_data_endpoint(symbol_ccxt):
    try:
        clean_symbol_for_file = symbol_ccxt.replace('/', '_').upper()
        file_path = DATA_DIR / f"{clean_symbol_for_file}.csv"

        if not file_path.exists():
            return jsonify({"error": f"File {file_path.name} not found at {file_path}"}), 404

        df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df = df.sort_values('Date').tail(30) # Get last 30 days for the chart
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return jsonify(df[['Date', 'Close']].to_dict(orient='records'))
    except Exception as e:
        print(f"Error in /historical_data/{symbol_ccxt}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_historical/<path:symbol_ccxt>') # Expects CCXT symbol
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

@app.route('/search_coins')
def search_coins_endpoint():
    query = request.args.get('q', '').lower()
    if not query: return jsonify([])
    results = []
    for symbol, info in market_data_static.items(): # Iterate over market_data_static
        if query in symbol.lower() or query in info['full_name'].lower():
            results.append({
                'symbol': symbol, 
                'full_name': info['full_name'],
                # Use ticker_symbol from market_data_static for the URL slug
                'url': f"/{info.get('ticker_symbol', symbol.split('/')[0]).lower()}" 
            })
    return jsonify(results)

if __name__ == '__main__':
    print(f"INFO: Starting Flask app. Project Root: {PROJECT_ROOT_APP}")
    print(f"INFO: Data directory for historical charts/downloads: {DATA_DIR}")
    if PredictPipeline and predict_pipeline_instance:
        print("INFO: Prediction pipeline is active.")
    else:
        print("WARNING: Prediction pipeline is NOT active. Predictions will be unavailable.")
    
    start_live_data_fetching()
    app.run(debug=True, port=5000)
