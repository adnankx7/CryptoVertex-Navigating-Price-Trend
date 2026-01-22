import threading
import time
import requests
import ccxt
from ..core.config import settings

class MarketService:
    def __init__(self):
        self.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"]
        self.coingecko_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "ADA": "cardano",
            "SOL": "solana",
            "XRP": "ripple"
        }
        self.symbol_map = {
            'btc': 'BTC/USDT',
            'eth': 'ETH/USDT',
            'ada': 'ADA/USDT',
            'sol': 'SOL/USDT',
            'xrp': 'XRP/USDT'
        }
        self.live_data = {symbol: {'price': None, 'change': None} for symbol in self.symbols}
        self.live_data_lock = threading.Lock()
        
        self.market_data_static = {
             "BTC/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Bitcoin", "ticker_symbol": "BTC", "logo_filename_stem": "bitcoin-btc", "chart_color": "#f2a900", "description": "Bitcoin is the first decentralized cryptocurrency..."},
             "ETH/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Ethereum", "ticker_symbol": "ETH", "logo_filename_stem": "ethereum-eth", "chart_color": "#627eea", "description": "Ethereum is a decentralized, open-source blockchain..."},
             "ADA/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Cardano", "ticker_symbol": "ADA", "logo_filename_stem": "cardano-ada", "chart_color": "#0033ad", "description": "Cardano is a proof-of-stake blockchain platform..."},
             "SOL/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "Solana", "ticker_symbol": "SOL", "logo_filename_stem": "solana-sol", "chart_color": "#00ffa3", "description": "Solana is a highly functional open source project..."},
             "XRP/USDT": {"market_cap": "Loading...", "volume": "Loading...", "full_name": "XRP", "ticker_symbol": "XRP", "logo_filename_stem": "xrp-xrp", "chart_color": "#346aa9", "description": "XRP is the native cryptocurrency for products developed by Ripple Labs..."}
        }
        
        self.exchange = ccxt.binance()
        
    def start_background_tasks(self):
        t1 = threading.Thread(target=self._coingecko_updater, daemon=True)
        t2 = threading.Thread(target=self._live_data_fetcher, daemon=True)
        t1.start()
        t2.start()

    def _fetch_coingecko_data(self):
        for symbol in self.symbols:
            base_currency = symbol.split('/')[0]
            coin_id = self.coingecko_ids.get(base_currency)
            if not coin_id: continue

            url = f'https://api.coingecko.com/api/v3/coins/{coin_id}'
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    market_data = data.get('market_data', {})
                    market_cap = market_data.get('market_cap', {}).get('usd')
                    volume_24h = market_data.get('total_volume', {}).get('usd')
                    
                    if symbol in self.market_data_static:
                        self.market_data_static[symbol]['market_cap'] = f"${market_cap:,.0f}" if market_cap else "N/A"
                        self.market_data_static[symbol]['volume'] = f"${volume_24h:,.0f}" if volume_24h else "N/A"
            except Exception:
                pass

    def _coingecko_updater(self):
        while True:
            self._fetch_coingecko_data()
            time.sleep(300)

    def _live_data_fetcher(self):
        while True:
            with self.live_data_lock:
                for symbol in self.symbols:
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        self.live_data[symbol]['price'] = ticker['last']
                        self.live_data[symbol]['change'] = ticker['percentage']
                    except Exception:
                        pass
            time.sleep(5)

    def get_market_summary(self):
        with self.live_data_lock:
             # Filter valid data
            valid_data = [
                (symbol, data) for symbol, data in self.live_data.items()
                if data['change'] is not None
            ]
            
            if not valid_data:
                return {
                    'live_data': self.live_data,
                    'top_gainers': [],
                    'top_losers': []
                }

            sorted_desc = sorted(valid_data, key=lambda x: x[1]['change'], reverse=True)
            top_gainers = [{'symbol': s, **d} for s, d in sorted_desc[:3]]
            
            sorted_asc = sorted(valid_data, key=lambda x: x[1]['change'])
            top_losers = [{'symbol': s, **d} for s, d in sorted_asc[:3]]

            return {
                'live_data': self.live_data,
                'top_gainers': top_gainers,
                'top_losers': top_losers
            }

    def get_coin_static_data(self, symbol: str):
        return self.market_data_static.get(symbol)

market_service = MarketService()
