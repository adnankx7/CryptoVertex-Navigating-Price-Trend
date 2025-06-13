import sched
import time
import ccxt
import csv
import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.exception import CustomException
from src.logger import logging

# Define the scheduler
scheduler = sched.scheduler(time.time, time.sleep)

# Ensure the 'data' directory exists
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
os.makedirs(DATA_DIR, exist_ok=True)

def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    while num_retries < max_retries:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            logging.debug(f"Successfully fetched OHLCV data for {symbol}.")
            return ohlcv
        except Exception as e:
            num_retries += 1
            logging.error(f"Retry {num_retries}/{max_retries} failed for {symbol} - {e}")
            time.sleep(1)
    raise CustomException(f"Failed to fetch OHLCV for {symbol} after {max_retries} attempts", sys)

def scrape_ohlcv(filename, exchange, max_retries, symbol, timeframe, since, limit):
    timeframe_duration_in_ms = exchange.parse_timeframe(timeframe) * 1000
    now = exchange.milliseconds()
    all_ohlcv = []
    fetch_since = since

    while fetch_since < now:
        try:
            ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
            if not ohlcv:
                break
            write_to_csv(filename, ohlcv)
            fetch_since = ohlcv[-1][0] + timeframe_duration_in_ms
            all_ohlcv.extend(ohlcv)
            logging.debug(
                f"Fetched {len(all_ohlcv)} candles for {symbol} "
                f"from {exchange.iso8601(all_ohlcv[0][0])} to {exchange.iso8601(all_ohlcv[-1][0])}"
            )
        except CustomException as ce:
            logging.error(f"CustomException: {ce}")
            raise ce
        except Exception as e:
            logging.error(f"Unexpected error while scraping {symbol}: {e}")
            raise CustomException(f"Unexpected error while scraping {symbol}: {e}", sys)

    return all_ohlcv

def write_to_csv(filename, data):
    file_path = os.path.join(DATA_DIR, filename)

    # Write new data
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def get_last_timestamp_from_csv(filename):
    try:
        file_path = os.path.join(DATA_DIR, filename)
        with open(file_path, 'r') as f:
            reader = list(csv.reader(f))
            if not reader:
                return None
            last_row = reader[-1]
            while not last_row or len(last_row) == 0:
                reader.pop()
                if not reader:
                    return None
                last_row = reader[-1]
            return int(last_row[0])
    except (FileNotFoundError, IndexError, ValueError):
        logging.warning(f"Could not read timestamp from {filename}, returning None")
        return None

def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol, timeframe, since, limit):
    try:
        exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
            'timeout': 30000,
        })

        retries = 3
        for attempt in range(retries):
            try:
                exchange.load_markets()
                break
            except Exception as e:
                if attempt < retries - 1:
                    logging.warning(f"Retry {attempt+1}/{retries} loading markets for {exchange_id}: {e}")
                    time.sleep(2)
                else:
                    raise CustomException(f"Failed to load markets for {exchange_id}: {e}", sys)

        if isinstance(since, str):
            since = exchange.parse8601(since)

        ohlcv = scrape_ohlcv(filename, exchange, max_retries, symbol, timeframe, since, limit)

        if ohlcv:
            logging.debug(
                f"Saved {len(ohlcv)} candles for {symbol} "
                f"from {exchange.iso8601(ohlcv[0][0])} to {exchange.iso8601(ohlcv[-1][0])} to {filename}"
            )
        else:
            logging.warning(f"No OHLCV data fetched for {symbol}. Nothing saved.")

    except CustomException as ce:
        logging.error(f"CustomException while scraping {symbol}: {ce}")
        raise ce
    except Exception as e:
        logging.error(f"Error scraping candles for {symbol}: {e}")
        raise CustomException(f"Error scraping candles for {symbol}: {e}", sys)

def run_scrape():
    start_time = datetime.now()
    symbols = ["BTC/USDT", "ADA/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]

    for symbol in symbols:
        filename = f"{symbol.replace('/', '_')}.csv"
        last_timestamp = get_last_timestamp_from_csv(filename)
        since = last_timestamp + 1 if last_timestamp else "1970-01-01T00:00:00Z"

        try:
            scrape_candles_to_csv(filename, "binance", 3, symbol, "1d", since, 100)
        except CustomException as e:
            logging.error(f"Error scraping data for {symbol}: {e}")

    end_time = datetime.now()
    logging.info(f"Start time: {start_time}")
    logging.info(f"End time: {end_time}")
    logging.info(f"Total Running time: {end_time - start_time}")

def scheduled_scrape():
    run_scrape()
    scheduler.enter(86400, 1, scheduled_scrape)

# Start the scheduler
scheduler.enter(0, 1, scheduled_scrape)
scheduler.run()
