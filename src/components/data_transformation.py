import os
import sys
import pandas as pd
from dataclasses import dataclass, field

# Ensure that the parent directory (Crypto Vertex) is in sys.path for correct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation process.
    """
    raw_data_folder: str = os.path.join(os.path.dirname(__file__), '../../data')
    output_folder: str = os.path.join(os.path.dirname(__file__), '../../artifacts')
    coin_list: list = field(default_factory=lambda: ['ETH_USDT', 'BTC_USDT', 'XRP_USDT', 'SOL_USDT', 'ADA_USDT'])
    column_names: list = field(default_factory=lambda: ['Timestamp', 'Open', 'High', 'Low', 'Price', 'Volume'])

class DataTransformation:
    """
    Class to handle data transformation including reading raw data,
    applying technical indicators, splitting datasets, and saving
    the processed data into artifacts folder.
    """
    def __init__(self):
        """
        Initialize DataTransformation with configuration and ensure output folder exists.
        """
        self.config = DataTransformationConfig()
        os.makedirs(self.config.output_folder, exist_ok=True)

    def add_features(self, df: pd.DataFrame, period=14, ema_span=5, sma_window=5) -> pd.DataFrame:
        """
        Add technical indicators RSI, EMA, and SMA to the dataframe.

        Args:
            df (pd.DataFrame): Input dataframe with price data.
            period (int): Period for RSI calculation.
            ema_span (int): Span for EMA calculation.
            sma_window (int): Window size for SMA calculation.

        Returns:
            pd.DataFrame: Dataframe with added technical indicator columns.
        """
        df = df.copy()
        delta = df['Price'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()

        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['EMA'] = df['Price'].ewm(span=ema_span, adjust=False).mean()
        df['SMA'] = df['Price'].rolling(window=sma_window, min_periods=1).mean()

        # ✅ Add next day's price as Target
        df['Target'] = df['Price'].shift(-1)

        return df.dropna()

    def transform_and_save(self):
        """
        Perform the data transformation process:
        - Read raw CSV files for each coin.
        - Convert timestamp and sort data.
        - Add technical indicators and target.
        - Split data into train, validation, and test sets.
        - Save the splits into the artifacts folder.
        """
        try:
            logging.info("Starting data transformation")

            for coin in self.config.coin_list:
                file_path = os.path.join(self.config.raw_data_folder, f"{coin}.csv")

                if not os.path.exists(file_path):
                    logging.warning(f"File not found for coin: {coin}")
                    continue

                df = pd.read_csv(file_path, names=self.config.column_names)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df = df.sort_values('Timestamp')

                df = self.add_features(df)

                # Split
                total = len(df)
                train_end = int(0.8 * total)
                val_end = train_end + int(0.1 * total)

                train_df = df.iloc[:train_end]
                val_df = df.iloc[train_end:val_end]
                test_df = df.iloc[val_end:]

                # Create coin-specific folder inside artifacts
                coin_folder = os.path.join(self.config.output_folder, coin)
                os.makedirs(coin_folder, exist_ok=True)

                # Save train, val, test data inside the coin folder
                train_df.to_csv(os.path.join(coin_folder, "train.csv"), index=False)
                val_df.to_csv(os.path.join(coin_folder, "val.csv"), index=False)
                test_df.to_csv(os.path.join(coin_folder, "test.csv"), index=False)

                logging.info(f"Saved train/val/test for {coin}")

            logging.info("Data transformation completed successfully")

        except Exception as e:
            raise CustomException("Error in data transformation", e)

    def run(self):
        """
        External method to run the data transformation process.
        """
        self.transform_and_save()

if __name__ == "__main__":
    dt = DataTransformation()
    dt.run()
