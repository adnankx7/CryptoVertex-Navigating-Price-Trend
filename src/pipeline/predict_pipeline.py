import os
import sys
import json
import numpy as np
import pandas as pd
import pathlib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# --- Determine PROJECT_ROOT and add to sys.path ---
try:
    SCRIPT_DIR_PIPELINE = pathlib.Path(__file__).resolve().parent
    SRC_ROOT_PIPELINE = SCRIPT_DIR_PIPELINE.parent
    PROJECT_ROOT_PIPELINE = SRC_ROOT_PIPELINE.parent
except NameError:
    PROJECT_ROOT_PIPELINE = pathlib.Path.cwd()

if str(PROJECT_ROOT_PIPELINE) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_PIPELINE))
# --- End of PROJECT_ROOT setup ---

try:
    from src.logger import logging
    from src.exception import CustomException
    from src.utils import load_pickle_object
except ImportError as e:
    sys.exit(1)

class PredictionPipelineConfig:
    COINS_TO_PREDICT: list[str] = ["ADA_USDT", "BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"]
    BASE_ARTIFACTS_DIR_NAME: str = "artifacts"
    DATA_PATHS_JSON_FILENAME: str = "data_paths.json"
    N_STEPS: int = 30
    FEATURE_COLS: list[str] = ['RSI', 'EMA', 'SMA', 'Target']
    TARGET_COLUMN_NAME: str = 'Target'

class CustomSequenceData:
    def __init__(self, historical_data_csv_path: str, coin_name: str):
        self.historical_data_csv_path = historical_data_csv_path
        self.coin_name = coin_name

class PredictPipeline:
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.project_root = PROJECT_ROOT_PIPELINE
        self.base_artifacts_dir = self.project_root / self.config.BASE_ARTIFACTS_DIR_NAME
        self.data_paths_json_file = self.project_root / self.config.DATA_PATHS_JSON_FILENAME
        self._load_data_paths_config()

    def _load_data_paths_config(self):
        self.all_data_paths_config = None
        try:
            with open(self.data_paths_json_file, 'r') as f:
                self.all_data_paths_config = json.load(f)
            logging.info(f"Data paths config loaded from: {self.data_paths_json_file}")
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            pass
        except Exception:
            pass

    def _get_model_artifacts_paths(self, coin_name: str) -> tuple[pathlib.Path | None, pathlib.Path | None]:
        model_filename = f"{coin_name}.h5"
        scaler_filename = f"{coin_name}_scaler.pkl"
        
        model_path = self.base_artifacts_dir / coin_name / "models" / model_filename
        scaler_path = self.base_artifacts_dir / coin_name / "scalers" / scaler_filename

        if not model_path.exists():
            model_path = None
        if not scaler_path.exists():
            scaler_path = None
            
        return model_path, scaler_path

    def get_historical_data_path_for_coin(self, coin_name: str) -> str | None:
        if not self.all_data_paths_config:
            return None

        if coin_name in self.all_data_paths_config and 'test' in self.all_data_paths_config[coin_name]:
            path_from_json = self.all_data_paths_config[coin_name]['test']
            if pathlib.Path(path_from_json).is_absolute():
                return str(path_from_json)
            else:
                return str(self.project_root / path_from_json)
        else:
            return None

    def predict(self, custom_sequence_data: CustomSequenceData) -> float | None:
        coin_name = custom_sequence_data.coin_name
        historical_data_csv_path = custom_sequence_data.historical_data_csv_path
        
        model_path, scaler_path = self._get_model_artifacts_paths(coin_name)

        if not model_path or not scaler_path:
            return None
        
        if not pathlib.Path(historical_data_csv_path).exists():
            return None

        try:
            logging.info(f"Starting prediction for {coin_name}...")
            model = load_model(str(model_path), custom_objects={'mse': 'mse', 'mae': 'mae'})
            scaler = load_pickle_object(str(scaler_path))

            if not isinstance(scaler, MinMaxScaler):
                raise CustomException(f"Object from {scaler_path} is not a valid MinMaxScaler.", sys)

            hist_df = pd.read_csv(historical_data_csv_path)
            
            missing_cols = [col for col in self.config.FEATURE_COLS if col not in hist_df.columns]
            if missing_cols:
                raise CustomException(f"Data for {coin_name} missing columns: {missing_cols}", sys)

            if self.config.TARGET_COLUMN_NAME not in self.config.FEATURE_COLS:
                raise CustomException(f"Config error: Target column '{self.config.TARGET_COLUMN_NAME}' not in FEATURE_COLS.", sys)
            
            target_idx = self.config.FEATURE_COLS.index(self.config.TARGET_COLUMN_NAME)

            if len(hist_df) < self.config.N_STEPS:
                raise CustomException(f"Data for {coin_name} has {len(hist_df)} rows; {self.config.N_STEPS} needed.", sys)
            
            last_sequence_df = hist_df[self.config.FEATURE_COLS].tail(self.config.N_STEPS)
            scaled_sequence = scaler.transform(last_sequence_df)
            X_input = np.reshape(scaled_sequence, (1, self.config.N_STEPS, len(self.config.FEATURE_COLS)))
            
            scaled_prediction = model.predict(X_input, verbose=0)[0, 0]
            
            dummy_for_inverse = np.zeros((1, len(self.config.FEATURE_COLS)))
            dummy_for_inverse[0, target_idx] = scaled_prediction
            unscaled_pred_array = scaler.inverse_transform(dummy_for_inverse)
            final_prediction = unscaled_pred_array[0, target_idx]

            logging.info(f"Predicted next price for {coin_name}: {final_prediction:.4f}")
            return final_prediction

        except CustomException:
            return None
        except Exception:
            return None

    def run_for_all_configured_coins(self) -> dict:
        logging.info("Starting batch prediction for all configured coins.")
        results = {}
        for coin in self.config.COINS_TO_PREDICT:
            hist_data_path = self.get_historical_data_path_for_coin(coin)
            
            if hist_data_path:
                custom_data = CustomSequenceData(historical_data_csv_path=hist_data_path, coin_name=coin)
                prediction = self.predict(custom_data)
                if prediction is not None:
                    results[coin] = f"{prediction:.4f}"
                else:
                    results[coin] = "Failed"
            else:
                results[coin] = "Skipped (No data path)"
        
        logging.info(f"Batch prediction summary: {results}")
        return results

if __name__ == "__main__":
    logging.info("predict_pipeline.py executed as main script.")
    
    try:
        pipeline = PredictPipeline()
        all_predictions = pipeline.run_for_all_configured_coins()

        btc_hist_path = pipeline.get_historical_data_path_for_coin("BTC_USDT")
        if btc_hist_path:
            btc_custom_data = CustomSequenceData(historical_data_csv_path=btc_hist_path, coin_name="BTC_USDT")
            btc_prediction = pipeline.predict(btc_custom_data)

    except Exception:
        pass
