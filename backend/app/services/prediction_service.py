import sys
import pandas as pd
from pathlib import Path
from ..core.config import settings

# Add backend directory to sys.path to allow src imports
if str(settings.BASE_DIR) not in sys.path:
    sys.path.insert(0, str(settings.BASE_DIR))

# Import the existing pipeline code
PredictPipeline = None
CustomSequenceData = None

try:
    from src.pipeline.predict_pipeline import PredictPipeline, CustomSequenceData
except ImportError as e:
    print(f"Warning: Could not import prediction pipeline: {e}")
    print(f"Predictions will be unavailable. Install ML dependencies: pip install tensorflow keras scikit-learn")

class PredictionService:
    def __init__(self):
        self.pipeline = None
        if PredictPipeline:
            try:
                self.pipeline = PredictPipeline()
                print("Prediction pipeline loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not initialize prediction pipeline: {e}")

    def get_prediction(self, symbol_ccxt: str) -> float | None:
        if not self.pipeline or not CustomSequenceData:
            return None
        
        coin_name_pipeline = symbol_ccxt.replace('/', '_')
        hist_path = self.pipeline.get_historical_data_path_for_coin(coin_name_pipeline)
        
        if hist_path and Path(hist_path).exists():
            custom_data = CustomSequenceData(
                historical_data_csv_path=hist_path,
                coin_name=coin_name_pipeline
            )
            return self.pipeline.predict(custom_data)
        return None

    def get_historical_data(self, symbol_ccxt: str):
        clean_symbol = symbol_ccxt.replace('/', '_').upper()
        file_path = settings.DATA_DIR / f"{clean_symbol}.csv"
        
        if not file_path.exists():
            return None
            
        try:
            df = pd.read_csv(file_path, header=None, names=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df = df.sort_values('Date').tail(30)
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            return df[['Date', 'Close']].to_dict(orient='records')
        except Exception:
            return None

prediction_service = PredictionService()
