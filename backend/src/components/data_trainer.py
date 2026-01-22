import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch 
import pathlib

# --- MODIFICATION: Define PROJECT_ROOT and adjust sys.path ---
try:
    SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    print("Error: Could not determine project root. Adjust pathlib.Path(__file__).resolve().parents index.")
    sys.exit(1)

sys.path.append(str(PROJECT_ROOT))

from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    import pickle
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def create_sequences(data, n_steps, target_column_index):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps, target_column_index])
    return np.array(X), np.array(y)

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)
    return np.mean(ratio) * 100

@dataclass
class ModelTrainerConfig:
    pass

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.feature_cols = ['RSI', 'EMA', 'SMA', 'Target']
        self.target_column_name = 'Target'
        self.volume_column_name = 'Volume'
        self.target_column_index = self.feature_cols.index(self.target_column_name)
        self.n_steps = 30

    def train_model(self, coin_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, base_artifacts_output_dir: str):
        logging.info(f"Starting model training process for {coin_name}...")
        try:
            coin_dir = os.path.join(base_artifacts_output_dir, coin_name)
            model_dir = os.path.join(coin_dir, 'models')
            plot_dir = os.path.join(coin_dir, 'plots')
            metrics_dir = os.path.join(coin_dir, 'metrics')
            scaler_dir = os.path.join(coin_dir, 'scalers')
            tuner_dir = os.path.join(coin_dir, 'hyperparameter_tuning')

            for d in [model_dir, plot_dir, metrics_dir, scaler_dir, tuner_dir]:
                os.makedirs(d, exist_ok=True)

            model_path = os.path.join(model_dir, f"{coin_name}.h5")
            plot_path = os.path.join(plot_dir, f"{coin_name}.png")
            metrics_excel_path = os.path.join(metrics_dir, f"{coin_name}_metrics.xlsx")
            metrics_image_path = os.path.join(metrics_dir, f"{coin_name}_metrics.png")
            scaler_path = os.path.join(scaler_dir, f"{coin_name}_scaler.pkl")

            for df_name, df_obj in [("train_df", train_df), ("val_df", val_df), ("test_df", test_df)]:
                missing_cols = [col for col in self.feature_cols if col not in df_obj.columns]
                if missing_cols:
                    raise CustomException(f"Missing feature columns {missing_cols} in {df_name} for {coin_name}", sys)

            scaler = MinMaxScaler()
            train_scaled_features = scaler.fit_transform(train_df[self.feature_cols])
            save_object(file_path=scaler_path, obj=scaler)
            logging.info(f"Saved MinMaxScaler for {coin_name} to: {scaler_path}")

            val_scaled_features = scaler.transform(val_df[self.feature_cols])
            test_scaled_features = scaler.transform(test_df[self.feature_cols])

            X_train, y_train = create_sequences(train_scaled_features, self.n_steps, self.target_column_index)
            X_val, y_val = create_sequences(val_scaled_features, self.n_steps, self.target_column_index)
            X_test, y_test = create_sequences(test_scaled_features, self.n_steps, self.target_column_index)

            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                raise CustomException(f"Not enough data to create sequences for {coin_name} with n_steps={self.n_steps}. "
                                      f"Train shape: {train_df.shape}, Val shape: {val_df.shape}", sys)

            num_input_features = X_train.shape[2]

            def build_model_for_tuner(hp):
                model = Sequential()
                num_layers = hp.Int('num_layers', 1, 3)
                units = hp.Int('units', 32, 256, step=32)
                dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
                learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
                for i in range(num_layers):
                    is_last = i == num_layers - 1
                    model.add(GRU(units=units, return_sequences=not is_last,
                                  input_shape=(self.n_steps, num_input_features) if i == 0 else ()))
                    model.add(Dropout(dropout_rate))
                model.add(Dense(1))
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
                return model

            tuner = RandomSearch(build_model_for_tuner, objective='val_loss', max_trials=10,
                                 executions_per_trial=1, directory=tuner_dir, project_name=f'{coin_name}_tuning', overwrite=True)
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            logging.info(f"Starting hyperparameter tuning for {coin_name}...")
            tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=50,
                         batch_size=64, callbacks=[early_stop], verbose=1)
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            logging.info(f"Best hyperparameters for {coin_name}: {best_hp.values}")

            final_model = tuner.hypermodel.build(best_hp)
            logging.info(f"Training final model for {coin_name}...")
            combined_X = np.concatenate((X_train, X_val), axis=0)
            combined_y = np.concatenate((y_train, y_val), axis=0)

            if combined_X.shape[0] == 0:
                raise CustomException(f"Combined training and validation data is empty for {coin_name}. Cannot train final model.", sys)

            final_model.fit(combined_X, combined_y, epochs=75, batch_size=64, verbose=1, callbacks=[early_stop])
            final_model.save(model_path)
            logging.info(f"Saved model for {coin_name} to: {model_path}")

            metrics_dict = {metric: np.nan for metric in ["MSE", "RMSE", "MAE", "R2_Score", "sMAPE (%)",
                                                           "TWAP_Actual", "TWAP_Predicted", "VWAP_Actual", "VWAP_Predicted"]}
            if X_test.shape[0] > 0:
                y_pred_scaled = final_model.predict(X_test)
                dummy_pred = np.zeros((len(y_pred_scaled), num_input_features))
                dummy_pred[:, self.target_column_index] = y_pred_scaled.flatten()
                y_pred = scaler.inverse_transform(dummy_pred)[:, self.target_column_index]

                dummy_true = np.zeros((len(y_test), num_input_features))
                dummy_true[:, self.target_column_index] = y_test.flatten()
                y_true = scaler.inverse_transform(dummy_true)[:, self.target_column_index]

                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

                logging.info(f"Evaluation metrics for {coin_name} on test set:")
                logging.info(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, sMAPE: {smape:.2f}%")

                relevant_test_slice = test_df.iloc[self.n_steps:self.n_steps + len(y_true)]
                vwap_actual = np.nan
                vwap_predicted = np.nan
                if self.volume_column_name in relevant_test_slice.columns:
                    volumes = relevant_test_slice[self.volume_column_name].values
                    if len(volumes) == len(y_true) and volumes.sum() != 0:
                        vwap_actual = (y_true * volumes).sum() / volumes.sum()
                        vwap_predicted = (y_pred * volumes).sum() / volumes.sum()

                twap_actual = y_true.mean() if len(y_true) > 0 else np.nan
                twap_predicted = y_pred.mean() if len(y_pred) > 0 else np.nan

                logging.info(f"TWAP Actual: {twap_actual:.4f}, TWAP Predicted: {twap_predicted:.4f}")
                logging.info(f"VWAP Actual: {vwap_actual:.4f}, VWAP Predicted: {vwap_predicted:.4f}")

                metrics_dict = {
                    "MSE": mse, "RMSE": rmse, "MAE": mae, "R2_Score": r2, "sMAPE (%)": smape,
                    "TWAP_Actual": twap_actual, "TWAP_Predicted": twap_predicted,
                    "VWAP_Actual": vwap_actual, "VWAP_Predicted": vwap_predicted
                }

                plt.figure(figsize=(14, 7))
                plt.plot(y_true, label='Actual Price', color='blue', alpha=0.7)
                plt.plot(y_pred, label='Predicted Price', color='red', linestyle='--')
                plt.title(f'{coin_name} Price Prediction (Test Set)')
                plt.xlabel('Time Step')
                plt.ylabel('Price (USDT)')
                plt.legend()
                plt.grid(True)
                plt.savefig(plot_path)
                plt.close()
                logging.info(f"Saved prediction plot to: {plot_path}")

            metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
            fig, ax = plt.subplots(figsize=(8, max(4, len(metrics_df) * 0.5))) # Adjusted figsize
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(fontweight='bold')
            plt.title(f'{coin_name} - Evaluation Metrics', fontsize=14, y=0.95 if len(metrics_df) < 7 else 1.0)
            plt.savefig(metrics_image_path, bbox_inches='tight', dpi=200)
            plt.close()
            logging.info(f"Saved metrics image to: {metrics_image_path}")

            metrics_df.to_excel(metrics_excel_path, index=False)
            logging.info(f"Saved metrics Excel to: {metrics_excel_path}")

            logging.info(f"Training process for {coin_name} completed successfully.")
            return model_path, scaler_path, plot_path, metrics_image_path, metrics_excel_path, metrics_dict

        except Exception as e:
            if not isinstance(e, CustomException):
                raise CustomException(str(e), sys) from e
            raise e


if __name__ == "__main__":
    data_paths_json_file = PROJECT_ROOT / "data_paths.json"

    try:
        with open(data_paths_json_file, 'r') as f:
            all_data_paths = json.load(f)
        logging.info(f"Successfully loaded data paths from {data_paths_json_file}")
    except FileNotFoundError:
        sys.exit(1)
    except json.JSONDecodeError:
        sys.exit(1)
    except Exception:
        sys.exit(1)

    coins_to_process = list(all_data_paths.keys())

    for coin_name in coins_to_process:
        logging.info(f"--- Starting training pipeline for {coin_name} ---")

        if coin_name not in all_data_paths:
            continue

        coin_specific_paths = all_data_paths[coin_name]
        train_path = coin_specific_paths.get("train")
        val_path = coin_specific_paths.get("val")
        test_path = coin_specific_paths.get("test")

        if not all([train_path, val_path, test_path]):
            continue

        train_path = os.path.join(PROJECT_ROOT, train_path) if not os.path.isabs(train_path) else train_path
        val_path = os.path.join(PROJECT_ROOT, val_path) if not os.path.isabs(val_path) else val_path
        test_path = os.path.join(PROJECT_ROOT, test_path) if not os.path.isabs(test_path) else test_path

        if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            continue

        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Loaded data for {coin_name}.")
        except Exception:
            continue

        if train_df.empty or val_df.empty:
            continue

        trainer = ModelTrainer()
        try:
            output_base_for_trainer = os.path.dirname(os.path.dirname(train_path))

            model_path_res, scaler_path_res, plot_path_res, metrics_img_path_res, metrics_xlsx_path_res, metrics_res = trainer.train_model(
                coin_name, train_df, val_df, test_df, output_base_for_trainer
            )
            logging.info(f"Training completed for {coin_name}.")
            logging.info(f"Model: {model_path_res}")
        except CustomException:
            pass
        except Exception:
            pass

        logging.info(f"--- Completed training pipeline for {coin_name} ---")
