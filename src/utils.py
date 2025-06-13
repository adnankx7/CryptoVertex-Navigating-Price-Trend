import os
import sys
import pickle
import pathlib

# --- Determine PROJECT_ROOT and add src to sys.path if not already present ---
try:
    SCRIPT_DIR_UTIL = pathlib.Path(__file__).resolve().parent
    PROJECT_ROOT_UTIL = SCRIPT_DIR_UTIL.parent
except NameError:
    PROJECT_ROOT_UTIL = pathlib.Path.cwd()

if str(PROJECT_ROOT_UTIL) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_UTIL))
# --- End of PROJECT_ROOT setup ---

try:
    from src.logger import logging
    from src.exception import CustomException
except ImportError:
    class LoggingPlaceholder:
        def info(self, message): pass
    logging = LoggingPlaceholder()  # type: ignore

    class CustomException(Exception):  # type: ignore
        def __init__(self, error_message, error_detail: sys = None):
            super().__init__(error_message)
            if error_detail and hasattr(error_detail, 'exc_info') and error_detail.exc_info()[2]:
                tb = error_detail.exc_info()[2]
                self.error_message = (f"Error in {tb.tb_frame.f_code.co_filename} line {tb.tb_lineno}: {error_message}")
            else:
                self.error_message = f"Error: {error_message}"
        def __str__(self): return self.error_message

def save_object(file_path: str, obj: object):
    """
    Saves a Python object to a file using pickle.
    Creates the directory if it doesn't exist.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object saved successfully to: {file_path}")
    except Exception as e:
        raise CustomException(f"Error saving object: {str(e)}", sys)

def load_pickle_object(file_path: str) -> object:
    """
    Loads a Python object from a pickle file.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No such file or directory: {file_path}")
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f"Object loaded successfully from: {file_path}")
        return obj
    except FileNotFoundError as e:
        raise CustomException(str(e), sys)
    except pickle.UnpicklingError as e:
        raise CustomException(f"Error unpickling object: {str(e)}", sys)
    except Exception as e:
        raise CustomException(f"Error loading pickle: {str(e)}", sys)

if __name__ == '__main__':
    logging.info("UTLIS.py executed directly for self-test.")
    try:
        test_obj = {"test_key": "test_value_utlis", "numbers": [100, 200, 300]}
        temp_test_dir = PROJECT_ROOT_UTIL / "temp_utlis_module_test"  # type: ignore
        temp_test_dir.mkdir(exist_ok=True)
        test_file_path = temp_test_dir / "temp_utlis_object.pkl"

        save_object(str(test_file_path), test_obj)
        logging.info(f"Test object saved to: {test_file_path}")

        loaded_obj = load_pickle_object(str(test_file_path))
        logging.info(f"Loaded test object: {loaded_obj}")
        assert loaded_obj == test_obj

        if test_file_path.exists():
            test_file_path.unlink()
        if temp_test_dir.exists() and not any(temp_test_dir.iterdir()):
            temp_test_dir.rmdir()
        logging.info("UTLIS.py self-tests passed.")
    except Exception:
        pass
