import sys
import logging

def error_message_detail(error, error_detail: sys):
    """Function to return detailed error message including the file name, line number, and error message."""
    _, _, exc_tb = error_detail.exc_info()  # Capture error details
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name
    error_message = f"Error occurred in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message[{str(error)}]"
    return error_message

class CustomException(Exception):
    """Custom Exception class to log error details."""
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Store detailed error message

    def __str__(self):
        return self.error_message

    def log_error(self):
        """Log the custom error to a log file using the logging module."""
        logging.error(f"Custom Exception: {self.error_message}")  # Log the error message with error level
