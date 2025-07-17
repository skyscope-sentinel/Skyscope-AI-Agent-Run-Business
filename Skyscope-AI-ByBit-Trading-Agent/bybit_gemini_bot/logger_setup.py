import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "trading_bot.log")

def setup_logging(log_level=logging.INFO, log_file=DEFAULT_LOG_FILE, max_bytes=10*1024*1024, backup_count=5):
    """
    Configures logging to write to both console and a rotating file.

    Args:
        log_level (int, optional): The logging level (e.g., logging.INFO, logging.DEBUG).
                                   Defaults to logging.INFO.
        log_file (str, optional): Path to the log file.
                                  Defaults to 'logs/trading_bot.log'.
        max_bytes (int, optional): Maximum size of the log file in bytes before rotation.
                                   Defaults to 10MB.
        backup_count (int, optional): Number of backup log files to keep.
                                      Defaults to 5.
    """
    # Ensure the log directory exists
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            print(f"Error creating log directory '{LOG_DIR}': {e}")
            # Fallback to console-only logging if directory creation fails
            _configure_console_logging(log_level)
            logging.error(f"Could not create log directory. File logging disabled. Error: {e}")
            return

    # Create a root logger
    logger = logging.getLogger()
    logger.setLevel(log_level) # Set the root logger level

    # Clear existing handlers to avoid duplicate logs if setup_logging is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter
    log_format = "%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Failed to set up file logging for '{log_file}': {e}. Logging to console only.")
        # Ensure console logging is still active if file handler fails
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
             _configure_console_logging(log_level, formatter)


def _configure_console_logging(log_level, formatter=None):
    """Helper to configure basic console logging, used as fallback."""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.hasHandlers(): # Clear any previous handlers
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    if formatter is None:
        log_format = "%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s"
        formatter = logging.Formatter(log_format)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.info("Configured fallback console logging.")


if __name__ == '__main__':
    # Example Usage:
    setup_logging(logging.DEBUG) # Set to DEBUG for more verbose output during testing

    logging.debug("This is a debug message.")
    logging.info("This is an info message.")
    logging.warning("This is a warning message.")
    logging.error("This is an error message.")
    logging.critical("This is a critical message.")

    # Test logger from another module (simulated)
    test_logger = logging.getLogger("my_module_test") # Get logger instance
    test_logger.info("Test message from a simulated different module.")

    # Test log rotation (manual trigger for demonstration is complex,
    # but you can check if the file is created and logs are written)
    print(f"\nLogging to console and to: {os.path.abspath(DEFAULT_LOG_FILE)}")
    print(f"Log directory is: {os.path.abspath(LOG_DIR)}")

    # Example of how another module would get the logger (it gets the root logger)
    # import logging
    # logger = logging.getLogger(__name__) # This gets a logger named 'logger_setup' or 'package.logger_setup'
    # logger.info("Another message from logger_setup itself using getLogger(__name__)")

    # logger_from_root = logging.getLogger() # This gets the root logger configured by setup_logging
    # logger_from_root.info("Message from root logger obtained via getLogger()")

    # Verify log directory creation if it didn't exist
    if not os.path.exists(LOG_DIR):
        print(f"Log directory '{LOG_DIR}' was NOT created (this is unexpected).")
    else:
        print(f"Log directory '{LOG_DIR}' exists.")

    # Verify log file creation
    if not os.path.exists(DEFAULT_LOG_FILE):
         print(f"Log file '{DEFAULT_LOG_FILE}' was NOT created (this is unexpected if no errors occurred).")
    else:
        print(f"Log file '{DEFAULT_LOG_FILE}' exists.")
        # You can manually check the content of logs/trading_bot.log
        # For a real test, you might read the file and check its content.
        try:
            with open(DEFAULT_LOG_FILE, 'r') as f:
                content = f.read()
                if "This is an info message." in content:
                    print("Successfully wrote to log file.")
                else:
                    print("Did not find expected message in log file.")
        except Exception as e:
            print(f"Error reading log file: {e}")
