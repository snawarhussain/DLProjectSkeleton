import os
import logging
import sys
from utils.config import ProjectConfig
from colorama import Fore, Style
# Ensure the directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ConsoleAndFileLogger:
    def __init__(self, log_file):
        self.terminal = sys.stdout  # Save the original stdout
        self.log_file = open(log_file, 'a')  # Open the log file in append mode

    def write(self, message):
        # Write to both terminal and log file
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()  # Ensure messages are flushed immediately

    def flush(self):
        # Flush both streams
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

def link_file(src, target):
    if os.path.isdir(target) or os.path.isfile(target):
        os.remove(target)
    os.system('ln -s {} {}'.format(src, target))

# Custom console formatter with colors
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Add colors based on the log level
        if record.levelno == logging.DEBUG:
            record.msg = f"{Fore.CYAN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.INFO:
            record.msg = f"{Fore.GREEN}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.CRITICAL:
            record.msg = f"{Fore.RED}{Style.BRIGHT}{record.msg}{Style.RESET_ALL}"
        return super().format(record)
    
def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        
def print_and_log_info(logger, string):
    logger.info(string)
    # print(string)

def get_logger(config:ProjectConfig, name='train'):
    log_path = os.path.join(config.project_directory, 'logs')
    log_dir = os.path.dirname(log_path)
    ensure_dir(log_dir)

    logger = logging.getLogger(name)
    # File handler for writing logs to a file
    file_handler = logging.FileHandler(log_path, mode='a')
    console_formatter = ColoredFormatter("[%(asctime)s] %(levelname)s: %(message)s")
    file_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)

    # Console handler for colored output
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_formatter = ColoredFormatter("[%(asctime)s] %(levelname)s: %(message)s")
    # console_handler.setFormatter(console_formatter)
    # logger.addHandler(console_handler)
    # Redirect stdout and stderr
    sys.stdout = ConsoleAndFileLogger(log_path)
    sys.stderr = sys.stdout  # Redirect stderr to the same stream
    return logger
