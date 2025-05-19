import logging
import traceback
import sys
from datetime import datetime
from pathlib import Path
from app.config.settings import settings  

class ErrorTraceFormatter(logging.Formatter):
    """Custom formatter that includes full traceback information"""
    def format(self, record):
        message = super().format(record)
        if record.exc_info:
            return f"{message}\nFull traceback:\n{''.join(traceback.format_exception(*record.exc_info))}"
        elif hasattr(record, 'stack_info') and record.stack_info:
            return f"{message}\nCall stack:\n{record.stack_info}"
        return message

def setup_logging():
    log_level = getattr(logging, settings.log_level.upper(), logging.DEBUG)
    settings.logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = settings.logs_dir / f"app_{timestamp}.log"
    formatter = ErrorTraceFormatter(settings.log_format)

    # Specify encoding explicitly for FileHandler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("app-logger")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    
    # In dev mode, add console handler with encoding
    if settings.env.lower() == "dev":
        console_handler = logging.StreamHandler()
        # If your StreamHandler supports it, set encoding (some might not support direct parameter)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger

def error_with_trace(msg, *args, **kwargs):
    kwargs['stack_info'] = True
    kwargs['exc_info'] = sys.exc_info()
    logger.error(msg, *args, **kwargs)

logger = setup_logging()

# Export logging methods
debug = logger.debug
info = logger.info
warning = logger.warning
error = error_with_trace
critical = logger.critical

__all__ = ['logger', 'debug', 'info', 'warning', 'error', 'critical']