import logging
import sys

def setup_logger(name: str = "benchmark", level: str = "INFO") -> logging.Logger:
    class SimpleFormatter(logging.Formatter):
        
        FORMATS = {
            logging.DEBUG: "[DEBUG] %(message)s",
            logging.INFO: "%(message)s",
            logging.WARNING: "[WARNING] %(message)s",
            logging.ERROR: "[ERROR] %(message)s",
        }
        
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(SimpleFormatter())
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logger()