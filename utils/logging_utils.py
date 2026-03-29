"""
Colored logging utility for VLS.

Usage:
    from utils.logging_utils import get_logger
    
    logger = get_logger(__name__)
    logger.info("This is info")
    logger.debug("This is debug")
    logger.warning("This is warning")
    logger.error("This is error")
"""

import logging
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
        'RESET': '\033[0m',       # Reset
        'BOLD': '\033[1m',
        'CYAN': '\033[96m',
        'HEADER': '\033[95m',
    }
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        if self.use_colors:
            levelname = record.levelname
            color = self.COLORS.get(levelname, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            bold = self.COLORS['BOLD']
            
            # Color the level name
            record.levelname = f"{color}{bold}{levelname}{reset}"
            
            # Color the message based on level
            record.msg = f"{color}{record.msg}{reset}"
        
        return super().format(record)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Get a colored logger instance.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: DEBUG)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Create formatter
    fmt = "%(levelname)s [%(name)s] %(message)s"
    formatter = ColoredFormatter(fmt, use_colors=True)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


# Convenience function for module-level logging with custom prefix
class SteerLogger:
    """
    A simple wrapper for colored logging with module prefix.
    
    Usage:
        log = SteerLogger("KeypointTracker")
        log.info("Got 5 keypoints")
        log.debug("Processing frame 10")
        log.error("Failed to detect keypoints")
    """
    
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green  
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[95m',   # Magenta
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'CYAN': '\033[96m',
    }
    
    def __init__(self, name: str, level: int = logging.DEBUG):
        self.name = name
        self.logger = get_logger(name, level)
    
    def _format(self, msg: str, color: str) -> str:
        reset = self.COLORS['RESET']
        bold = self.COLORS['BOLD']
        return f"{color}{bold}[{self.name}]{reset} {color}{msg}{reset}"
    
    def debug(self, msg: str):
        self.logger.debug(msg)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def critical(self, msg: str):
        self.logger.critical(msg)


# Quick access function
def log_info(prefix: str, msg: str):
    """Quick colored info log."""
    print(f"\033[92m\033[1m[{prefix}]\033[0m \033[92m{msg}\033[0m")

def log_debug(prefix: str, msg: str):
    """Quick colored debug log."""
    print(f"\033[94m\033[1m[{prefix}]\033[0m \033[94m{msg}\033[0m")

def log_warning(prefix: str, msg: str):
    """Quick colored warning log."""
    print(f"\033[93m\033[1m[{prefix}]\033[0m \033[93m{msg}\033[0m")

def log_error(prefix: str, msg: str):
    """Quick colored error log."""
    print(f"\033[91m\033[1m[{prefix}]\033[0m \033[91m{msg}\033[0m")

