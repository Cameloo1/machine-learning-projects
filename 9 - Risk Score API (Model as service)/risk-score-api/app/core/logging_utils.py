"""JSON logging utilities."""
import json
import logging
from typing import Dict


_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str = "risk_score_api") -> logging.Logger:
    """
    Get or create a logger configured for JSON logging to stdout.
    
    Avoids adding duplicate handlers if called multiple times.
    
    Args:
        name: Logger name (default: "risk_score_api").
        
    Returns:
        Configured logger instance.
    """
    # Return existing logger if already configured
    if name in _loggers:
        return _loggers[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if logger.handlers:
        _loggers[name] = logger
        return logger
    
    # Create handler for stdout
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    
    # Simple formatter (just the message, which will be JSON)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    logger.propagate = False  # Prevent propagation to root logger
    
    _loggers[name] = logger
    return logger


def log_event(logger: logging.Logger, event: str, payload: Dict) -> None:
    """
    Log a JSON string with event and payload data.
    
    The logged JSON will have:
    - "event": event name
    - All keys from payload
    
    Args:
        logger: Logger instance to use.
        event: Event name/type.
        payload: Dictionary of additional data to include in the log.
    """
    log_data = {
        "event": event,
        **payload
    }
    
    log_message = json.dumps(log_data)
    logger.info(log_message)
