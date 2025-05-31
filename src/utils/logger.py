"""Logging utilities for the Mobile Crashes RCA Agent."""

import logging
import logging.config
from pathlib import Path
from datetime import datetime
from config import LOGGING_CONFIG

def setup_logging():
    """Setup logging configuration and create necessary directories."""
    # Create directories
    today = datetime.now().strftime("%Y%m%d")
    directories = [
        'logs',
        f'logs/{today}',
        'data/outputs',
        'data/outputs/images',
        'data/cache'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Setup logging with daily organization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update log file paths with timestamp
    log_config = LOGGING_CONFIG.get('logging', {})
    if 'handlers' in log_config:
        if 'file' in log_config['handlers']:
            log_config['handlers']['file']['filename'] = f"logs/{today}/rca_agent_{timestamp}.log"
        if 'daily_file' in log_config['handlers']:
            log_config['handlers']['daily_file']['filename'] = f"logs/{today}/rca_agent_daily_{timestamp}.log"
    
    # Apply logging configuration
    try:
        logging.config.dictConfig(log_config)
    except Exception as e:
        # Fallback to basic logging if config fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/{today}/rca_agent_{timestamp}.log"),
                logging.StreamHandler()
            ]
        )
        print(f"Warning: Failed to apply logging config, using fallback: {e}")
    
    logger = logging.getLogger('RCA_Agent')
    logger.info(f"Starting Mobile Crashes RCA Agent - Log file: logs/{today}/rca_agent_{timestamp}.log")
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance."""
    if name is None:
        name = 'RCA_Agent'
    return logging.getLogger(name)

def setup_directories_and_logging():
    """Legacy function for backward compatibility."""
    return setup_logging() 