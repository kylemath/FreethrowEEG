"""
Logging configuration for the FreethrowEEG application.
"""

import logging
import sys

def setup_logger():
    """
    Configure and return the application logger.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('FreethrowEEG')
    return logger

# Create a singleton logger instance
logger = setup_logger() 