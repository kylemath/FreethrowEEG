#!/usr/bin/env python3
"""
FreethrowEEG - Records EEG data during freethrow shooting.

This program records your brain activity while shooting free throws,
helping you understand your mental state during successful and unsuccessful shots.
"""

import sys
from src.freethrow_app import FreethrowApp
from src.utils.logger import logger

def main():
    """Main entry point for the application."""
    app = FreethrowApp()
    try:
        logger.info("Starting FreethrowEEG application")
        app.start()
    except KeyboardInterrupt:
        logger.info("\nStopping application...")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        app.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main() 