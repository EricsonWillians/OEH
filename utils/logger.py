"""
utils/logger.py

This module provides a centralized logging utility for the OEH project.
It configures and returns loggers that output detailed, timestamped messages
to standard output and optionally to file, ensuring consistency across all project modules.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.debug("Debugging message")
    
    # Optional file logging
    from utils.logger import setup_file_logging
    setup_file_logging("simulation.log")
"""

import logging
import sys
import os
from typing import Any, Optional

def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with the specified name. If the logger is not already configured,
    it sets up a StreamHandler that outputs to sys.stdout using a detailed log format.
    
    Args:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set log level; you may adjust this or integrate with a configuration setting.
        logger.setLevel(logging.DEBUG)

        # Define a detailed formatter.
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Create a StreamHandler to output logs to standard output.
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        # Add the handler to the logger.
        logger.addHandler(handler)

        # Optionally, disable propagation to avoid duplicate logs.
        logger.propagate = False

    return logger

def setup_file_logging(log_file: str, level: int = logging.DEBUG) -> None:
    """
    Sets up logging to a file for all loggers in the application.
    
    Args:
        log_file (str): Path to the log file
        level (int): Logging level for the file handler
    """
    # Create the directory for the log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure a file handler
    root_logger = logging.getLogger()
    
    # Set the root logger level to ensure logs of the specified level are processed
    if root_logger.level == 0:  # Default level is 0 (NOTSET)
        root_logger.setLevel(level)
    
    # Define a detailed formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create and add the file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Add to root logger so all loggers can output to this file
    root_logger.addHandler(file_handler)
    
    # Log that file logging has been set up
    logger = get_logger("utils.logger")
    logger.info(f"File logging set up to: {log_file}")