"""
Logging Utilities Module
========================

This module provides logging setup and utilities for the AMR project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logger(name: str, level: str = "INFO", 
                log_file: Optional[str] = None,
                console_output: bool = True,
                file_output: bool = True,
                format_string: Optional[str] = None) -> logging.Logger:
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        file_output: Whether to output to file
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_project_logging(config: Dict[str, Any], 
                         project_name: str = "amr_project") -> logging.Logger:
    """
    Setup project-wide logging configuration.
    
    Args:
        config: Logging configuration dictionary
        project_name: Name of the project
        
    Returns:
        Main project logger
    """
    # Extract configuration
    log_level = config.get('level', 'INFO')
    log_format = config.get('format', 
                           '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_logging = config.get('console_logging', True)
    file_logging = config.get('file_logging', True)
    log_dir = config.get('log_dir', 'logs')
    
    # Create log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = None
    if file_logging:
        log_file = Path(log_dir) / f"{project_name}_{timestamp}.log"
    
    # Setup main logger
    main_logger = setup_logger(
        name=project_name,
        level=log_level,
        log_file=str(log_file) if log_file else None,
        console_output=console_logging,
        file_output=file_logging,
        format_string=log_format
    )
    
    # Setup module loggers
    module_names = [
        'data_processing',
        'tokenization', 
        'training',
        'evaluation',
        'inference'
    ]
    
    for module_name in module_names:
        module_logger = setup_logger(
            name=f"{project_name}.{module_name}",
            level=log_level,
            log_file=str(log_file) if log_file else None,
            console_output=console_logging,
            file_output=file_logging,
            format_string=log_format
        )
    
    main_logger.info(f"Logging setup completed for {project_name}")
    if log_file:
        main_logger.info(f"Log file: {log_file}")
    
    return main_logger


class LoggerMixin:
    """Mixin class to add logging capabilities to other classes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the class."""
        class_name = self.__class__.__name__
        module_name = self.__class__.__module__
        logger_name = f"{module_name}.{class_name}"
        
        return logging.getLogger(logger_name)
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, logger: logging.Logger, total_items: int, 
                 log_interval: int = 100):
        """
        Initialize progress logger.
        
        Args:
            logger: Logger instance
            total_items: Total number of items to process
            log_interval: Interval for logging progress
        """
        self.logger = logger
        self.total_items = total_items
        self.log_interval = log_interval
        self.processed_items = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1) -> None:
        """
        Update progress counter.
        
        Args:
            increment: Number of items processed
        """
        self.processed_items += increment
        
        if self.processed_items % self.log_interval == 0 or \
           self.processed_items == self.total_items:
            self._log_progress()
    
    def _log_progress(self) -> None:
        """Log current progress."""
        percentage = (self.processed_items / self.total_items) * 100
        elapsed_time = datetime.now() - self.start_time
        
        if self.processed_items > 0:
            avg_time_per_item = elapsed_time.total_seconds() / self.processed_items
            remaining_items = self.total_items - self.processed_items
            estimated_remaining = avg_time_per_item * remaining_items
            
            self.logger.info(
                f"Progress: {self.processed_items}/{self.total_items} "
                f"({percentage:.1f}%) - "
                f"Elapsed: {elapsed_time} - "
                f"ETA: {estimated_remaining:.0f}s"
            )
        else:
            self.logger.info(f"Progress: {self.processed_items}/{self.total_items} ({percentage:.1f}%)")
    
    def finish(self) -> None:
        """Log completion."""
        total_time = datetime.now() - self.start_time
        self.logger.info(
            f"✅ Completed processing {self.total_items} items in {total_time}"
        )


def create_file_logger(log_file: str, logger_name: str = "file_logger",
                      level: str = "INFO") -> logging.Logger:
    """
    Create a dedicated file logger.
    
    Args:
        log_file: Path to log file
        logger_name: Name of the logger
        level: Logging level
        
    Returns:
        File logger instance
    """
    return setup_logger(
        name=logger_name,
        level=level,
        log_file=log_file,
        console_output=False,
        file_output=True
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
