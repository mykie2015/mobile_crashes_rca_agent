"""Error handling utilities for the Mobile Crashes RCA Agent."""

import logging
import traceback
from typing import Any, Dict, Optional
from datetime import datetime

class RCAError(Exception):
    """Base exception for RCA Agent errors."""
    pass

class CrashAnalysisError(RCAError):
    """Exception for crash analysis specific errors."""
    pass

class LLMConnectionError(RCAError):
    """Exception for LLM connection issues."""
    pass

class DataValidationError(RCAError):
    """Exception for data validation issues."""
    pass

class ErrorHandler:
    """Centralized error handling for the RCA Agent."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_counts = {}
    
    def handle_error(self, error: Exception, context: str = "", data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle and log errors with context."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Track error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        error_info = {
            'error_type': error_type,
            'error_message': error_message,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'count': self.error_counts[error_type],
            'traceback': traceback.format_exc(),
            'data': data or {}
        }
        
        # Log based on error type
        if isinstance(error, (ConnectionError, LLMConnectionError)):
            self.logger.error(f"Connection error in {context}: {error_message}")
        elif isinstance(error, DataValidationError):
            self.logger.warning(f"Data validation error in {context}: {error_message}")
        elif isinstance(error, CrashAnalysisError):
            self.logger.error(f"Crash analysis error in {context}: {error_message}")
        else:
            self.logger.error(f"Unexpected error in {context}: {error_message}")
        
        return error_info
    
    def retry_with_backoff(self, func, max_retries: int = 3, backoff_factor: float = 2.0):
        """Retry a function with exponential backoff."""
        import time
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    self.handle_error(e, f"Final retry attempt {attempt + 1}")
                    raise
                
                wait_time = backoff_factor ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
    
    def validate_crash_data(self, crash_data: Any) -> bool:
        """Validate crash data structure."""
        try:
            if not isinstance(crash_data, (list, dict)):
                raise DataValidationError("Crash data must be a list or dictionary")
            
            if isinstance(crash_data, list):
                for item in crash_data:
                    if not isinstance(item, dict):
                        raise DataValidationError("Each crash item must be a dictionary")
                    
                    required_fields = ['id', 'timestamp', 'exception']
                    for field in required_fields:
                        if field not in item:
                            raise DataValidationError(f"Missing required field: {field}")
            
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(f"Validation failed: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered."""
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_breakdown': self.error_counts,
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
        } 