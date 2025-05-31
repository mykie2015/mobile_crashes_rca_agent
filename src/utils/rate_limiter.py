"""Rate limiting utilities for API calls."""

import time
import threading
from typing import Dict, Any
from collections import deque
import logging

class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = deque()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            while self.requests and now - self.requests[0] > 60:
                self.requests.popleft()
            
            # Check if we need to wait
            if len(self.requests) >= self.requests_per_minute:
                wait_time = 60 - (now - self.requests[0])
                if wait_time > 0:
                    self.logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                    # Clean up again after waiting
                    now = time.time()
                    while self.requests and now - self.requests[0] > 60:
                        self.requests.popleft()
            
            # Record this request
            self.requests.append(now) 