"""
Langfuse handler for observability and tracking
"""

import time
import uuid
from typing import Any, Dict, List, Optional, Union
from functools import wraps
import logging
from contextlib import contextmanager

try:
    from langfuse import Langfuse
    from langfuse.decorators import observe, langfuse_context
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create dummy classes for when Langfuse is not available
    class Langfuse:
        def __init__(self, *args, **kwargs): pass
        def trace(self, *args, **kwargs): return self
        def span(self, *args, **kwargs): return self
        def generation(self, *args, **kwargs): return self
        def score(self, *args, **kwargs): return self
        def flush(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def end(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): return self
    
    def observe(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    langfuse_context = None

from .config import ObservabilityConfig

logger = logging.getLogger(__name__)


class LangfuseHandler:
    """Handler for Langfuse observability integration"""
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        """Initialize Langfuse handler"""
        self.config = config or ObservabilityConfig.from_env()
        self.langfuse = None
        self.current_trace = None
        
        if self.config.langfuse_enabled and LANGFUSE_AVAILABLE:
            self._initialize_langfuse()
        else:
            logger.warning("Langfuse is disabled or not available. Observability features will be limited.")
    
    def _initialize_langfuse(self):
        """Initialize Langfuse client"""
        try:
            if not self.config.langfuse_public_key or not self.config.langfuse_secret_key:
                logger.warning("Langfuse keys not provided. Using default test keys.")
                
            self.langfuse = Langfuse(
                public_key=self.config.langfuse_public_key or "lf-pk-test",
                secret_key=self.config.langfuse_secret_key or "lf-sk-test",
                host=self.config.langfuse_host
            )
            logger.info(f"Langfuse initialized with host: {self.config.langfuse_host}")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self.langfuse = None
    
    @contextmanager
    def trace_session(self, name: str, session_id: Optional[str] = None, 
                     user_id: Optional[str] = None, **metadata):
        """Context manager for tracing a complete session"""
        if not self.langfuse:
            yield None
            return
        
        session_id = session_id or self.config.session_id or str(uuid.uuid4())
        user_id = user_id or self.config.user_id
        
        trace = self.langfuse.trace(
            name=name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        old_trace = self.current_trace
        self.current_trace = trace
        
        try:
            yield trace
        finally:
            self.current_trace = old_trace
            if self.langfuse:
                self.langfuse.flush()
    
    @contextmanager
    def trace_agent_step(self, name: str, **metadata):
        """Context manager for tracing an agent step"""
        if not self.langfuse or not self.current_trace:
            yield None
            return
        
        span = self.current_trace.span(
            name=name,
            metadata=metadata
        )
        
        start_time = time.time()
        try:
            yield span
        except Exception as e:
            span.update(
                level="ERROR",
                status_message=str(e)
            )
            raise
        finally:
            end_time = time.time()
            span.update(
                end_time=end_time,
                metadata={
                    **metadata,
                    "duration_ms": round((end_time - start_time) * 1000, 2)
                }
            )
            span.end()
    
    @contextmanager
    def trace_llm_call(self, name: str, model: str, input_data: Any, **metadata):
        """Context manager for tracing LLM calls"""
        if not self.langfuse or not self.current_trace:
            yield None
            return
        
        generation = self.current_trace.generation(
            name=name,
            model=model,
            input=input_data,
            metadata=metadata
        )
        
        start_time = time.time()
        try:
            yield generation
        except Exception as e:
            generation.update(
                level="ERROR",
                status_message=str(e)
            )
            raise
        finally:
            end_time = time.time()
            generation.update(
                end_time=end_time,
                metadata={
                    **metadata,
                    "duration_ms": round((end_time - start_time) * 1000, 2)
                }
            )
            generation.end()
    
    def log_llm_response(self, generation, output: str, usage: Optional[Dict] = None, 
                        cost: Optional[float] = None):
        """Log LLM response data"""
        if not generation:
            return
        
        update_data = {"output": output}
        
        if usage:
            update_data["usage"] = usage
        
        if cost:
            update_data["cost"] = cost
        
        generation.update(**update_data)
    
    def log_tool_usage(self, tool_name: str, inputs: Dict, outputs: Any, 
                      duration_ms: Optional[float] = None):
        """Log tool usage"""
        if not self.langfuse or not self.current_trace:
            return
        
        metadata = {
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": str(outputs)[:1000],  # Truncate long outputs
        }
        
        if duration_ms:
            metadata["duration_ms"] = duration_ms
        
        self.current_trace.span(
            name=f"tool_{tool_name}",
            metadata=metadata
        ).end()
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error information"""
        if not self.langfuse or not self.current_trace:
            return
        
        self.current_trace.span(
            name="error",
            level="ERROR",
            metadata={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context
            }
        ).end()
    
    def add_score(self, name: str, value: Union[int, float], comment: Optional[str] = None):
        """Add a score to the current trace"""
        if not self.langfuse or not self.current_trace:
            return
        
        self.langfuse.score(
            trace_id=self.current_trace.id,
            name=name,
            value=value,
            comment=comment
        )
    
    def flush(self):
        """Flush all pending data to Langfuse"""
        if self.langfuse:
            self.langfuse.flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush() 