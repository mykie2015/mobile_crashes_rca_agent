"""
Decorators for observability tracing
"""

import time
import functools
from typing import Any, Callable, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Global observability handler - will be set by the main application
_global_handler = None


def set_global_handler(handler):
    """Set the global observability handler"""
    global _global_handler
    _global_handler = handler


def get_global_handler():
    """Get the global observability handler"""
    return _global_handler


def trace_llm_call(model: str, name: Optional[str] = None):
    """Decorator for tracing LLM calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_global_handler()
            if not handler:
                return func(*args, **kwargs)
            
            function_name = name or f"llm_{func.__name__}"
            
            # Extract input from args/kwargs for tracing
            input_data = {"args": str(args)[:500], "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}}
            
            with handler.trace_llm_call(
                name=function_name,
                model=model,
                input_data=input_data
            ) as generation:
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    # Log the response
                    if generation and hasattr(result, 'content'):
                        handler.log_llm_response(
                            generation=generation,
                            output=result.content,
                            usage=getattr(result, 'usage_metadata', None),
                        )
                    elif generation:
                        handler.log_llm_response(
                            generation=generation,
                            output=str(result)[:1000],
                        )
                    
                    return result
                    
                except Exception as e:
                    if handler:
                        handler.log_error(e, f"LLM call {function_name}")
                    raise
                    
        return wrapper
    return decorator


def trace_agent_step(name: Optional[str] = None, metadata: Optional[Dict] = None):
    """Decorator for tracing agent steps"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_global_handler()
            if not handler:
                return func(*args, **kwargs)
            
            step_name = name or f"step_{func.__name__}"
            step_metadata = metadata or {}
            
            with handler.trace_agent_step(name=step_name, **step_metadata) as span:
                try:
                    result = func(*args, **kwargs)
                    
                    # Add result metadata to span if available
                    if span and hasattr(result, 'metadata'):
                        span.update(metadata={**step_metadata, "result_metadata": result.metadata})
                    
                    return result
                    
                except Exception as e:
                    if handler:
                        handler.log_error(e, f"Agent step {step_name}")
                    raise
                    
        return wrapper
    return decorator


def trace_tool_call(tool_name: str):
    """Decorator for tracing tool calls"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_global_handler()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                if handler:
                    handler.log_tool_usage(
                        tool_name=tool_name,
                        inputs={"args": str(args)[:200], "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}},
                        outputs=result,
                        duration_ms=round((end_time - start_time) * 1000, 2)
                    )
                
                return result
                
            except Exception as e:
                end_time = time.time()
                if handler:
                    handler.log_tool_usage(
                        tool_name=tool_name,
                        inputs={"args": str(args)[:200], "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}},
                        outputs=f"ERROR: {str(e)}",
                        duration_ms=round((end_time - start_time) * 1000, 2)
                    )
                    handler.log_error(e, f"Tool call {tool_name}")
                raise
                
        return wrapper
    return decorator


def observe_performance(name: Optional[str] = None):
    """Decorator for observing function performance"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            function_name = name or func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                duration_ms = round((end_time - start_time) * 1000, 2)
                
                logger.info(f"Function {function_name} completed in {duration_ms}ms")
                
                # Log to observability handler if available
                handler = get_global_handler()
                if handler and handler.current_trace:
                    handler.current_trace.span(
                        name=f"performance_{function_name}",
                        metadata={
                            "duration_ms": duration_ms,
                            "function": function_name
                        }
                    ).end()
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration_ms = round((end_time - start_time) * 1000, 2)
                logger.error(f"Function {function_name} failed after {duration_ms}ms: {e}")
                raise
                
        return wrapper
    return decorator 