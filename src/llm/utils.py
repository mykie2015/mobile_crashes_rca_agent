"""LLM utility functions."""

from typing import Union
from .base import BaseLLMClient
from .gpt_client import GPTClient
from .claude_client import ClaudeClient
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import MODEL_CONFIG

def get_llm_client(provider: str = 'openai', **kwargs) -> BaseLLMClient:
    """Get an LLM client instance based on provider."""
    
    if provider == 'openai':
        config = MODEL_CONFIG.get('models', {}).get('openai', {})
        return GPTClient(
            model_name=config.get('default_model', 'gpt-4o-mini'),
            api_key=config.get('api_key'),
            api_base=config.get('api_base'),
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 4000),
            **kwargs
        )
    elif provider == 'claude':
        config = MODEL_CONFIG.get('models', {}).get('claude', {})
        return ClaudeClient(
            model_name=config.get('default_model', 'claude-3-sonnet-20240229'),
            api_key=config.get('api_key'),
            temperature=config.get('temperature', 0.1),
            max_tokens=config.get('max_tokens', 4000),
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def create_evaluator_llm(**kwargs) -> BaseLLMClient:
    """Create an LLM client specifically for evaluation tasks."""
    eval_config = MODEL_CONFIG.get('evaluation', {})
    
    # Use OpenAI for evaluation with low temperature for consistency
    return get_llm_client(
        provider='openai',
        temperature=eval_config.get('temperature', 0.1),
        **kwargs
    ) 