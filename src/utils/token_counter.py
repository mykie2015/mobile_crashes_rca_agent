"""Token counting utilities."""

import re
import logging
from typing import Union, Dict, Any

class TokenCounter:
    """Utility for counting tokens in text."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def count_tokens(self, text: str, method: str = 'simple') -> int:
        """Count tokens in text using different methods."""
        if method == 'simple':
            return self._simple_count(text)
        elif method == 'words':
            return self._word_count(text)
        elif method == 'tiktoken':
            return self._tiktoken_count(text)
        else:
            raise ValueError(f"Unknown counting method: {method}")
    
    def _simple_count(self, text: str) -> int:
        """Simple token estimation: ~4 characters per token."""
        return len(text) // 4
    
    def _word_count(self, text: str) -> int:
        """Word-based token estimation: ~1.3 tokens per word."""
        words = len(re.findall(r'\b\w+\b', text))
        return int(words * 1.3)
    
    def _tiktoken_count(self, text: str) -> int:
        """Use tiktoken for accurate GPT token counting."""
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            return len(encoding.encode(text))
        except ImportError:
            self.logger.warning("tiktoken not available, falling back to simple count")
            return self._simple_count(text)
    
    def estimate_cost(self, tokens: int, model: str = 'gpt-4o-mini') -> float:
        """Estimate cost based on token count and model."""
        # Rough cost estimates (input tokens)
        costs_per_1k = {
            'gpt-4o-mini': 0.000015,  # $0.015 per 1K tokens
            'gpt-4': 0.03,            # $30 per 1M tokens
            'gpt-3.5-turbo': 0.0015,  # $1.50 per 1M tokens
        }
        
        cost_per_token = costs_per_1k.get(model, 0.000015) / 1000
        return tokens * cost_per_token
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze a prompt and return token statistics."""
        simple_count = self._simple_count(prompt)
        word_count = self._word_count(prompt)
        tiktoken_count = self._tiktoken_count(prompt)
        
        return {
            'text_length': len(prompt),
            'simple_token_count': simple_count,
            'word_based_count': word_count,
            'tiktoken_count': tiktoken_count,
            'estimated_cost_gpt4mini': self.estimate_cost(tiktoken_count, 'gpt-4o-mini'),
            'estimated_cost_gpt4': self.estimate_cost(tiktoken_count, 'gpt-4')
        } 