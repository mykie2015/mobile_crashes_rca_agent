"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat completion for the given messages."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that the client can connect to the API."""
        pass
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for the given text."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def format_messages(self, human_message: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """Format messages for chat completion."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": human_message})
        return messages 