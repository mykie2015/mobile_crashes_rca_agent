"""OpenAI GPT client implementation."""

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from .base import BaseLLMClient

class GPTClient(BaseLLMClient):
    """OpenAI GPT client using LangChain."""
    
    def __init__(self, model_name: str, api_key: str, api_base: str = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        self.api_base = api_base
        self._client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            config = {
                'model': self.model_name,
                'api_key': self.api_key,
                'temperature': self.config.get('temperature', 0.1),
                'max_tokens': self.config.get('max_tokens', 4000)
            }
            
            if self.api_base:
                config['base_url'] = self.api_base
            
            self._client = ChatOpenAI(**config)
            self.logger.info(f"Initialized GPT client with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize GPT client: {e}")
            raise
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Generate a completion for the given prompt."""
        try:
            messages = [("human", prompt)]
            response = self._client.invoke(messages)
            return response.content
        except Exception as e:
            self.logger.error(f"GPT completion failed: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate a chat completion for the given messages."""
        try:
            # Convert to LangChain format
            langchain_messages = []
            for msg in messages:
                role = "human" if msg["role"] == "user" else msg["role"]
                langchain_messages.append((role, msg["content"]))
            
            response = self._client.invoke(langchain_messages)
            return response.content
        except Exception as e:
            self.logger.error(f"GPT chat completion failed: {e}")
            raise
    
    def invoke(self, messages: List[tuple]) -> Any:
        """Direct invoke method for LangChain compatibility."""
        try:
            return self._client.invoke(messages)
        except Exception as e:
            self.logger.error(f"GPT invoke failed: {e}")
            raise
    
    def validate_connection(self) -> bool:
        """Validate that the client can connect to the API."""
        try:
            test_response = self.complete("Hello")
            return bool(test_response)
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    @property
    def client(self):
        """Get the underlying LangChain client."""
        return self._client 