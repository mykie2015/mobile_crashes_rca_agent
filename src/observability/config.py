"""
Configuration for observability features
"""

import os
import yaml
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class ObservabilityConfig(BaseModel):
    """Configuration for observability features"""
    
    # Langfuse Configuration
    langfuse_enabled: bool = Field(default=False, description="Enable Langfuse observability")
    langfuse_host: str = Field(default="http://localhost:3000", description="Langfuse server URL")
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    
    # Tracing Configuration
    trace_llm_calls: bool = Field(default=True, description="Trace LLM API calls")
    trace_agent_steps: bool = Field(default=True, description="Trace agent workflow steps")
    trace_tool_usage: bool = Field(default=True, description="Trace tool usage")
    
    # Performance Monitoring
    track_token_usage: bool = Field(default=True, description="Track token usage")
    track_latency: bool = Field(default=True, description="Track response latency")
    track_costs: bool = Field(default=True, description="Track API costs")
    
    # Session Configuration
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default="system", description="User identifier")
    
    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Create configuration from environment variables"""
        return cls(
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            langfuse_host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            trace_llm_calls=os.getenv("TRACE_LLM_CALLS", "true").lower() == "true",
            trace_agent_steps=os.getenv("TRACE_AGENT_STEPS", "true").lower() == "true",
            trace_tool_usage=os.getenv("TRACE_TOOL_USAGE", "true").lower() == "true",
            track_token_usage=os.getenv("TRACK_TOKEN_USAGE", "true").lower() == "true",
            track_latency=os.getenv("TRACK_LATENCY", "true").lower() == "true",
            track_costs=os.getenv("TRACK_COSTS", "true").lower() == "true",
            session_id=os.getenv("SESSION_ID"),
            user_id=os.getenv("USER_ID", "system")
        )
    
    @classmethod
    def from_config_file(cls, config_path: str = "config.yml") -> "ObservabilityConfig":
        """Create configuration from YAML config file with environment variable overrides"""
        # Try to find config file in multiple locations
        possible_paths = [
            config_path,
            f"../{config_path}",
            f"../../{config_path}",
            os.path.join(os.path.dirname(__file__), f"../../{config_path}")
        ]
        
        config_data = {}
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                break
        
        # Extract observability config
        obs_config = config_data.get('observability', {})
        langfuse_config = obs_config.get('langfuse', {})
        
        # Environment variables override config file
        return cls(
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", str(obs_config.get("langfuse_enabled", False))).lower() == "true",
            langfuse_host=os.getenv("LANGFUSE_HOST") or langfuse_config.get("host", "http://localhost:3000"),
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY") or langfuse_config.get("public_key"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY") or langfuse_config.get("secret_key"),
            trace_llm_calls=os.getenv("TRACE_LLM_CALLS", str(obs_config.get("trace_llm_calls", True))).lower() == "true",
            trace_agent_steps=os.getenv("TRACE_AGENT_STEPS", str(obs_config.get("trace_agent_steps", True))).lower() == "true",
            trace_tool_usage=os.getenv("TRACE_TOOL_USAGE", str(obs_config.get("trace_tool_usage", True))).lower() == "true",
            track_token_usage=os.getenv("TRACK_TOKEN_USAGE", str(obs_config.get("track_token_usage", True))).lower() == "true",
            track_latency=os.getenv("TRACK_LATENCY", str(obs_config.get("track_latency", True))).lower() == "true",
            track_costs=os.getenv("TRACK_COSTS", str(obs_config.get("track_costs", True))).lower() == "true",
            session_id=os.getenv("SESSION_ID"),
            user_id=os.getenv("USER_ID", "system")
        ) 