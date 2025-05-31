"""Configuration management for the Mobile Crashes RCA Agent."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_name: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    config = _replace_env_vars(config)
    return config

def _replace_env_vars(obj: Any) -> Any:
    """Recursively replace environment variables in configuration."""
    if isinstance(obj, dict):
        return {k: _replace_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
        env_spec = obj[2:-1]
        if ':' in env_spec:
            env_var, default = env_spec.split(':', 1)
            return os.getenv(env_var, default)
        else:
            return os.getenv(env_spec, obj)
    else:
        return obj

# Load all configurations
try:
    MODEL_CONFIG = load_config('model_config')
    PROMPT_TEMPLATES = load_config('prompt_templates')
    LOGGING_CONFIG = load_config('logging_config')
except Exception as e:
    print(f"Warning: Failed to load configuration: {e}")
    MODEL_CONFIG = {}
    PROMPT_TEMPLATES = {}
    LOGGING_CONFIG = {}

__all__ = ['MODEL_CONFIG', 'PROMPT_TEMPLATES', 'LOGGING_CONFIG', 'load_config'] 