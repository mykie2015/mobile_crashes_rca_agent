"""Prompt template management."""

from typing import Dict, Any, Optional
from config import PROMPT_TEMPLATES

class PromptTemplate:
    """A prompt template with variable substitution."""
    
    def __init__(self, template: str, name: str = None):
        self.template = template
        self.name = name
    
    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable for template '{self.name}': {e}")
    
    def get_variables(self) -> list:
        """Extract variable names from the template."""
        import re
        return re.findall(r'\{(\w+)\}', self.template)

def load_template(category: str, template_name: str) -> PromptTemplate:
    """Load a template from configuration."""
    try:
        template_config = PROMPT_TEMPLATES.get(category, {})
        template_str = template_config.get(template_name)
        
        if not template_str:
            raise ValueError(f"Template not found: {category}.{template_name}")
        
        return PromptTemplate(template_str, f"{category}.{template_name}")
    except Exception as e:
        raise ValueError(f"Failed to load template {category}.{template_name}: {e}")

# Pre-defined templates
class CrashAnalysisTemplates:
    """Pre-defined templates for crash analysis."""
    
    @staticmethod
    def get_system_prompt() -> PromptTemplate:
        """Get the system prompt for crash analysis."""
        return load_template('crash_analysis', 'system_prompt')
    
    @staticmethod
    def get_rca_evaluation_prompt() -> PromptTemplate:
        """Get the RCA evaluation prompt."""
        return load_template('evaluation', 'rca_evaluation_prompt')
    
    @staticmethod
    def get_rcf_evaluation_prompt() -> PromptTemplate:
        """Get the RCF evaluation prompt."""
        return load_template('evaluation', 'rcf_evaluation_prompt')
    
    @staticmethod
    def get_workflow_prompt() -> PromptTemplate:
        """Get the comprehensive workflow prompt."""
        return load_template('workflow', 'comprehensive_analysis_prompt')

def get_exception_recommendation(exception_type: str) -> str:
    """Get recommendation for specific exception type."""
    recommendations = PROMPT_TEMPLATES.get('exception_recommendations', {})
    return recommendations.get(exception_type, f"Investigate and fix {exception_type} occurrences") 