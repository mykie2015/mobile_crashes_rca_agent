"""Prompt chaining utilities for complex workflows."""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import logging

@dataclass
class ChainStep:
    """A single step in a prompt chain."""
    name: str
    prompt_template: str
    processor: Optional[Callable] = None  # Optional post-processing function
    dependencies: List[str] = None  # Names of steps this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class PromptChainer:
    """Chain multiple prompts together for complex workflows."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.steps: Dict[str, ChainStep] = {}
        self.results: Dict[str, Any] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_step(self, step: ChainStep):
        """Add a step to the chain."""
        self.steps[step.name] = step
        self.logger.info(f"Added step '{step.name}' to chain")
    
    def execute_chain(self, initial_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the entire prompt chain."""
        if initial_data:
            self.results.update(initial_data)
        
        executed_steps = set()
        
        # Execute steps in dependency order
        while len(executed_steps) < len(self.steps):
            progress_made = False
            
            for step_name, step in self.steps.items():
                if step_name in executed_steps:
                    continue
                
                # Check if all dependencies are satisfied
                if all(dep in executed_steps for dep in step.dependencies):
                    self._execute_step(step)
                    executed_steps.add(step_name)
                    progress_made = True
            
            if not progress_made:
                unexecuted = set(self.steps.keys()) - executed_steps
                raise RuntimeError(f"Circular dependency or missing dependency in steps: {unexecuted}")
        
        return self.results
    
    def _execute_step(self, step: ChainStep):
        """Execute a single step in the chain."""
        self.logger.info(f"Executing step: {step.name}")
        
        try:
            # Format the prompt with current results
            formatted_prompt = step.prompt_template.format(**self.results)
            
            # Execute the prompt
            if self.llm_client:
                response = self.llm_client.complete(formatted_prompt)
            else:
                self.logger.warning(f"No LLM client set, skipping execution of {step.name}")
                response = f"[Mock response for {step.name}]"
            
            # Apply post-processor if available
            if step.processor:
                response = step.processor(response)
            
            # Store the result
            self.results[step.name] = response
            self.logger.info(f"Completed step: {step.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute step {step.name}: {e}")
            raise
    
    def get_result(self, step_name: str) -> Any:
        """Get the result of a specific step."""
        return self.results.get(step_name)
    
    def clear_results(self):
        """Clear all stored results."""
        self.results.clear()

class CrashAnalysisChain:
    """Pre-built prompt chain for crash analysis workflow."""
    
    @staticmethod
    def create_analysis_chain(llm_client=None) -> PromptChainer:
        """Create a complete crash analysis prompt chain."""
        chainer = PromptChainer(llm_client)
        
        # Step 1: Pattern Analysis
        pattern_step = ChainStep(
            name="pattern_analysis",
            prompt_template="""
            Analyze the following crash data for patterns:
            
            Crash Data: {crash_data}
            
            Identify:
            1. Most frequent exception types
            2. Device patterns
            3. OS version patterns
            4. App version patterns
            
            Provide a structured analysis in JSON format.
            """,
            processor=lambda x: x.strip()
        )
        chainer.add_step(pattern_step)
        
        # Step 2: Root Cause Analysis (depends on pattern analysis)
        rca_step = ChainStep(
            name="root_cause_analysis",
            prompt_template="""
            Based on the pattern analysis:
            {pattern_analysis}
            
            Perform root cause analysis. Identify the most likely causes for each pattern.
            Focus on technical causes that can be addressed by the development team.
            """,
            dependencies=["pattern_analysis"],
            processor=lambda x: x.strip()
        )
        chainer.add_step(rca_step)
        
        # Step 3: Recommendations (depends on RCA)
        recommendations_step = ChainStep(
            name="recommendations",
            prompt_template="""
            Based on the root cause analysis:
            {root_cause_analysis}
            
            Generate specific, actionable recommendations to fix the identified issues.
            Prioritize recommendations by impact and implementation difficulty.
            """,
            dependencies=["root_cause_analysis"],
            processor=lambda x: x.strip()
        )
        chainer.add_step(recommendations_step)
        
        return chainer

# Utility functions for common post-processors
def json_processor(response: str) -> Dict[str, Any]:
    """Extract and parse JSON from response."""
    import json
    import re
    
    # Try to find JSON in the response
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, return empty dict
    return {}

def list_processor(response: str) -> List[str]:
    """Extract list items from response."""
    import re
    
    # Find numbered or bulleted lists
    items = re.findall(r'(?:^\d+\.|^[-*])\s*(.+)$', response, re.MULTILINE)
    return [item.strip() for item in items] 