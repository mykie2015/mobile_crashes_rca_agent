"""Few-shot learning utilities for prompt engineering."""

from typing import List, Dict, Any
import json

class FewShotExample:
    """A single few-shot example."""
    
    def __init__(self, input_data: Dict[str, Any], expected_output: str, context: str = ""):
        self.input_data = input_data
        self.expected_output = expected_output
        self.context = context
    
    def format(self, input_template: str, output_template: str = None) -> str:
        """Format the example for inclusion in a prompt."""
        formatted_input = input_template.format(**self.input_data)
        
        if output_template:
            formatted_output = output_template.format(output=self.expected_output)
        else:
            formatted_output = f"Output: {self.expected_output}"
        
        if self.context:
            return f"Context: {self.context}\nInput: {formatted_input}\n{formatted_output}"
        else:
            return f"Input: {formatted_input}\n{formatted_output}"

class FewShotExamples:
    """Collection of few-shot examples for crash analysis."""
    
    def __init__(self):
        self.examples = []
        self._load_default_examples()
    
    def add_example(self, example: FewShotExample):
        """Add a new example to the collection."""
        self.examples.append(example)
    
    def get_examples(self, num_examples: int = None) -> List[FewShotExample]:
        """Get a subset of examples."""
        if num_examples is None:
            return self.examples
        return self.examples[:num_examples]
    
    def format_for_prompt(self, num_examples: int = 3) -> str:
        """Format examples for inclusion in a prompt."""
        examples_text = []
        
        for i, example in enumerate(self.get_examples(num_examples), 1):
            example_text = f"Example {i}:\n{example.format('{crash_data}', 'Analysis: {output}')}"
            examples_text.append(example_text)
        
        return "\n\n".join(examples_text)
    
    def _load_default_examples(self):
        """Load default examples for crash analysis."""
        
        # Example 1: NullPointerException
        example1 = FewShotExample(
            input_data={
                "crash_data": json.dumps({
                    "exception": "NullPointerException",
                    "device": "iPhone 13",
                    "os": "iOS 16.2",
                    "app_version": "2.1.5",
                    "frequency": 45
                })
            },
            expected_output="High priority issue affecting iOS users. Root cause likely in null object access. Recommend implementing defensive null checks and validation.",
            context="Mobile app crash with null pointer exception"
        )
        
        # Example 2: OutOfMemoryError
        example2 = FewShotExample(
            input_data={
                "crash_data": json.dumps({
                    "exception": "OutOfMemoryError",
                    "device": "Samsung Galaxy S21",
                    "os": "Android 12",
                    "app_version": "2.1.4",
                    "frequency": 23
                })
            },
            expected_output="Memory management issue affecting Android devices. Investigate image loading, caching strategies, and memory leaks. Priority: Medium.",
            context="Android app experiencing memory issues"
        )
        
        # Example 3: Network error
        example3 = FewShotExample(
            input_data={
                "crash_data": json.dumps({
                    "exception": "NetworkOnMainThreadException",
                    "device": "Pixel 6",
                    "os": "Android 13",
                    "app_version": "2.1.5",
                    "frequency": 12
                })
            },
            expected_output="Threading violation - network operations on main thread. Move network calls to background threads. Quick fix available.",
            context="Network operation blocking main UI thread"
        )
        
        self.examples = [example1, example2, example3]

# Pre-defined example sets for different analysis types
class CrashAnalysisExamples:
    """Pre-defined example sets for crash analysis tasks."""
    
    @staticmethod
    def get_pattern_analysis_examples() -> FewShotExamples:
        """Get examples for pattern analysis tasks."""
        examples = FewShotExamples()
        # Add specific pattern analysis examples
        return examples
    
    @staticmethod
    def get_recommendation_examples() -> FewShotExamples:
        """Get examples for recommendation generation."""
        examples = FewShotExamples()
        # Add specific recommendation examples
        return examples 