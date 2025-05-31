"""
Basic completion example using the refactored Mobile Crashes RCA Agent.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.llm import get_llm_client
from src.utils import setup_logging
from src.prompt_engineering import CrashAnalysisTemplates

def main():
    """Basic completion example."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting basic completion example")
    
    # Get LLM client
    llm_client = get_llm_client('openai')
    
    # Test basic completion
    test_prompt = """
    Analyze this crash report:
    
    Exception: NullPointerException
    Device: iPhone 13
    OS: iOS 16.2
    App Version: 2.1.5
    
    Provide a brief analysis and recommendation.
    """
    
    try:
        response = llm_client.complete(test_prompt)
        print("🤖 LLM Response:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        logger.info("Basic completion example completed successfully")
        
    except Exception as e:
        logger.error(f"Completion failed: {e}")
        print(f"❌ Error: {e}")

def test_templates():
    """Test prompt templates."""
    print("\n🧪 Testing Prompt Templates:")
    print("=" * 50)
    
    # Test system prompt template
    system_template = CrashAnalysisTemplates.get_system_prompt()
    print("System Prompt:")
    print(system_template.template[:200] + "...")
    
    # Test RCA evaluation template
    rca_template = CrashAnalysisTemplates.get_rca_evaluation_prompt()
    print(f"\nRCA Template variables: {rca_template.get_variables()}")

if __name__ == "__main__":
    main()
    test_templates() 