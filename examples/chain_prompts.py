"""
Prompt chaining example using the refactored Mobile Crashes RCA Agent.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.llm import get_llm_client
from src.utils import setup_logging
from src.prompt_engineering import CrashAnalysisChain

def main():
    """Prompt chaining example."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting prompt chaining example")
    
    # Get LLM client
    llm_client = get_llm_client('openai')
    
    # Create analysis chain
    chain = CrashAnalysisChain.create_analysis_chain(llm_client)
    
    # Sample crash data
    sample_crashes = [
        {
            "id": "crash001",
            "exception": "NullPointerException",
            "device": "iPhone 13",
            "os": "iOS 16.2",
            "app_version": "2.1.5",
            "frequency": 45
        },
        {
            "id": "crash002",
            "exception": "OutOfMemoryError",
            "device": "Samsung Galaxy S21",
            "os": "Android 12",
            "app_version": "2.1.4",
            "frequency": 23
        }
    ]
    
    print("🔗 Prompt Chaining Example")
    print("=" * 50)
    
    try:
        # Execute the chain
        results = chain.execute_chain({
            "crash_data": json.dumps(sample_crashes, indent=2)
        })
        
        print("📊 Chain Results:")
        print("=" * 30)
        
        for step_name, result in results.items():
            if step_name != "crash_data":  # Skip input data
                print(f"\n🔸 {step_name.replace('_', ' ').title()}:")
                print("-" * 40)
                print(result[:500] + "..." if len(result) > 500 else result)
        
        logger.info("Prompt chaining example completed successfully")
        
    except Exception as e:
        logger.error(f"Chain execution failed: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 